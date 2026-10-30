import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inference import load_model, generate_text

app = FastAPI(title="NudiLLM API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_CHAT = None
MODEL_BASE = None
TOKENIZER = None
DEVICE = "cpu"  # Keep web inference isolated from background training.


@app.on_event("startup")
async def startup_event():
    global MODEL_CHAT, MODEL_BASE, TOKENIZER
    print("Initializing NudiLLM for Web UI...")

    sp_model_path = Path("tokenizer/kannada_bpe.model")
    if not sp_model_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {sp_model_path}")

    TOKENIZER = spm.SentencePieceProcessor()
    TOKENIZER.Load(str(sp_model_path))

    ckpt_chat_path = Path("checkpoints/best.pt")
    if ckpt_chat_path.exists():
        MODEL_CHAT, _ = load_model(str(ckpt_chat_path), DEVICE)

    ckpt_base_path = Path("checkpoints/nudi_v0_best.pt")
    if ckpt_base_path.exists():
        MODEL_BASE, _ = load_model(str(ckpt_base_path), DEVICE)

    print("NudiLLM is ready to serve via FastAPI!")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"request": request}
    )


@app.get("/visualize")
async def visualize_page(request: Request):
    return templates.TemplateResponse(
        request=request, name="visualize.html", context={"request": request}
    )


class GenerateRequest(BaseModel):
    prompt: str
    use_chat_format: bool = True


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if (MODEL_CHAT is None and req.use_chat_format) or (
        MODEL_BASE is None and not req.use_chat_format
    ) or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Requested model not loaded properly.")

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        if req.use_chat_format:
            formatted_prompt = f"<|user|>\n{req.prompt}\n<|ai|>\n"
            active_model = MODEL_CHAT
        else:
            formatted_prompt = req.prompt
            active_model = MODEL_BASE

        generated_text = generate_text(
            model=active_model,
            sp=TOKENIZER,
            prompt=formatted_prompt,
            device=DEVICE,
            max_new_tokens=150,
            temperature=0.3,
            top_k=50,
            top_p=0.9,
        )
        return {"result": generated_text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class VisualizeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    use_chat_format: bool = True
    max_new_tokens: int = Field(default=32, ge=1, le=80)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=20, ge=0, le=100)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.3, ge=1.0, le=2.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


def _token_details(token_id: int):
    return {
        "id": int(token_id),
        "piece": TOKENIZER.id_to_piece(int(token_id)),
        "text": TOKENIZER.decode([int(token_id)]),
    }


@app.post("/api/visualize")
async def visualize_generation(req: VisualizeRequest):
    """Return an auditable, token-by-token trace from the real NudiLLM model."""
    if TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Tokenizer is not loaded.")

    model = MODEL_CHAT if req.use_chat_format else MODEL_BASE
    if model is None:
        mode = "chat" if req.use_chat_format else "base"
        raise HTTPException(status_code=503, detail=f"The {mode} checkpoint is not loaded.")

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    formatted_prompt = (
        f"<|user|>\n{req.prompt}\n<|ai|>\n"
        if req.use_chat_format else req.prompt
    )
    prompt_ids = TOKENIZER.encode(formatted_prompt)
    if not prompt_ids:
        prompt_ids = [TOKENIZER.bos_id()]

    context_length = model.config.context_length
    if len(prompt_ids) > context_length:
        prompt_ids = prompt_ids[-context_length:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    visible_prompt_tokens = [_token_details(token_id) for token_id in prompt_ids]
    steps = []
    generated_ids = []

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device=DEVICE)
        generator.manual_seed(req.seed)

    model.eval()
    try:
        with torch.inference_mode():
            for index in range(req.max_new_tokens):
                context = input_ids[:, -context_length:]
                logits, _ = model(context)
                next_logits = logits[:, -1, :].float()

                # Match the model's repetition-penalty behavior.
                if req.repetition_penalty != 1.0:
                    for seen_id in set(input_ids[0].tolist()):
                        if next_logits[0, seen_id] > 0:
                            next_logits[0, seen_id] /= req.repetition_penalty
                        else:
                            next_logits[0, seen_id] *= req.repetition_penalty

                next_logits = next_logits / req.temperature

                if req.top_k > 0:
                    threshold = torch.topk(
                        next_logits, min(req.top_k, next_logits.size(-1))
                    ).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                if req.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumulative_probs - sorted_probs >= req.top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    filtered_logits = torch.full_like(next_logits, float("-inf"))
                    next_logits = filtered_logits.scatter(1, sorted_indices, sorted_logits)

                probabilities = F.softmax(next_logits, dim=-1)
                candidate_count = min(5, probabilities.size(-1))
                top_probs, top_ids = torch.topk(probabilities[0], candidate_count)

                sampled = torch.multinomial(probabilities, num_samples=1, generator=generator)
                token_id = int(sampled.item())
                detail = _token_details(token_id)
                candidate_list = [
                    {
                        **_token_details(int(candidate_id)),
                        "probability": float(probability),
                    }
                    for probability, candidate_id in zip(
                        top_probs.tolist(), top_ids.tolist()
                    )
                ]

                steps.append({
                    "index": index + 1,
                    "context_tokens": int(context.shape[1]),
                    "candidates": candidate_list,
                    "selected": detail,
                    "is_eos": token_id == model.config.eos_id,
                })

                generated_ids.append(token_id)
                input_ids = torch.cat([input_ids, sampled], dim=1)
                if token_id == model.config.eos_id:
                    break
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Visualization generation failed: {exc}")

    config = model.config
    return {
        "prompt": req.prompt,
        "formatted_prompt": formatted_prompt,
        "mode": "chat" if req.use_chat_format else "base",
        "tokenizer": {
            "vocab_size": TOKENIZER.get_piece_size(),
            "prompt_tokens": visible_prompt_tokens,
            "prompt_token_count": len(prompt_ids),
        },
        "model": {
            "name": "NudiLLM",
            "parameters": model.count_parameters(),
            "layers": config.n_layers,
            "attention_heads": config.n_heads,
            "embedding_dimension": config.n_embd,
            "context_length": config.context_length,
            "vocabulary_size": config.vocab_size,
            "architecture": "Decoder-only Transformer",
        },
        "sampling": {
            "temperature": req.temperature,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "repetition_penalty": req.repetition_penalty,
        },
        "steps": steps,
        "generated_token_ids": generated_ids,
        "generated_text": TOKENIZER.decode(generated_ids),
        "stopped_on_eos": bool(steps and steps[-1]["is_eos"]),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
