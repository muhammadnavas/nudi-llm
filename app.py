import os
import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sentencepiece as spm
import uvicorn

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import our inference functions
from inference import load_model, generate_text

# Initialize FastAPI
app = FastAPI(title="NudiLLM API")

# Mount static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for the models and tokenizer
MODEL_CHAT = None
MODEL_BASE = None
TOKENIZER = None
DEVICE = "cpu"  # Force CPU so it doesn't crash the background training

@app.on_event("startup")
async def startup_event():
    global MODEL_CHAT, MODEL_BASE, TOKENIZER
    print("Initializing NudiLLM for Web UI...")
    
    # Load Tokenizer
    sp_model_path = Path("tokenizer/kannada_bpe.model")
    if not sp_model_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {sp_model_path}")
    
    TOKENIZER = spm.SentencePieceProcessor()
    TOKENIZER.Load(str(sp_model_path))
    
    # Load Chat Model
    ckpt_chat_path = Path("checkpoints/best.pt")
    if ckpt_chat_path.exists():
        MODEL_CHAT, _ = load_model(str(ckpt_chat_path), DEVICE)
    
    # Load Base Model
    ckpt_base_path = Path("checkpoints/nudi_v0_best.pt")
    if ckpt_base_path.exists():
        MODEL_BASE, _ = load_model(str(ckpt_base_path), DEVICE)
        
    print("✅ NudiLLM is ready to serve via FastAPI!")

# Serve the HTML frontend
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

# Pydantic model for incoming JSON request
class GenerateRequest(BaseModel):
    prompt: str
    use_chat_format: bool = True

# API Endpoint for generating text
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if (MODEL_CHAT is None and req.use_chat_format) or (MODEL_BASE is None and not req.use_chat_format) or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Requested model not loaded properly.")
        
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="No prompt provided.")
        
    try:
        # Format prompt based on mode and select model
        if req.use_chat_format:
            formatted_prompt = f"<|user|>\n{req.prompt}\n<|ai|>\n"
            active_model = MODEL_CHAT
        else:
            formatted_prompt = req.prompt
            active_model = MODEL_BASE
        
        # Generate text using our existing function
        generated_text = generate_text(
            model=active_model,
            sp=TOKENIZER,
            prompt=formatted_prompt,
            device=DEVICE,
            max_new_tokens=150,
            temperature=0.3,
            top_k=50,
            top_p=0.9
        )
        return {"result": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
