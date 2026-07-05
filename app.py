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

# Global variables for the model and tokenizer
MODEL = None
TOKENIZER = None
DEVICE = "cpu"  # Force CPU so it doesn't crash the background training

@app.on_event("startup")
async def startup_event():
    global MODEL, TOKENIZER
    print("Initializing NudiLLM for Web UI...")
    
    # Load Tokenizer
    sp_model_path = Path("tokenizer/kannada_bpe.model")
    if not sp_model_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {sp_model_path}")
    
    TOKENIZER = spm.SentencePieceProcessor()
    TOKENIZER.Load(str(sp_model_path))
    
    # Load Model
    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    MODEL, _ = load_model(str(ckpt_path), DEVICE)
    print("✅ NudiLLM is ready to serve via FastAPI!")

# Serve the HTML frontend
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Pydantic model for incoming JSON request
class GenerateRequest(BaseModel):
    prompt: str

# API Endpoint for generating text
@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")
        
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="No prompt provided.")
        
    try:
        # Format prompt for Nudi v1 (Instruction Fine-Tuned)
        formatted_prompt = f"<|user|>\n{req.prompt}\n<|ai|>\n"
        
        # Generate text using our existing function
        generated_text = generate_text(
            model=MODEL,
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
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
