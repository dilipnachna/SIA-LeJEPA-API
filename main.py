# main.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import torch

app = FastAPI(title="SIA-LeJEPA-API", version="1.0.0")

# Allow CORS so Render URL can be called from other hosts (safe for prototype)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

# Optional simple API key auth (set RENDER_API_KEY in Render secrets)
API_KEY = os.environ.get("RENDER_API_KEY", None)

# Attempt to import LeJEPA from installed package or local code
LEJEPA_AVAILABLE = True
try:
    from lejepa.models.lejepa_model import LeJEPA
    from lejepa.utils.tokenizer import SimpleTokenizer
except Exception as e:
    # If lejepa isn't available, set flag and provide a safe dummy fallback
    LEJEPA_AVAILABLE = False
    print("LeJEPA import failed:", e)

class Input(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "SIA-LeJEPA API running. LeJEPA available: {}".format(LEJEPA_AVAILABLE)}

@app.post("/predict")
def predict_latent(input: Input, x_api_key: str = Header(None)):
    # simple auth if API_KEY is set
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    text = input.text or ""
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    if LEJEPA_AVAILABLE:
        # load model lazily on first request if desired
        global _model, _tokenizer, _device
        try:
            _device
        except NameError:
            _device = "cpu"
        if "_model" not in globals():
            # adjust model name/path as appropriate to repo or pretrained weights
            _model = LeJEPA.from_pretrained("lejepa-small").to(_device)
            _tokenizer = SimpleTokenizer()

        tokens = _tokenizer.encode(text)
        with torch.no_grad():
            latent = _model.encode(tokens)
            consistency = float(torch.norm(latent).item())
            latent_dim = latent.shape[-1] if hasattr(latent, "shape") else len(latent)
    else:
        # Dummy fallback: simple hash-based pseudo-signal (safe for prototype)
        import hashlib, numpy as np
        b = hashlib.sha256(text.encode("utf-8")).digest()
        arr = np.frombuffer(b, dtype=np.uint8).astype(float)
        consistency = float(arr.sum() % 100) / 10.0
        latent_dim = len(arr)

    return {
        "latent_norm": consistency,
        "latent_dim": int(latent_dim),
        "lejepa_available": LEJEPA_AVAILABLE,
        "message": "latent computed (prototype)"
    }

# expose a minimal OpenAPI JSON index at /openapi.json so Custom GPT can reference it
@app.get("/openapi.json", include_in_schema=False)
def openapi_json():
    return {
        "openapi": "3.0.0",
        "info": {"title": "SIA-LeJEPA API", "version": "1.0.0"},
        "paths": {
            "/predict": {
                "post": {
                    "summary": "Compute JEPA latent",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}
                            }
                        }
                    },
                    "responses": {"200":{"description":"OK"}}
                }
            }
        }
    }
