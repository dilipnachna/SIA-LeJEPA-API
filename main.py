from fastapi import FastAPI
from pydantic import BaseModel
import torch

# --------------------------------------
# NOTE:
# This assumes you installed the lejepa package
# or the model code is available in your environment.
# If the import path differs, adjust accordingly.
# --------------------------------------

try:
    from lejepa.models.lejepa_model import LeJEPA
    from lejepa.utils.tokenizer import SimpleTokenizer
except ImportError:
    raise ImportError("LeJEPA package not found. Make sure it's installed or available in the repo.")

app = FastAPI(
    title="SIA-LeJEPA API",
    description="A minimal API wrapper for JEPA/LeJEPA latent reasoning.",
    version="1.0.0"
)

# Load model once at startup
device = "cpu"
model = LeJEPA.from_pretrained("lejepa-small").to(device)
tokenizer = SimpleTokenizer()

class Input(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "SIA-LeJEPA API is running."}

@app.post("/predict")
def predict_latent(input: Input):
    tokens = tokenizer.encode(input.text)

    with torch.no_grad():
        latent = model.encode(tokens)
        # Example signal: L2 norm = "consistency"
        consistency = float(torch.norm(latent).item())
    
    return {
        "latent_norm": consistency,
        "latent_dim": latent.shape[-1] if hasattr(latent, "shape") else len(latent),
        "safe": True,
        "message": "LeJEPA latent computed successfully"
    }
