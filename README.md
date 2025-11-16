# SIA-LeJEPA-API

A minimal FastAPI wrapper exposing a simple `/predict` endpoint for computing
latent JEPA/LeJEPA representations. Designed for use as a Custom GPT Action.

## Endpoints

### GET /
Check if the API is running.

### POST /predict
Input:
```json
{
  "text": "your message"
}
