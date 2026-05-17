import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class CompletionPayload(BaseModel):
    prompt: str

@app.post("/v1/completions")
async def handle_evaluation(payload: CompletionPayload):
    print(f"\n[NETWORK TRAFFIC INBOUND]: {payload.prompt}")
    text = payload.prompt.lower()
    
    if "bypass" in text or "override" in text:
        mock_response = '{"safe": false, "policy": "Category_02", "reason": "Jailbreak sequence detected via manual override string."}'
    else:
        mock_response = '{"safe": true, "policy": "None", "reason": "Text string cleared safety boundaries."}'

    return {"choices": [{"text": mock_response}]}

if __name__ == "__main__":
    # Binding to 0.0.0.0 allows the Windows host to bridge over to the WSL network card smoothly
    uvicorn.run(app, host="0.0.0.0", port=8000)
