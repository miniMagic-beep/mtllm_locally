from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import outlines
from outlines.types import JsonSchema

# ---------------------
# Load model once
# ---------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# Outlines-wrapped for constrained decoding
ol_model = outlines.from_transformers(hf_model, tokenizer)

# ---------------------
# FastAPI setup
# ---------------------
app = FastAPI(title="TinyLlama (system+user, JSON-schema optional)")

# ---- Schemas ----
class RichContent(BaseModel):
    type: str
    text: Optional[str] = None  # only 'text' supported for now

class Message(BaseModel):
    role: str                                  # "system" or "user"
    content: Union[str, List[RichContent]]     # supports Gemini-style list

class ResponseFormat(BaseModel):
    type: str                                  # must be "json_schema"
    schema: Dict[str, Any]                     # plain JSON Schema object (required)

class ChatRequest(BaseModel):
    messages: List[Message]
    json_schema: Dict[str, Any]                # <-- renamed to avoid collision
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 128

class ChatResponse(BaseModel):
    prompt: str
    response: Any  # dict if JSON-constrained, str otherwise

# ---- Utilities ----
def flatten_content(content: Union[str, List[RichContent]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        # accept both dicts or parsed objects
        if isinstance(c, dict):
            if c.get("type") == "text" and c.get("text"):
                parts.append(c["text"])
        else:
            if c.type == "text" and c.text:
                parts.append(c.text)
    return "\n".join(parts)

def build_prompt(messages: List[Message]) -> str:
    prompt = ""
    for m in messages:
        text = flatten_content(m.content)
        if m.role == "system":
            prompt += f"[SYSTEM]: {text}\n"
        elif m.role == "user":
            prompt += f"[USER]: {text}\n"
    prompt += "[ASSISTANT]: "
    return prompt

# ---------------------
# Endpoints
# ---------------------
@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        prompt = build_prompt(req.messages)

        # Wrap client schema for Outlines
        schema = JsonSchema(req.json_schema)

        # Direct Outlines call with schema as output_type
        result = ol_model(
            prompt,
            output_type=schema,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature
        )
        print(result)
        return ChatResponse(prompt=prompt, response=result, raw=json.dumps(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))