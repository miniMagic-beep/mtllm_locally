from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import outlines
from outlines.types import JsonSchema
import pprint

# ---------------------
# Load model once (unchanged)
# ---------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
ol_model = outlines.from_transformers(hf_model, tokenizer)

# ---------------------
# FastAPI setup
# ---------------------
app = FastAPI(title="TinyLlama (OpenAI-compatible shim)")
is_local = True
traing_data_count = 0
# ---- Existing helpers (unchanged) ----
class RichContent(BaseModel):
    type: str
    text: Optional[str] = None  # only 'text' supported for now

class Message(BaseModel):
    role: str                                  # "system" | "user" | "assistant"
    content: Union[str, List[RichContent]]     # supports list parts (text only here)

def flatten_content(content: Union[str, List[RichContent]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        if isinstance(c, dict):
            if c.get("type") == "text" and c.get("text"):
                parts.append(c["text"])
        else:
            if c.type == "text" and c.text:
                parts.append(c.text)
    return "\n".join(parts)

def build_prompt(messages: List[Message]) -> str:
    # Optionally print the full message list for debugging
    pprint.pprint(messages)
    prompt = ""
    for m in messages:
        text = flatten_content(m.content)
        if m.role == "system":
            prompt += f"[SYSTEM]: {text}\n"
        elif m.role == "user":
            prompt += f"[USER]: {text}\n"
        # elif m.role == "assistant":
        #     prompt += f"[ASSISTANT]: {text}\n"
    
    prompt += "[ASSISTANT]: "
    return prompt


#Saving Training Data
import os
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")

def append_example(system: str, user: str, response: str):
    """Append system/user/response triple to JSONL file with tags."""
    rec = {
        "system": f"[SYSTEM]: {system}" if system else "",
        "user": f"[USER]: {user}" if user else "",
        "response": f"[ASSISTANT]: {response}" if response else ""
    }
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")



def save_the_data(messages: List[Union[dict, object]]):
    """
    Extract last system, last user, and last assistant.
    Save as {system, user, response} with [SYSTEM], [USER], [ASSISTANT] tags included.
    """
    if not messages:
        return
    last = messages[-1]
    role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
    if role != "assistant":
        return

    # Helper to extract content
    def content_of(m):
        return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")

    # Last system
    system_msg = next(
        (content_of(m) for m in reversed(messages)
         if (m.get("role") if isinstance(m, dict) else getattr(m,"role",None)) == "system"),
        ""
    )
    # Last user
    user_msg = next(
        (content_of(m) for m in reversed(messages)
         if (m.get("role") if isinstance(m, dict) else getattr(m,"role",None)) == "user"),
        ""
    )
    # Last assistant (the response we want)
    response_msg = content_of(last)
    append_example(system_msg, user_msg, response_msg)


# ---- OpenAI-compatible minimal models ----
class JSONSchemaWrapper(BaseModel):
    name: Optional[str] = None
    schema: Dict[str, Any]
    strict: Optional[bool] = True

class ResponseFormat(BaseModel):
    type: str                                    # "json_schema" or "text"
    json_schema: Optional[JSONSchemaWrapper] = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[RichContent]]

class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 128
    response_format: Optional[ResponseFormat] = None
    tools: Optional[List[Dict[str, Any]]] = None  # ignored in this minimal shim
    stream: Optional[bool] = False                # ignored (no streaming here)

class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionsResponse(BaseModel):
    id: str = "chatcmpl-local-1"
    object: str = "chat.completion"
    created: int = 0
    model: str = "local"
    choices: List[Choice]

def train_and_eval_local_model():
    print("Eval")
    return True
# ---------------------
# Endpoints
# ---------------------
@app.get("/healthz")
def health():
    return {"status": "ok"}

#This tell which model to use
@app.get("/which")
def which():
    return {"is_local": is_local}

# OpenAI-compatible route (minimal changes from your original)
@app.post("/chat/completions", response_model=ChatCompletionsResponse)
def chat(req: ChatCompletionsRequest):
    global is_local
    if not(is_local):
        save_the_data(req.messages)
        return ChatCompletionsResponse(
            choices=[Choice(index=0, message=ChoiceMessage(content=req.messages[-1].content))]
        )
        training_data_count += 1
        #Eval Local_model_after_100
        if training_data_count > 100:
            if train_and_eval_local_model():
                is_local = True

    try:
        # reuse your prompt builder
        # (Pydantic coercion lets us pass ChatMessage as Message)
        prompt = build_prompt([Message(**m.model_dump()) for m in req.messages])
        # Decide whether to constrain with JSON schema
        
        
        use_schema = (
            req.response_format
            and req.response_format.type == "json_schema"
            and req.response_format.json_schema
            and req.response_format.json_schema.schema
        )

        if use_schema:
            schema = JsonSchema(req.response_format.json_schema.schema)
            result = ol_model(
                prompt,
                output_type=schema,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature
            )
            #print(result)
            # Result can be dict; normalize to string for OpenAI content
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        else:
            # plain text generation
            result = ol_model(
                prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature
            )
            content = str(result)

        return ChatCompletionsResponse(
            choices=[Choice(index=0, message=ChoiceMessage(content=content))]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def count_examples() -> int:
    """Return number of stored examples"""
    if not os.path.exists(TRAIN_FILE):
        return 0
    return sum(1 for _ in open(TRAIN_FILE, "r", encoding="utf-8"))

class CollectItem(BaseModel):
    messages: List[ChatMessage]        # same shape as your /chat message array
    assistant: str                     # ground-truth desired answer

