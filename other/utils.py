import outlines
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import pprint

#from mtllm import Model

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
hf_model = AutoModelForCausalLM.from_pretrained(
MODEL_NAME,
device_map="auto",
torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

class RichContent(BaseModel):
    type: str
    text: Optional[str] = None  # only 'text' supported for now

class Message(BaseModel):
    role: str                                  # "system" | "user" | "assistant"
    content: Union[str, List[RichContent]]   

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

ol_model = outlines.from_transformers(hf_model, tokenizer)

def infer_json(prompt,schema,req):
    result = ol_model(
                prompt,
                output_type=schema,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature
            )
    return result

def train():
    print("Traning")

def eval():
    print('Eva;')