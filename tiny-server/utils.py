import outlines
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import pprint
import os

#from mtllm import Model


class RichContent(BaseModel):
    type: str
    text: Optional[str] = None  # only 'text' supported for now

class Message(BaseModel):
    role: str                                  # "system" | "user" | "assistant"
    content: Union[str, List[RichContent]]   

def flatten_content(content: Union[str, List["RichContent"], None]) -> str:
    if isinstance(content, str):
        return content

    parts = []
    if content:  # check it's not None or empty
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and c.get("text"):
                    parts.append(str(c["text"]))
            else:  # assume it's RichContent-like
                if getattr(c, "type", None) == "text" and getattr(c, "text", None):
                    parts.append(str(c.text))

    # Always return a string, even if no parts
    return "\n".join(parts) if parts else ""
  

def build_prompt(messages: List[Message]) -> str:
    prompt = ""
    for m in messages:
        text = flatten_content(m.content)
        if m.role == "system":
            prompt += f"[SYSTEM]: {text}\n"
        elif m.role == "user":
            prompt += f"[USER]: {text}\n"
   
    prompt += "[ASSISTANT]:"
    return prompt



#Saving Training Data
import os
DATA_DIR = "data/train.jsonl"
TRAIN_FILE = os.path.join(DATA_DIR)



def count_lines(path: str) -> int:
    # Ensure parent directories exist
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Create file if missing, then count lines
    with open(path, "a+", encoding="utf-8") as f:
        f.seek(0)
        return sum(1 for _ in f)

def number_of_train_data() -> int:
    lines = count_lines(TRAIN_FILE)
    return lines

def append_example(system: str, user: str, response: str):
    """Append system/user/response triple to JSONL file with tags."""
    rec = {
        "system": f"[SYSTEM]: {system}" if system else "",
        "user": f"[USER]: {user}" if user else "",
        "assistant": f"[ASSISTANT]: {response}" if response else ""
    }
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def save_the_data(messages: List[Message]):
    system = ""
    user =  ""
    assistant = ""

    for m in messages:
        text = flatten_content(m.content)
        if m.role == "system":
            system = text
        elif m.role == "user":
            user = text
        elif m.role == "assistant":
            assistant = text
    append_example(system, user, assistant)





def train():
    print("Traning")

def eval():
    print('Eva;')



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

class ModeRequest(BaseModel):
    new_mode: str

class EvalRequest(BaseModel):
    verdict: bool
        

#To save state
class State:
    def __init__(self, path: str = "state.json"):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.mode = data.get("mode", "global")
            self.eval_count = data.get("eval_count", 0)
            self.train_count = data.get("train_count", 0)
        else:
            # defaults
            self.mode = "global"
            self.eval_count = 0
            self.train_count = 0
            self.save()

    def save(self) -> dict:
        """Save current state and return it as dict."""
        data = {"mode": self.mode, "eval_count": self.eval_count, "train_count": self.train_count}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def set_mode(self, mode: str) -> dict:
        self.mode = mode
        return self.save()

    def increment_eval(self, n: int = 1) -> dict:
        self.eval_count += n
        return self.save()
    
    def increment_train(self, n: int = 1) -> dict:
        self.train_count += n
        return self.save()

    def reset_train(self) -> dict:
        """Reset training count to 0."""
        self.train_count = 0
        return self.save()
