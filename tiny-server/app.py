from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json
import outlines
from outlines.types import JsonSchema
import pprint

from utils import *

from model import infer_json,infer_text,train


# ---------------------
# Load model once (unchanged)
# ---------------------


# ---------------------
# FastAPI setup
# ---------------------
app = FastAPI(title="TinyLlama (OpenAI-compatible shim)")


#Gets the Number of Records Present
training_data_count = number_of_train_data()
# ---- Existing helpers (unchanged) ----
state = State()

    







# def append_example(system: str, user: str, response: str):
#     """Append system/user/response triple to JSONL file with tags."""
#     rec = {
#         "system": f"[SYSTEM]: {system}" if system else "",
#         "user": f"[USER]: {user}" if user else "",
#         "response": f"[ASSISTANT]: {response}" if response else ""
#     }
#     with open(TRAIN_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(rec, ensure_ascii=False) + "\n")



# def save_the_data(messages: List[Union[dict, object]]):
#     """
#     Extract last system, last user, and last assistant.
#     Save as {system, user, response} with [SYSTEM], [USER], [ASSISTANT] tags included.
#     """
#     if not messages:
#         return
#     last = messages[-1]
#     role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
#     if role != "assistant":
#         return

#     # Helper to extract content
#     def content_of(m):
#         return m.get("content") if isinstance(m, dict) else getattr(m, "content", "")

#     # Last system
#     system_msg = next(
#         (content_of(m) for m in reversed(messages)
#          if (m.get("role") if isinstance(m, dict) else getattr(m,"role",None)) == "system"),
#         ""
#     )
#     # Last user
#     user_msg = next(
#         (content_of(m) for m in reversed(messages)
#          if (m.get("role") if isinstance(m, dict) else getattr(m,"role",None)) == "user"),
#         ""
#     )
#     # Last assistant (the response we want)
#     response_msg = content_of(last)
#     append_example(system_msg, user_msg, response_msg)




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
    return {"mode": state.mode}

#To change the mode after traning
@app.post("/mode")
def set_mode(req: ModeRequest):
    global mode
    mode = req.new_mode
    #Save the state
    state.change_mode(mode)
    return {"mode": mode}

@app.post("/eval")
def set_mode(req: EvalRequest):
    if req.verdict:   # only increment if true
        state.increment_eval()
        if state.eval_count>10:
            state.change_mode("local")
    else:
        state.eval_count = 0
        state.mode = "idle"
        state.save()
    
    return {"eval_count": state.eval_count}

# OpenAI-compatible route (minimal changes from your original)
@app.post("/chat/completions", response_model=ChatCompletionsResponse)
def chat(req: ChatCompletionsRequest):
    global state
    global training_data_count
    if state.mode == "global":
        save_the_data([Message(**m.model_dump()) for m in req.messages])
        training_data_count += 1
        if training_data_count > 100:
            train()
            state.change_mode("idle")
        return ChatCompletionsResponse(
    choices=[Choice(index=0, message=ChoiceMessage(content=req.messages[-1].content))]
)

    try:
        # reuse your prompt builder
        # (Pydantic coercion lets us pass ChatMessage as Message)
        prompt = build_prompt([Message(**m.model_dump()) for m in req.messages])
        # Decide whether to constrain with JSON schema
        print(prompt)
        
        use_schema = (
            req.response_format
            and req.response_format.type == "json_schema"
            and req.response_format.json_schema
            and req.response_format.json_schema.schema
        )

        if use_schema:
            schema = JsonSchema(req.response_format.json_schema.schema)
            result = infer_json(prompt, schema, req)
            #print(result)
            # Result can be dict; normalize to string for OpenAI content
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        else:
            # plain text generation
            result = infer_text(prompt,req)
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

