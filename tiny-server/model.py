import outlines
from pydantic import BaseModel
from typing import Optional, Any, Dict, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch, json
from peft import PeftModel, PeftConfig
import os

import subprocess
import threading
import requests

BASE_URL = "http://localhost:7000"

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR = "./local_models/tinyllama-chat"

# Try loading from local directory
if os.path.exists(LOCAL_DIR):
    print("Loading model and tokenizer from local directory...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
else:
    print("Local copy not found. Downloading from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Save locally
    os.makedirs(LOCAL_DIR, exist_ok=True)
    tokenizer.save_pretrained(LOCAL_DIR)
    hf_model.save_pretrained(LOCAL_DIR)
    print(f"Model saved to {LOCAL_DIR}")



def load_adapters():
    global hf_model
    print("Loading LORA Adapters")
        # Check if LoRA adapters are available and switch to them if present
    try:
        # Try to load LoRA adapter if available
        lora_path = "adapters/"
        if os.path.exists(lora_path):
            peft_config = PeftConfig.from_pretrained(lora_path)
            hf_model = PeftModel.from_pretrained(hf_model, lora_path)
            print("Loaded LoRA adapter from", lora_path)
        else:
            print("Path Failed")
    except ImportError:
        print("peft library not installed; skipping LoRA adapter check.")


#This will do the inference
def infer_json(prompt,schema,req):
    #print(prompt)
    result = ol_model(
                prompt,
                output_type=schema,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature
            )
    print(result)
    return result

def infer_text(prompt,req):
    result = ol_model(
        prompt,
        max_new_tokens=100,
        temperature=req.temperature,
        eos_token_id=tokenizer.eos_token_id
    )
    print(result)
    return result



load_adapters()
ol_model = outlines.from_transformers(hf_model, tokenizer)

def after_training():
    load_adapters()
    print("Adapters loaded")
    # Send a POST request to BASE_URL/mode to set mode to "eval"
    try:
        response = requests.post(f"{BASE_URL}/mode", json={"new_mode": "eval"})
        if response.status_code == 200:
            print("Mode set to eval:", response.json())
        else:
            print("Failed to set mode:", response.text)
    except Exception as e:
        print("Error sending mode request:", e)



def train():
    process = subprocess.Popen(["python", "finetune.py"])

    def monitor():
        process.wait()   # wait until training finishes
        after_training()

    threading.Thread(target=monitor, daemon=True).start()

