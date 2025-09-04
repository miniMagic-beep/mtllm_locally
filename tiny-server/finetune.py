# train_tinyllama.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os

def load_model(model_name: str):
    # """Load TinyLlama model with 8-bit quantization and LoRA PEFT setup."""
    # Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name,
    #     padding_side="right",
    # )
    # tokenizer.pad_token = tokenizer.eos_token

    # # Quantization config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     bnb_8bit_compute_dtype=torch.bfloat16
    # )

    # # Model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     quantization_config=bnb_config
    # )

    # # Prepare model for k-bit training
    # model.gradient_checkpointing_enable()
    # model = prepare_model_for_kbit_training(model)
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LOCAL_DIR = "./local_models/tinyllama-chat"

    # Try loading from local directory
    if os.path.exists(LOCAL_DIR):
        print("Loading model and tokenizer from local directory...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_DIR,
            device_map="auto",
            #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            torch_dtype=torch.float32
        )



    # LoRA config
    lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"],
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    )
    model_lora = get_peft_model(model, lora_cfg)
    model_lora.gradient_checkpointing_enable()
    model_lora.enable_input_require_grads()
    model_lora.config.use_cache = False
    

    print(model_lora.print_trainable_parameters())
    return model_lora, tokenizer


IGNORE_INDEX = -100

def format_dataset(data_point, tokenizer, max_len=256):
    # Build prompt components
    sys = f"{data_point['system']}\n"
    usr = f"{data_point['user']}\n"
    prefix = sys + usr # <-- optional tag to mark answer start
    ans = f"{data_point['assistant']}{tokenizer.eos_token}"

    # Tokenize separately so we can mask system+user
    tok_prefix = tokenizer(prefix, add_special_tokens=False)
    tok_answer = tokenizer(ans, add_special_tokens=False)

    input_ids = tok_prefix.input_ids + tok_answer.input_ids
    attention_mask = [1] * len(input_ids)

    # Labels: ignore system+user, train only on assistant+EOS
    labels = [IGNORE_INDEX] * len(tok_prefix.input_ids) + tok_answer.input_ids

    # Truncate if too long
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]

    # Pad if shorter
    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [IGNORE_INDEX] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
def train():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Load model and tokenizer
    model_lora, tokenizer = load_model(model_name)

    # Preprocess dataset
    dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
    dataset = dataset.map(lambda x: format_dataset(x, tokenizer), remove_columns=["system","user","assistant"])



    tmp = dataset.train_test_split(test_size=0.1)
    train_dataset = tmp["train"]
    test_dataset = tmp["test"]

    # Enable model parallelism if multiple GPUs
    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./training",
        remove_unused_columns=False,
        per_device_train_batch_size=5,
        gradient_checkpointing=True,
        gradient_accumulation_steps=5,
        #max_steps=100,
        num_train_epochs=30,
        learning_rate=2.5e-5,
        logging_steps=5,
        #fp16=True,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=5,
        do_eval=True,
        #label_names=["input_ids","attention_mask"],
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model_lora,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=None,
        args=training_args
    )

    # Train
    trainer.train()
    output_dir = "adapters/"
    model_lora.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    # Backwards-compatible: same defaults
    train()

if __name__ == "__main__":
    main()