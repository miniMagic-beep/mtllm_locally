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

def load_model(model_name: str):
    """Load TinyLlama model with 8-bit quantization and LoRA PEFT setup."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    # Prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        peft_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    print(model.print_trainable_parameters())
    return model, tokenizer


def preprocess_dataset(dataset_name: str, max_length: int = 256):
    """Load dataset and format it for causal LM training."""

    def format_dataset(data_point):
        prompt = f"""###SYSTEM: Based on INPUT title generate the prompt for generative model

###INPUT: {data_point['act']}

###PROMPT: {data_point['prompt']}
"""
        tokens = tokenizer(prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens['input_ids'].copy()
        return tokens

    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(format_dataset)
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    return train_dataset, test_dataset


def main():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

    # Load model and tokenizer
    model, tokenizer = load_model(model_name)

    # Preprocess dataset
    train_dataset, test_dataset = preprocess_dataset("fka/awesome-chatgpt-prompts")

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
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        max_steps=200,
        learning_rate=2.5e-5,
        logging_steps=5,
        fp16=True,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=5,
        do_eval=True,
        label_names=["input_ids", "labels", "attention_mask"],
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()