
import os
# Force single GPU usage to prevent deadlocks/splitting on Kaggle T4x2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from unsloth import FastLanguageModel
import pandas as pd
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ==========================================
# CONFIGURATION
# ==========================================
# We use the Instruct version for better QA performance
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
NEW_MODEL_NAME = "llama-3-8b-clarity-v1"
OUTPUT_DIR = "./results_llama_unsloth"
MAX_SEQ_LENGTH = 2048 # Unsloth handles longer context easily
DTYPE = torch.float16 # Auto detection can fail on T4, explicit is better
LOAD_IN_4BIT = True 

# ==========================================
# SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """You are an expert political analyst. 
Your task is to analyze the following Question and Answer pair from a political interview.
Classify the Answer into one of the following Evasion categories:
- Explicit (Clear Reply)
- Dodging
- Deflection
- Claims ignorance
- Declining to answer
- Clarification
- General
- Implicit
- Partial/half-answer
"""

def format_prompts(examples):
    output_texts = []
    # Unsloth uses a specific chat template, but we can stick to standard Llama-3 format manually or use tokenizer
    for q, a, label in zip(examples['interview_question'], examples['interview_answer'], examples['evasion_label']):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: {q}\nAnswer: {a}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{label}<|eot_id|>"
        output_texts.append(text)
    return output_texts

def main():
    print("Loading Model via Unsloth...")
    # 1. Force DTYPE to float16 immediately here
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE, # Explicitly set this, don't use None/Auto on T4
        load_in_4bit = LOAD_IN_4BIT,
    )

    # Do model patching to add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none", 
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # =======================================================
    # 🔴 CRITICAL FIX FOR TESLA T4 COMPATIBILITY 🔴
    # =======================================================
    # This prevents the "BFloat16 not implemented" crash
    model.config.torch_dtype = torch.float16 
    
    # Force-cast any stubborn internal buffers (like rotary embeddings) to FP16
    for name, module in model.named_modules():
        if "norm" in name.lower():
            module.to(torch.float32) # Norm layers are safer in FP32
            
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.bfloat16:
            buffer.data = buffer.data.to(torch.float16)
    # =======================================================

    print("Loading Dataset...")
    if os.path.exists("data/train.csv"):
        df = pd.read_csv("data/train.csv")
    else:
        dataset = load_dataset("ailsntua/QEvasion")
        df = dataset['train'].to_pandas()
    
    dataset = Dataset.from_pandas(df)

    # TRL SFTTrainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "interview_question", # Must point to a real column even if ignored
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 1, # Changed to 1 to prevent Kaggle deadlocks
        formatting_func = format_prompts,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 450, # ~1 Epoch (3500 items / 8 batch_size) 
            learning_rate = 2e-4,
            fp16 = True,       # Force True for T4
            bf16 = False,      # Force False for T4
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
            report_to = "none",
        ),
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Model...")
    model.save_pretrained(NEW_MODEL_NAME) 
    print(f"Model saved to {NEW_MODEL_NAME}")
if __name__ == "__main__":
    main()