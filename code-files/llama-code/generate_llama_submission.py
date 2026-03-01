
import os
import torch
import pandas as pd
from unsloth import FastLanguageModel
from transformers import TextStreamer
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "llama-3-8b-clarity-v1" # Path where finetune_llama.py saved the model
TEST_FILE = "data/test.csv"
MAX_SEQ_LENGTH = 2048
dtype = None
load_in_4bit = True

# MAPPING (Evasion -> Clarity)
EVASION_TO_CLARITY = {
    'Explicit': 'Clear Reply',
    'Claims ignorance': 'Clear Non-Reply',
    'Clarification': 'Clear Non-Reply',
    'Declining to answer': 'Clear Non-Reply',
    'Deflection': 'Ambivalent',
    'Dodging': 'Ambivalent',
    'General': 'Ambivalent',
    'Implicit': 'Ambivalent',
    'Partial/half-answer': 'Ambivalent'
}

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

def format_prompt_for_inference(q, a):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nQuestion: {q}\nAnswer: {a}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def main():
    print(f"Loading Model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} does not exist. Did training finish?")
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    print("Loading Test Data...")
    if os.path.exists(TEST_FILE):
        df = pd.read_csv(TEST_FILE)
    else:
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        df = dataset['test'].to_pandas()
    
    print(f"Processing {len(df)} samples...")
    
    task2_predictions = []
    task1_predictions = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        prompt = format_prompt_for_inference(row['interview_question'], row['interview_answer'])
        inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

        # Generate
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 128, # Increased to allow full label generation
            use_cache = True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode
        generated_text = tokenizer.batch_decode(outputs)[0]
        
        # Extract the assistant's response (after the last header)
        response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").strip()
        
        if i < 5:
            print(f"\n--- DEBUG SAMPLE {i} ---")
            print(f"Prop: {row['interview_question'][:50]}...")
            print(f"Raw Output: '{response}'")
        
        # Clean up weird generation artifacts if any
        # We match against known labels to be safe
        predicted_label = "Dodging" # Valid Task 2 Fallback (Ambivalent is Task 1!)
        for known_label in EVASION_TO_CLARITY.keys():
            if known_label.lower() in response.lower():
                predicted_label = known_label
                break
        
        task2_predictions.append(predicted_label)
        
        # Hierarchical Mapping
        task1_predictions.append(EVASION_TO_CLARITY.get(predicted_label, "Ambivalent"))

    # Save Prediction Files
    with open("prediction_llama_task2", "w") as f:
        for p in task2_predictions: f.write(f"{p}\n")
        
    with open("prediction_llama_task1_hierarchical", "w") as f:
        for p in task1_predictions: f.write(f"{p}\n")
        
    print("Saved predictions: 'prediction_llama_task2' and 'prediction_llama_task1_hierarchical'")

if __name__ == "__main__":
    main()
