
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

# ==========================================
# CONFIGURATION
# ==========================================
TASK1_MODEL_DIR = "./results_task1/best_model"
TASK2_MODEL_DIR = "./results_task2/best_model"
TEST_FILE = "data/test.csv"
MAX_LENGTH = 512

def load_test_data():
    if os.path.exists(TEST_FILE):
        return pd.read_csv(TEST_FILE)
    else:
        # Fallback to HF
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        return dataset['test'].to_pandas()

def generate_predictions(model_dir, task_name, output_filename="prediction"):
    print(f"\n--- Generating Predictions for {task_name} ---")
    
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Skipping.")
        return

    # Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Load Data
    test_df = load_test_data()
    print(f"Loaded {len(test_df)} test samples.")
    
    # Preprocess
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(test_df['interview_question'], test_df['interview_answer'])]
    
    # Tokenize
    encodings = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).numpy()
        
    # Convert IDs to Labels
    id2label = model.config.id2label
    pred_labels = [id2label[i] for i in predictions]
    
    # Save to file (Extensionless 'prediction' file)
    with open(output_filename, "w") as f:
        for label in pred_labels:
            f.write(f"{label}\n")
            
    print(f"Saved predictions to '{output_filename}'")
    
    # Verification
    with open(output_filename, "r") as f:
        lines = f.readlines()
        print(f"Verification: File has {len(lines)} lines. (Expected: {len(test_df)})")
        print(f"First 3 predictions: {[l.strip() for l in lines[:3]]}")

if __name__ == "__main__":
    # Remove previous prediction files
    if os.path.exists("prediction_task1"): os.remove("prediction_task1")
    if os.path.exists("prediction_task2"): os.remove("prediction_task2")

    # Generate for Task 1
    generate_predictions(TASK1_MODEL_DIR, "Task 1 (Clarity)", output_filename="prediction_task1")
    
    # Generate for Task 2
    generate_predictions(TASK2_MODEL_DIR, "Task 2 (Evasion)", output_filename="prediction_task2")
    
    print("\nDONE. Ready for submission.")
