
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# CONFIGURATION
# ==========================================
# We use the TASK 2 (Evasion) model to predict Task 1
TASK2_MODEL_DIR = "./results_task2/best_model" 
TEST_FILE = "data/test.csv"
MAX_LENGTH = 512

# MAPPING (Derived from training data)
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

def load_test_data():
    if os.path.exists(TEST_FILE):
        return pd.read_csv(TEST_FILE)
    else:
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        return dataset['test'].to_pandas()

def generate_hierarchical_prediction(model_dir, output_filename="prediction_task1_hierarchical"):
    print(f"\n--- Generating Hierarchical Predictions (Task 2 -> Task 1) ---")
    
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Warning: Predictions will fail if model is missing.")
    
    # Check if we can load model (might be running in a fresh env without the model trained yet)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        return

    id2label = model.config.id2label
    
    # Load Data
    test_df = load_test_data()
    print(f"Loaded {len(test_df)} test samples.")
    
    # Preprocess
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(test_df['interview_question'], test_df['interview_answer'])]
    
    # Tokenize
    encodings = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    
    # Predict (Evasion Labels)
    model.eval()
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).numpy()
        
    # Mapping Logic
    final_clarity_labels = []
    
    for pred_id in predictions:
        evasion_label = id2label[pred_id]
        if evasion_label in EVASION_TO_CLARITY:
            clarity_label = EVASION_TO_CLARITY[evasion_label]
        else:
            print(f"WARNING: Unknown evasion label '{evasion_label}'. Defaulting to 'Ambivalent'.")
            clarity_label = "Ambivalent"
        final_clarity_labels.append(clarity_label)
    
    # Save to file
    with open(output_filename, "w") as f:
        for label in final_clarity_labels:
            f.write(f"{label}\n")
            
    print(f"Saved hierarchical predictions to '{output_filename}'")
    print(f"First 3: {final_clarity_labels[:3]}")

if __name__ == "__main__":
    generate_hierarchical_prediction(TASK2_MODEL_DIR, output_filename="prediction_task1_hierarchical")
