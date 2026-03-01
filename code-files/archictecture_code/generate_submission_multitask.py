
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from transformers import AutoModel, AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "results_multitask/best_model"
TEST_FILE = "data/test.csv"
MAX_LENGTH = 512

# ==========================================
# CUSTOM MODEL (Must match training)
# ==========================================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.task1_head = nn.Linear(hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits_task1 = self.task1_head(cls_output)
        logits_task2 = self.task2_head(cls_output)
        
        return logits_task1, logits_task2

def load_test_data():
    if os.path.exists(TEST_FILE):
        return pd.read_csv(TEST_FILE)
    else:
        print("Downloading from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        return dataset['test'].to_pandas()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found.")
        return

    # 1. Load Config (Labels)
    with open(f"{MODEL_PATH}/labels_config.json", "r") as f:
        config = json.load(f)
        l2id_t1 = config["task1"]
        l2id_t2 = config["task2"]
        
    id2l_t1 = {i: l for l, i in l2id_t1.items()}
    id2l_t2 = {i: l for l, i in l2id_t2.items()}
    
    # 2. Load Model
    # Note: We need to initialize the class with the SAME base model name as training
    # But since we are loading weights, we can initialize with base, then load state_dict
    print("Initializing Model Structure...")
    # Assuming base model name available or hardcoded. Ideally saved in config. 
    # For now, hardcoding as per training script.
    BASE_MODEL_NAME = "microsoft/deberta-v3-large" 
    
    model = MultiTaskModel(BASE_MODEL_NAME, len(l2id_t1), len(l2id_t2))
    
    print(f"Loading Weights from {MODEL_PATH}...")
    # Transformers `save_model` typically saves `pytorch_model.bin` or `model.safetensors`
    # and config.json. Since we used `trainer.save_model()`, it saved the state dict.
    # However, since `MultiTaskModel` is a custom nn.Module, Trainer might have saved it properly
    # if it detected it. Let's assume standard pytorch loading.
    
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_PATH, "model.safetensors")
        if not os.path.exists(weights_path):
             print("Could not find weights file (pytorch_model.bin or model.safetensors)")
             return
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        
    model.load_state_dict(state_dict)
    
    # 3. Predict
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    test_df = load_test_data()
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(test_df['interview_question'], test_df['interview_answer'])]
    
    print("Tokenizing...")
    encodings = tokenizer(inputs, max_length=512, truncation=True, padding=True, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference on {device}...")
    model.to(device)
    model.eval()
    
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    preds_t1 = []
    preds_t2 = []
    
    BATCH_SIZE = 8
    num_samples = len(inputs)
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch = {k: v[i:i+BATCH_SIZE] for k, v in encodings.items()}
            logits1, logits2 = model(**batch)
            
            p1 = torch.argmax(logits1, dim=-1).cpu().numpy()
            p2 = torch.argmax(logits2, dim=-1).cpu().numpy()
            
            preds_t1.extend([id2l_t1[x] for x in p1])
            preds_t2.extend([id2l_t2[x] for x in p2])
            
    # Save
    with open("prediction_multitask_task1", "w") as f:
        for p in preds_t1: f.write(f"{p}\n")
        
    with open("prediction_multitask_task2", "w") as f:
        for p in preds_t2: f.write(f"{p}\n")
        
    print("Saved predictions: prediction_multitask_task1, prediction_multitask_task2")

if __name__ == "__main__":
    main()
