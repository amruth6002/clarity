
import os
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import torch.nn.functional as F
import glob
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# CRITICAL: Force Single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# CONFIGURATION
# ==========================================
KFOLD_DIR = "./results_kfold"
TEST_FILE = "data/test.csv"
MAX_LENGTH = 512
N_FOLDS = 5

# ==========================================
# MODEL STRUCTURE
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

def get_best_model_path(fold_dir):
    """
    Finds the best model path. 
    Priority 1: 'best_model' folder
    Priority 2: The latest 'checkpoint-XXX' folder
    """
    # 1. Check for best_model
    best_path = os.path.join(fold_dir, "best_model")
    if os.path.exists(best_path):
        return best_path
    
    # 2. Check for checkpoints
    checkpoints = glob.glob(os.path.join(fold_dir, "checkpoint-*"))
    if checkpoints:
        # Sort by number (checkpoint-100, checkpoint-200...)
        try:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"  WARNING: 'best_model' not found. Using latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        except:
            return checkpoints[-1] # Fallback
            
    return None

def main():
    if not os.path.exists(KFOLD_DIR):
        print(f"Error: K-Fold directory {KFOLD_DIR} not found.")
        return

    # 1. Load Config (Try Fold 0, then 1, etc.)
    config = None
    for i in range(N_FOLDS):
        path = f"{KFOLD_DIR}/fold_{i}/best_model/labels_config.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                config = json.load(f)
            break
            
    if config is None:
        print("Error: Could not find `labels_config.json` in any fold. Using Hardcoded defaults.")
        # Hardcoded Fallback
        l2id_t1 = {'Ambivalent': 0, 'Clear Non-Reply': 1, 'Clear Reply': 2}
        l2id_t2 = {'Claims ignorance': 0, 'Clarification': 1, 'Declining to answer': 2, 'Deflection': 3, 'Dodging': 4, 'Explicit': 5, 'General': 6, 'Implicit': 7, 'Partial/half-answer': 8}
    else:
        l2id_t1 = config["task1"]
        l2id_t2 = config["task2"]
        
    id2l_t1 = {i: l for l, i in l2id_t1.items()}
    id2l_t2 = {i: l for l, i in l2id_t2.items()}
    
    # 2. Prepare Data
    BASE_MODEL_NAME = "microsoft/deberta-v3-large" 
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    test_df = load_test_data()
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(test_df['interview_question'], test_df['interview_answer'])]
    
    print("Tokenizing...")
    encodings = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    # 3. Ensemble Inference
    num_samples = len(inputs)
    avg_probs_t1 = np.zeros((num_samples, len(l2id_t1)))
    avg_probs_t2 = np.zeros((num_samples, len(l2id_t2)))
    
    BATCH_SIZE = 8
    models_used = 0
    
    print(f"Starting Ensemble Inference for {N_FOLDS} Folds...")
    
    for fold in range(N_FOLDS):
        fold_dir = f"{KFOLD_DIR}/fold_{fold}"
        model_path = get_best_model_path(fold_dir)
        
        if model_path is None:
            print(f"SKIPPING FOLD {fold}: No model found in {fold_dir}")
            continue
            
        print(f"Loading Fold {fold}: {model_path}")
        
        model = MultiTaskModel(BASE_MODEL_NAME, len(l2id_t1), len(l2id_t2))
        
        # Load Weights
        weights_path = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(weights_path):
                 print(f"  Error: Weights missing in {model_path}. Skipping.")
                 continue
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu")
            
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        fold_probs_t1 = []
        fold_probs_t2 = []
        
        with torch.no_grad():
            for i in range(0, num_samples, BATCH_SIZE):
                batch = {k: v[i:i+BATCH_SIZE] for k, v in encodings.items()}
                logits1, logits2 = model(**batch)
                
                p1 = F.softmax(logits1, dim=-1).cpu().numpy()
                p2 = F.softmax(logits2, dim=-1).cpu().numpy()
                
                fold_probs_t1.append(p1)
                fold_probs_t2.append(p2)
                
        # Aggregate (Store sum, divide later)
        if len(fold_probs_t1) > 0:
            avg_probs_t1 += np.concatenate(fold_probs_t1, axis=0)
            avg_probs_t2 += np.concatenate(fold_probs_t2, axis=0)
            models_used += 1
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    if models_used == 0:
        print("CRITICAL ERROR: No models were successfully loaded!")
        return

    print(f"Inference Complete. Averaging over {models_used} models...")
    avg_probs_t1 /= models_used
    avg_probs_t2 /= models_used
    
    # 4. Argmax
    final_preds_t1 = np.argmax(avg_probs_t1, axis=1)
    final_preds_t2 = np.argmax(avg_probs_t2, axis=1)
    
    # 5. Save
    with open("prediction_kfold_task1", "w") as f:
        for p in final_preds_t1: f.write(f"{id2l_t1[p]}\n")
        
    with open("prediction_kfold_task2", "w") as f:
        for p in final_preds_t2: f.write(f"{id2l_t2[p]}\n")
        
    print("Saved ensemble predictions: prediction_kfold_task1, prediction_kfold_task2")

if __name__ == "__main__":
    main()
