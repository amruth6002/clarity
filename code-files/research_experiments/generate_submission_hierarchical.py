
import os
import torch
import torch.nn as nn
import pandas as pd
import json
from transformers import AutoModel, AutoTokenizer

# CRITICAL: Force Single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "results_hierarchical/best_model"
TEST_FILE = "data/test.csv"
MAX_LENGTH = 256 # Must match training

# ==========================================
# CUSTOM ARCHITECTURE (Must match training!)
# ==========================================
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key, value, key_padding_mask=None):
        if key_padding_mask is not None:
             key_padding_mask = (key_padding_mask == 0)
        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return self.layer_norm(query + attn_output)

class HierarchicalBiEncoder(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        self.cross_attn = CrossAttentionLayer(self.hidden_size)
        
        self.task1_head = nn.Linear(self.hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(self.hidden_size, num_labels_task2)

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a):
        out_q = self.encoder(input_ids=input_ids_q, attention_mask=attention_mask_q)
        embed_q = out_q.last_hidden_state 
        
        out_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        embed_a = out_a.last_hidden_state 
        
        interaction = self.cross_attn(query=embed_q, key=embed_a, value=embed_a, key_padding_mask=attention_mask_a)
        
        mask_expanded = attention_mask_q.unsqueeze(-1).expand(interaction.size()).float()
        sum_embeddings = torch.sum(interaction * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits_task1 = self.task1_head(pooled_output)
        logits_task2 = self.task2_head(pooled_output)
        
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
        print("Run 'python code-files/research_experiments/train_hierarchical.py' first.")
        return

    # 1. Load Config (Labels)
    config_path = os.path.join(MODEL_PATH, "labels_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(os.path.dirname(MODEL_PATH), "labels_config.json")
        
    if os.path.exists(config_path):
        print(f"Loading Config from: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
            l2id_t1 = config["task1"]
            l2id_t2 = config["task2"]
    else:
        print("WARNING: labels_config.json not found. Using defaults.")
        l2id_t1 = {'Ambivalent': 0, 'Clear Non-Reply': 1, 'Clear Reply': 2}
        l2id_t2 = {'Claims ignorance': 0, 'Clarification': 1, 'Declining to answer': 2, 'Deflection': 3, 'Dodging': 4, 'Explicit': 5, 'General': 6, 'Implicit': 7, 'Partial/half-answer': 8}

    id2l_t1 = {i: l for l, i in l2id_t1.items()}
    id2l_t2 = {i: l for l, i in l2id_t2.items()}
    
    # 2. Load Model
    print("Initializing Hierarchical Structure...")
    BASE_MODEL_NAME = "microsoft/deberta-v3-large" 
    
    model = HierarchicalBiEncoder(BASE_MODEL_NAME, len(l2id_t1), len(l2id_t2))
    
    print(f"Loading Weights from {MODEL_PATH}...")
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_PATH, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        
    model.load_state_dict(state_dict, strict=False)
    
    # 3. Predict
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    test_df = load_test_data()
    
    print("Tokenizing (Dual Tower)...")
    # Must match training preprocess exactly
    q_enc = tokenizer(test_df['interview_question'].tolist(), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    a_enc = tokenizer(test_df['interview_answer'].tolist(), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference on {device}...")
    model.to(device)
    model.eval()
    
    preds_t1 = []
    preds_t2 = []
    
    BATCH_SIZE = 8
    num_samples = len(test_df)
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_q_ids = q_enc["input_ids"][i:i+BATCH_SIZE].to(device)
            batch_q_mask = q_enc["attention_mask"][i:i+BATCH_SIZE].to(device)
            batch_a_ids = a_enc["input_ids"][i:i+BATCH_SIZE].to(device)
            batch_a_mask = a_enc["attention_mask"][i:i+BATCH_SIZE].to(device)
            
            logits1, logits2 = model(batch_q_ids, batch_q_mask, batch_a_ids, batch_a_mask)
            
            p1 = torch.argmax(logits1, dim=-1).cpu().numpy()
            p2 = torch.argmax(logits2, dim=-1).cpu().numpy()
            
            preds_t1.extend([id2l_t1[x] for x in p1])
            preds_t2.extend([id2l_t2[x] for x in p2])
            
    # Save
    with open("prediction_hierarchical_task1", "w") as f:
        for p in preds_t1: f.write(f"{p}\n")
        
    with open("prediction_hierarchical_task2", "w") as f:
        for p in preds_t2: f.write(f"{p}\n")
        
    print("Saved predictions: prediction_hierarchical_task1, prediction_hierarchical_task2")

if __name__ == "__main__":
    main()
