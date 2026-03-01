
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
MODEL_PATH = "results_feature_rich/best_model"
TEST_FILE = "data/test.csv"
MAX_LENGTH = 512

# ==========================================
# ADVANCED ARCHITECTURE COMPONENTS (Must match training!)
# ==========================================
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super().__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = torch.stack(list(all_hidden_states)[self.layer_start:], dim=0) 
        weight_factor = self.layer_weights / self.layer_weights.sum()
        weight_factor = weight_factor.view(-1, 1, 1, 1)
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        return weighted_average[:, 0]

class MultiSampleDropout(nn.Module):
    def __init__(self, input_dim, output_dim, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.classifier = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        logits_list = [self.classifier(dropout(x)) for dropout in self.dropouts]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)

class FeatureRichModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = self.encoder.config
        
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=config.num_hidden_layers, 
            layer_start=4
        )
        
        self.task1_head = MultiSampleDropout(config.hidden_size, num_labels_task1)
        self.task2_head = MultiSampleDropout(config.hidden_size, num_labels_task2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        feature_vector = self.pooler(outputs.hidden_states)
        
        logits_task1 = self.task1_head(feature_vector)
        logits_task2 = self.task2_head(feature_vector)
        
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
        print("Run 'python code-files/research_experiments/train_feature_rich.py' first.")
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
    print("Initializing Feature-Rich Structure...")
    BASE_MODEL_NAME = "microsoft/deberta-v3-large" 
    
    model = FeatureRichModel(BASE_MODEL_NAME, len(l2id_t1), len(l2id_t2))
    
    print(f"Loading Weights from {MODEL_PATH}...")
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_PATH, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        
    # strict=False to ignore loss weights if they were saved
    model.load_state_dict(state_dict, strict=False)
    
    # 3. Predict
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    test_df = load_test_data()
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(test_df['interview_question'], test_df['interview_answer'])]
    
    print("Tokenizing...")
    encodings = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding=True, return_tensors="pt")
    
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
    with open("prediction_featurerich_task1", "w") as f:
        for p in preds_t1: f.write(f"{p}\n")
        
    with open("prediction_featurerich_task2", "w") as f:
        for p in preds_t2: f.write(f"{p}\n")
        
    print("Saved predictions: prediction_featurerich_task1, prediction_featurerich_task2")

if __name__ == "__main__":
    main()
