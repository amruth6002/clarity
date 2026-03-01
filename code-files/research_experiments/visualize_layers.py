
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoModel, AutoConfig

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "results_feature_rich/best_model"
OUTPUT_PLOT = "layer_importance.png"
BASE_MODEL_NAME = "microsoft/deberta-v3-large"

# ==========================================
# MODEL DEFINITION (Must Match Training)
# ==========================================
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super().__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
        )

class FeatureRichModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = self.encoder.config
        
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=config.num_hidden_layers, 
            layer_start=4
        )

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path {MODEL_PATH} not found.")
        print("Run 'python code-files/research_experiments/train_feature_rich.py' first.")
        return

    print(f"Loading weights from {MODEL_PATH}...")
    
    # 1. Initialize Model Wrapper
    # (We don't need the heads for visualization, just the pooler)
    # We use dummy labels counts as they don't affect pooler weights
    model = FeatureRichModel(BASE_MODEL_NAME, 3, 9)
    
    # 2. Load State Dict
    weights_path = os.path.join(MODEL_PATH, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_PATH, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")
        
    # strict=False because we might be missing some Dropout/Heads layers if logic slightly differed, 
    # but we ONLY care about 'pooler.layer_weights'
    model.load_state_dict(state_dict, strict=False)
    
    # 3. Extract Weights
    # Weights are: model.pooler.layer_weights
    raw_weights = model.pooler.layer_weights.detach().cpu().numpy()
    
    # Normalize them exactly how the model does in forward pass:
    # weight_factor = self.layer_weights / self.layer_weights.sum()
    norm_weights = raw_weights / raw_weights.sum()
    
    # 4. Plot
    layers = range(model.pooler.layer_start, model.pooler.layer_start + len(norm_weights))
    
    data = pd.DataFrame({
        "Layer Index": layers,
        "Importance Weight": norm_weights
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Layer Index", y="Importance Weight", data=data, palette="viridis")
    plt.title("Learned Layer Importance for Evasion Detection\n(Weighted Layer Pooling)", fontsize=15)
    plt.xlabel("DeBERTa Layer (0-24)", fontsize=12)
    plt.ylabel("Normalized Weight", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save
    plt.savefig(OUTPUT_PLOT)
    print(f"Success! Plot saved to: {os.path.abspath(OUTPUT_PLOT)}")
    print("\nInterpretation Guide:")
    print("- High bars in LAST layers (20-24): Model relies on high-level semantics/logic.")
    print("- High bars in MIDDLE layers (10-15): Model relies on syntax/phrasing.")
    print("- This plot is Figure 3 in your Research Paper!")

if __name__ == "__main__":
    main()
