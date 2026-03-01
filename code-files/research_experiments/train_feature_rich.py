
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset

# CRITICAL: Force Single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EPOCHS = 4
LEARNING_RATE = 1e-5
OUTPUT_DIR = "./results_feature_rich"
TASK1_COL = "clarity_label"
TASK2_COL = "evasion_label"

# ==========================================
# ADVANCED ARCHITECTURE COMPONENTS
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
        # all_hidden_states is a tuple of tensors: (Layer0, Layer1, ... Layer24)
        # We want to pool from layer_start to end
        all_layer_embedding = torch.stack(list(all_hidden_states)[self.layer_start:], dim=0) 
        # Shape: [num_layers, batch, seq, hidden]
        
        weight_factor = self.layer_weights / self.layer_weights.sum()
        # Reshape weights for broadcasting: [num_layers, 1, 1, 1]
        weight_factor = weight_factor.view(-1, 1, 1, 1)
        
        # Weighted Average
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0)
        # Shape: [batch, seq, hidden]
        
        # Extract CLS token from the weighted average
        return weighted_average[:, 0]

class MultiSampleDropout(nn.Module):
    def __init__(self, input_dim, output_dim, num_samples=5, dropout_rate=0.5):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.classifier = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x: [batch, input_dim]
        # Pass through 5 distinct dropouts, then average the logits
        logits_list = [self.classifier(dropout(x)) for dropout in self.dropouts]
        return torch.mean(torch.stack(logits_list, dim=0), dim=0)

# ==========================================
# FEATURE-RICH MODEL
# ==========================================
class FeatureRichModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = self.encoder.config
        
        # Weighted Pooling: Use last 20 layers (Skip first 4 which are very low-level)
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=config.num_hidden_layers, 
            layer_start=4
        )
        
        # Multi-Sample Dropout Heads
        self.task1_head = MultiSampleDropout(config.hidden_size, num_labels_task1)
        self.task2_head = MultiSampleDropout(config.hidden_size, num_labels_task2)
        
        self.loss_fct = nn.CrossEntropyLoss()
        
        # Initialize weights for pooler
        self._init_weights(self.pooler.layer_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.encoder.config.initializer_range)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels_task1=None, labels_task2=None, **kwargs):
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            output_hidden_states=True # Crucial: Return all layers
        )
        
        # Extract Weighted Feature Representation
        feature_vector = self.pooler(outputs.hidden_states)
        
        # Classify
        logits_task1 = self.task1_head(feature_vector)
        logits_task2 = self.task2_head(feature_vector)
        
        loss = None
        if labels_task1 is not None and labels_task2 is not None:
            loss1 = self.loss_fct(logits_task1, labels_task1)
            loss2 = self.loss_fct(logits_task2, labels_task2)
            loss = loss1 + loss2
            
        return {
            "loss": loss,
            "logits_task1": logits_task1,
            "logits_task2": logits_task2
        }

# ==========================================
# DATA LOADING - STANDARD (Concatenation)
# ==========================================
def load_data():
    if os.path.exists("data/train.csv"):
        print("Loading local CSV...")
        train_df = pd.read_csv("data/train.csv")
    else:
        print("Downloading dataset...")
        from datasets import load_dataset
        ds = load_dataset("ailsntua/QEvasion")
        train_df = ds['train'].to_pandas()
    return train_df

# ==========================================
# TRAINER (Standard)
# ==========================================
class StandardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Need to allow **kwargs broadly
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels_task1=inputs.get("labels_task1"),
            labels_task2=inputs.get("labels_task2")
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# ==========================================
# MAIN
# ==========================================
def main():
    train_df = load_data()
    print(f"Train Size: {len(train_df)}")

    labels1 = sorted(train_df[TASK1_COL].unique())
    l2id1 = {l: i for i, l in enumerate(labels1)}
    
    labels2 = sorted(train_df[TASK2_COL].unique())
    l2id2 = {l: i for i, l in enumerate(labels2)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['interview_question'], examples['interview_answer'])]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        
        model_inputs["labels_task1"] = [l2id1[l] for l in examples[TASK1_COL]]
        model_inputs["labels_task2"] = [l2id2[l] for l in examples[TASK2_COL]]
        return model_inputs

    train_ds = HFDataset.from_pandas(train_df).map(preprocess, batched=True)
    
    model = FeatureRichModel(MODEL_NAME, len(labels1), len(labels2))
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=1,
        remove_unused_columns=True, 
        report_to="none"
    )

    trainer = StandardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("Starting Feature-Rich Training (Weighted Pooling + Multi-Sample Dropout)...")
    trainer.train()
    
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    
    import json
    with open(f"{OUTPUT_DIR}/best_model/labels_config.json", "w") as f:
        json.dump({"task1": l2id1, "task2": l2id2}, f)
        
    print(f"Done! Saved to {OUTPUT_DIR}/best_model")

if __name__ == "__main__":
    main()
