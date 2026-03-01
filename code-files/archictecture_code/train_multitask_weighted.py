
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# CRITICAL: Force Single GPU to avoid DataParallel/NCCL instability with custom models on Kaggle
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
OUTPUT_DIR = "./results_multitask_weighted" # New output directory
TASK1_COL = "clarity_label"
TASK2_COL = "evasion_label"

# ==========================================
# CUSTOM MODEL WITH WEIGHTED LOSS
# ==========================================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2, weights_task1=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.task1_head = nn.Linear(hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(hidden_size, num_labels_task2)
        
        # Weighted Loss for Task 1 (Clarity) to handle imbalance
        if weights_task1 is not None:
            # Check if weights are on the correct device when initializing?
            # nn.CrossEntropyLoss stores the weight tensor. We need to make sure it moves with the model.
            # Best way is to register it as a buffer if we were writing a pure custom component, 
            # but usually passing it to CrossEntropyLoss is enough provided the module is moved to device.
            self.loss_fct1 = nn.CrossEntropyLoss(weight=weights_task1)
        else:
            self.loss_fct1 = nn.CrossEntropyLoss()
            
        self.loss_fct2 = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels_task1=None, labels_task2=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        logits_task1 = self.task1_head(cls_output)
        logits_task2 = self.task2_head(cls_output)
        
        loss = None
        if labels_task1 is not None and labels_task2 is not None:
            loss1 = self.loss_fct1(logits_task1, labels_task1)
            loss2 = self.loss_fct2(logits_task2, labels_task2)
            loss = loss1 + loss2
            
        return {
            "loss": loss,
            "logits_task1": logits_task1,
            "logits_task2": logits_task2
        }

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    if os.path.exists("data/train.csv"):
        print("Loading from local CSVs...")
        train_df = pd.read_csv("data/train.csv")
    else:
        print("Downloading from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        train_df = dataset['train'].to_pandas()
        
    # User requested NO validation split - use full data for training
    return train_df

# ==========================================
# CUSTOM TRAINER
# ==========================================
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_task1 = inputs.get("labels_task1")
        labels_task2 = inputs.get("labels_task2")
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels_task1=labels_task1,
            labels_task2=labels_task2
        )
        
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

# ==========================================
# MAIN
# ==========================================
def main():
    train_df = load_data()
    print(f"Train (Full): {len(train_df)}")

    # Prepare Labels
    labels1 = sorted(train_df[TASK1_COL].unique())
    l2id1 = {l: i for i, l in enumerate(labels1)}
    
    labels2 = sorted(train_df[TASK2_COL].unique())
    l2id2 = {l: i for i, l in enumerate(labels2)}
    
    print(f"Task 1 Labels ({len(labels1)}): {l2id1}")
    print(f"Task 2 Labels ({len(labels2)}): {l2id2}")

    # ---------------------------------------------------------
    # COMPUTE CLASS WEIGHTS FOR TASK 1
    # ---------------------------------------------------------
    # Logic: Inverse frequency to penalize missed rare classes more
    counts = train_df[TASK1_COL].value_counts().sort_index()
    # Align counts with label IDs
    counts_aligned = [counts[l] for l in labels1]
    total_samples = sum(counts_aligned)
    num_classes = len(labels1)
    
    # Sklearn formula: n_samples / (n_classes * n_samples_j)
    weights = [total_samples / (num_classes * c) for c in counts_aligned]
    
    # Convert to Tensor (Float)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    
    print(f"Task 1 Label Counts: {dict(zip(labels1, counts_aligned))}")
    print(f"Task 1 Computed Weights: {weights}")
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['interview_question'], examples['interview_answer'])]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        
        model_inputs["labels_task1"] = [l2id1[l] for l in examples[TASK1_COL]]
        model_inputs["labels_task2"] = [l2id2[l] for l in examples[TASK2_COL]]
        return model_inputs

    train_ds = HFDataset.from_pandas(train_df).map(preprocess, batched=True)
    
    # PASS WEIGHTS TO MODEL
    # Important: weights_tensor needs to be moved to GPU. 
    # Usually Trainer moves the model. If we pass tensor to init, it stays in CPU until model is moved.
    # We'll rely on Trainer's model.to(device) which recursively moves submodules (loss functions included).
    model = MultiTaskModel(MODEL_NAME, len(labels1), len(labels2), weights_task1=weights_tensor)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="no", # DISABLE EVALUATION
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,
        save_total_limit=1, # Save only best/latest
        remove_unused_columns=True, 
        report_to="none"
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("Starting Multi-Task Training (Weighted Task 1)...")
    trainer.train()
    
    # Save Custom Model
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print(f"Model saved to {OUTPUT_DIR}/best_model")
    
    # Save label mappings for inference
    import json
    with open(f"{OUTPUT_DIR}/best_model/labels_config.json", "w") as f:
        json.dump({"task1": l2id1, "task2": l2id2}, f)

if __name__ == "__main__":
    main()
