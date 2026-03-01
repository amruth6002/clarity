
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from sklearn.model_selection import StratifiedKFold
import shutil

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
OUTPUT_BASE_DIR = "./results_kfold"
TASK1_COL = "clarity_label"
TASK2_COL = "evasion_label"
N_SPLITS = 5

# ==========================================
# CUSTOM MODEL (Baseline - Proven Best)
# ==========================================
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.task1_head = nn.Linear(hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(hidden_size, num_labels_task2)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels_task1=None, labels_task2=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        logits_task1 = self.task1_head(cls_output)
        logits_task2 = self.task2_head(cls_output)
        
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
# DATA LOADING
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
# CUSTOM TRAINER
# ==========================================
class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
    print(f"Total Dataset Size: {len(train_df)}")

    # Prepare Labels
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

    # K-FOLD LOOP
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # Stratify based on Task 1 (Clarity) as it's the primary task
    y_stratify = train_df[TASK1_COL]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_stratify)):
        print(f"\n{'='*20}")
        print(f"STARTING FOLD {fold}/{N_SPLITS-1}")
        print(f"{'='*20}")
        
        fold_output_dir = f"{OUTPUT_BASE_DIR}/fold_{fold}"
        
        # Split Data
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train samples: {len(fold_train_df)} | Val samples: {len(fold_val_df)}")
        
        # Create Datasets
        train_ds = HFDataset.from_pandas(fold_train_df).map(preprocess, batched=True)
        val_ds = HFDataset.from_pandas(fold_val_df).map(preprocess, batched=True)
        
        # Initialize Model (Brand new for each fold)
        model = MultiTaskModel(MODEL_NAME, len(labels1), len(labels2))
        
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            eval_strategy="epoch", # We HAVE validation data now, so we can evaluate!
            save_strategy="epoch",
            save_total_limit=1, # Keep only best per fold
            load_best_model_at_end=True, # Load best weights at end of fold
            metric_for_best_model="loss",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE*2,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            fp16=True,
            remove_unused_columns=True, 
            report_to="none"
        )
        
        trainer = MultiTaskTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
        )
        
        trainer.train()
        
        # Save Best Model for this Fold
        best_model_path = f"{fold_output_dir}/best_model"
        trainer.save_model(best_model_path)
        print(f"Fold {fold} Model Saved to: {best_model_path}")
        
        # Save Config (Crucial for inference)
        import json
        with open(f"{best_model_path}/labels_config.json", "w") as f:
            json.dump({"task1": l2id1, "task2": l2id2}, f)
            
        # Clean up checkpoints to save space
        checkpoint_dirs = [d for d in os.listdir(fold_output_dir) if d.startswith("checkpoint")]
        for d in checkpoint_dirs:
            shutil.rmtree(os.path.join(fold_output_dir, d))
            
    print("\nALL FOLDS COMPLETED SUCCESSFULLLY.")

if __name__ == "__main__":
    main()
