
import os


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EPOCHS = 4
LEARNING_RATE = 1e-5
OUTPUT_DIR = "./results_multitask"
TASK1_COL = "clarity_label"
TASK2_COL = "evasion_label"

# ==========================================
# CUSTOM MODEL
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
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] token
        
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['interview_question'], examples['interview_answer'])]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
        
        model_inputs["labels_task1"] = [l2id1[l] for l in examples[TASK1_COL]]
        model_inputs["labels_task2"] = [l2id2[l] for l in examples[TASK2_COL]]
        return model_inputs

    train_ds = HFDataset.from_pandas(train_df).map(preprocess, batched=True)
    
    model = MultiTaskModel(MODEL_NAME, len(labels1), len(labels2))

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
        remove_unused_columns=True, 
        report_to="none"
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None, # NO VALIDATION SET
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("Starting Multi-Task Training...")
    trainer.train()
    
    # Save Custom Model
    # Trainer.save_model() usually saves the model.state_dict(), which is fine.
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print(f"Model saved to {OUTPUT_DIR}/best_model")
    
    # Save label mappings for inference
    import json
    with open(f"{OUTPUT_DIR}/best_model/labels_config.json", "w") as f:
        json.dump({"task1": l2id1, "task2": l2id2}, f)

if __name__ == "__main__":
    main()
