
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence

# CRITICAL: Force Single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 256 # Shorter max length per segment (Q and A separately) to fit in memory
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EPOCHS = 4
LEARNING_RATE = 1e-5
OUTPUT_DIR = "./results_hierarchical"
TASK1_COL = "clarity_label"
TASK2_COL = "evasion_label"

# ==========================================
# NOVEL ARCHITECTURE: HIERARCHICAL BI-ENCODER
# ==========================================
class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key, value, key_padding_mask=None):
        # query: [Batch, SeqQ, Dim]
        # key/value: [Batch, SeqA, Dim]
        
        # MultiheadAttention requires key_padding_mask to be [Batch, Seq] where True is ignored/padded.
        # DeBERTa attention_mask is 1 for attend, 0 for ignore.
        # So we need to invert it: True where mask is 0.
        
        if key_padding_mask is not None:
             # Invert: 1 -> False (Don't Ignore), 0 -> True (Ignore)
             key_padding_mask = (key_padding_mask == 0)

        attn_output, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        return self.layer_norm(query + attn_output) # Residual connection

class HierarchicalBiEncoder(nn.Module):
    def __init__(self, model_name, num_labels_task1, num_labels_task2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Cross Attention: Q attends to A
        self.cross_attn = CrossAttentionLayer(self.hidden_size)
        
        # Heads
        self.task1_head = nn.Linear(self.hidden_size, num_labels_task1)
        self.task2_head = nn.Linear(self.hidden_size, num_labels_task2)
        
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids_q, attention_mask_q, input_ids_a, attention_mask_a, labels_task1=None, labels_task2=None, **kwargs):
        # 1. Encode Question independenty
        out_q = self.encoder(input_ids=input_ids_q, attention_mask=attention_mask_q)
        embed_q = out_q.last_hidden_state # [B, SeqQ, H]
        
        # 2. Encode Answer independently
        out_a = self.encoder(input_ids=input_ids_a, attention_mask=attention_mask_a)
        embed_a = out_a.last_hidden_state # [B, SeqA, H]
        
        # 3. Explicit Interaction (Cross Attention)
        # Query = Question, Key/Value = Answer
        # "For every token in Question, find relevant context in Answer"
        interaction = self.cross_attn(query=embed_q, key=embed_a, value=embed_a, key_padding_mask=attention_mask_a)
        
        # 4. Pooling (Mean over Question sequence)
        # We perform mean pooling over the interaction representation
        # Use Mask to ignore padding tokens in the Question
        mask_expanded = attention_mask_q.unsqueeze(-1).expand(interaction.size()).float()
        sum_embeddings = torch.sum(interaction * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask # [B, H]
        
        # 5. Classification
        logits_task1 = self.task1_head(pooled_output)
        logits_task2 = self.task2_head(pooled_output)
        
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
# DATA LOADING & PREPROCESSING
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

def custom_collate_fn(batch):
    # Batch is a list of dicts from the dataset
    # We need to stack the inputs. 
    # Since we have separate Q and A inputs of varying lengths, we must pad manually here.
    
    input_ids_q = [torch.tensor(item['input_ids_q']) for item in batch]
    attention_mask_q = [torch.tensor(item['attention_mask_q']) for item in batch]
    input_ids_a = [torch.tensor(item['input_ids_a']) for item in batch]
    attention_mask_a = [torch.tensor(item['attention_mask_a']) for item in batch]
    
    labels_task1 = torch.tensor([item['labels_task1'] for item in batch])
    labels_task2 = torch.tensor([item['labels_task2'] for item in batch])
    
    # Pad sequences
    input_ids_q = pad_sequence(input_ids_q, batch_first=True, padding_value=0)
    attention_mask_q = pad_sequence(attention_mask_q, batch_first=True, padding_value=0)
    input_ids_a = pad_sequence(input_ids_a, batch_first=True, padding_value=0)
    attention_mask_a = pad_sequence(attention_mask_a, batch_first=True, padding_value=0)
    
    return {
        "input_ids_q": input_ids_q,
        "attention_mask_q": attention_mask_q,
        "input_ids_a": input_ids_a,
        "attention_mask_a": attention_mask_a,
        "labels_task1": labels_task1,
        "labels_task2": labels_task2
    }

# ==========================================
# CUSTOM TRAINER
# ==========================================
class HierarchicalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids_q=inputs["input_ids_q"],
            attention_mask_q=inputs["attention_mask_q"],
            input_ids_a=inputs["input_ids_a"],
            attention_mask_a=inputs["attention_mask_a"],
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
    
    print(f"Task 1 Labels: {l2id1}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        # Tokenize Question separately
        q_enc = tokenizer(examples['interview_question'], truncation=True, max_length=MAX_LENGTH)
        # Tokenize Answer separately
        a_enc = tokenizer(examples['interview_answer'], truncation=True, max_length=MAX_LENGTH)
        
        batch_size = len(examples['interview_question'])
        formatted_batch = {
            "input_ids_q": q_enc["input_ids"],
            "attention_mask_q": q_enc["attention_mask"],
            "input_ids_a": a_enc["input_ids"],
            "attention_mask_a": a_enc["attention_mask"],
            "labels_task1": [l2id1[l] for l in examples[TASK1_COL]],
            "labels_task2": [l2id2[l] for l in examples[TASK2_COL]]
        }
        return formatted_batch

    train_ds = HFDataset.from_pandas(train_df).map(preprocess, batched=True, remove_columns=list(train_df.columns))
    
    model = HierarchicalBiEncoder(MODEL_NAME, len(labels1), len(labels2))
    
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
        remove_unused_columns=False, # Critical for custom collator passing custom keys
        report_to="none"
    )

    trainer = HierarchicalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=custom_collate_fn
    )

    print("Starting Hierarchical Bi-Encoder Training...")
    trainer.train()
    
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    
    import json
    with open(f"{OUTPUT_DIR}/best_model/labels_config.json", "w") as f:
        json.dump({"task1": l2id1, "task2": l2id2}, f)
        
    print(f"Done! Saved to {OUTPUT_DIR}/best_model")

if __name__ == "__main__":
    main()
