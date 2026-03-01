
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset as HFDataset
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./results_task1"
TASK_LABEL_COL = "clarity_label"

# ==========================================
# DATA LOADING
# ==========================================
def load_data():
    """
    Tries to load data from local CSVs (uploaded to Kaggle) or downloads from HF.
    """
    if os.path.exists("data/train.csv"):
        print("Loading from local CSVs...")
        train_df = pd.read_csv("data/train.csv")
        val_df = pd.read_csv("data/val.csv")
        test_df = pd.read_csv("data/test.csv")
    else:
        print("Downloading from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Create validation split if not exists
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df[TASK_LABEL_COL])

    return train_df, val_df, test_df

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess_function(examples, tokenizer, label2id):
    # Inputs: [CLS] Question [SEP] Answer [SEP]
    inputs = [f"Question: {q} Answer: {a}" for q, a in zip(examples['interview_question'], examples['interview_answer'])]
    
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    
    if TASK_LABEL_COL in examples:
        model_inputs["labels"] = [label2id[l] for l in examples[TASK_LABEL_COL]]
        
    return model_inputs

# ==========================================
# METRICS
# ==========================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy": (predictions == labels).mean()
    }

# ==========================================
# MAIN
# ==========================================
def main():
    # 1. Load Data
    train_df, val_df, test_df = load_data()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Prepare Labels
    unique_labels = sorted(train_df[TASK_LABEL_COL].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    print(f"Labels: {label2id}")

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 4. Convert to HuggingFace Dataset
    train_ds = HFDataset.from_pandas(train_df)
    val_ds = HFDataset.from_pandas(val_df)
    test_ds = HFDataset.from_pandas(test_df)

    # 5. Map Preprocessing
    encode = lambda x: preprocess_function(x, tokenizer, label2id)
    train_encoded = train_ds.map(encode, batched=True)
    val_encoded = val_ds.map(encode, batched=True)
    
    # 6. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # 7. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("Starting Training...")
    trainer.train()

    # 9. Evaluate & Predict
    print("Evaluating on Validation Set...")
    metrics = trainer.evaluate()
    print(metrics)

    # 10. Save Model
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    print(f"Model saved to {OUTPUT_DIR}/best_model")

if __name__ == "__main__":
    main()
