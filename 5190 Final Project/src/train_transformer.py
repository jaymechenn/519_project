import os
import pandas as pd
import torch

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import numpy as np
from sklearn.metrics import accuracy_score
torch.manual_seed(42)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Resources", "model_inputs", "transformer")
MODEL_DIR = os.path.join(BASE_DIR, "models", "bert")
os.makedirs(MODEL_DIR, exist_ok=True)


train_df = pd.read_csv(os.path.join(DATA_DIR, "train_transformer.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val_transformer.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_transformer.csv"))


train_dataset = Dataset.from_pandas(train_df[["headline_clean", "label"]])
val_dataset = Dataset.from_pandas(val_df[["headline_clean", "label"]])
test_dataset = Dataset.from_pandas(test_df[["headline_clean", "label"]])


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["headline_clean"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    save_total_limit=1
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


trainer.train()

logits = trainer.predict(val_dataset).predictions
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
labels = val_df["label"].values

best_t = 0.5
best_acc = 0

for t in np.linspace(0.3, 0.7, 41):
    preds = (probs >= t).astype(int)
    acc = accuracy_score(labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_t = t

print("BEST THRESHOLD:", best_t)

trainer.evaluate(val_dataset)
trainer.evaluate(test_dataset)


torch.save(model.state_dict(), os.path.join(BASE_DIR, "model.pt"))