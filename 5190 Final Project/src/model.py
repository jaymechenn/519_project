import os
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

class Model:
    def __init__(self, weights_path=None):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )

        if weights_path is not None and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def predict(self, X):
        probs = []

        with torch.no_grad():
            for text in X:
                inputs = self.tokenizer(
                    str(text),
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=128
                )

                outputs = self.model(**inputs)
                p = torch.softmax(outputs.logits, dim=1)[0][1].item()
                probs.append(p)

        probs = np.array(probs)
        preds = (probs >= 0.5).astype(int)
        return preds.tolist()