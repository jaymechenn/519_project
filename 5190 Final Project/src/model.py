from typing import Any, Iterable, List

import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn


FEATURES_PER_VIEW = 131072
FEATURE_DIM = FEATURES_PER_VIEW * 2
NUM_CLASSES = 2


class NewsClassifier(nn.Module):
    """
    PyTorch checkpoint model for NBC vs FoxNews headline classification.

    The evaluator passes headline strings into predict(...). This class builds
    character n-gram hash features, applies IDF weights saved in the checkpoint,
    then uses a PyTorch linear classifier whose weights are loaded from .pt.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        super().__init__()
        self.word_vectorizer = HashingVectorizer(
            n_features=FEATURES_PER_VIEW,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.char_vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            n_features=FEATURES_PER_VIEW,
            alternate_sign=False,
            norm=None,
            lowercase=True,
        )
        self.register_buffer("word_idf", torch.ones(FEATURES_PER_VIEW, dtype=torch.float32))
        self.register_buffer("char_idf", torch.ones(FEATURES_PER_VIEW, dtype=torch.float32))
        self.classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)

    def _vectorize(self, headlines: Iterable[Any]) -> torch.Tensor:
        texts = [str(headline) for headline in headlines]
        word_features = self.word_vectorizer.transform(texts)
        char_features = self.char_vectorizer.transform(texts)
        word_dense = torch.tensor(word_features.toarray(), dtype=torch.float32)
        char_dense = torch.tensor(char_features.toarray(), dtype=torch.float32)
        word_dense = word_dense * self.word_idf
        char_dense = char_dense * self.char_idf
        word_dense = torch.nn.functional.normalize(word_dense, p=2, dim=1)
        char_dense = torch.nn.functional.normalize(char_dense, p=2, dim=1)
        features = torch.cat([word_dense, char_dense], dim=1)
        return features

    def forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            features = inputs.float()
        else:
            features = self._vectorize(inputs)
        return self.classifier(features)

    def predict(self, batch: Iterable[Any]) -> List[int]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(list(batch))
            return logits.argmax(dim=1).cpu().tolist()


Model = NewsClassifier


def get_model() -> NewsClassifier:
    return NewsClassifier()
