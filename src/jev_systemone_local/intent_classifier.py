"""Pure-Python task classifier for /v1/recommend (character n-gram TF-IDF + multinomial logistic regression).

The weights in intent_model.json are trained by eval/recommend/train_classifier.py (scikit-learn, not a runtime
dependency). This module reimplements only the prediction side, so the endpoint needs no extra packages and
answers in well under a millisecond.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("intent_model.json")
_SPACES = re.compile(r"\s\s+")  # same normalisation as scikit-learn's analyzer="char"


class IntentClassifier:
    def __init__(self, model: dict) -> None:
        self.classes: list[str] = model["classes"]
        self.min_n, self.max_n = model["ngram"]
        self.intercept: list[float] = model["intercept"]
        self.features: dict[str, list[float]] = model["features"]  # ngram -> [idf, coef per class...]

    def _ngrams(self, text: str) -> Counter:
        text = _SPACES.sub(" ", text.lower())
        grams: Counter = Counter()
        for n in range(self.min_n, min(self.max_n, len(text)) + 1):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                if gram in self.features:
                    grams[gram] += 1
        return grams

    def predict_proba(self, text: str) -> dict[str, float]:
        weights = {g: (1.0 + math.log(c)) * self.features[g][0] for g, c in self._ngrams(text).items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        scores = list(self.intercept)
        if norm:
            for gram, w in weights.items():
                x = w / norm
                for k, coef in enumerate(self.features[gram][1:]):
                    scores[k] += x * coef
        top = max(scores)
        exp = [math.exp(s - top) for s in scores]
        total = sum(exp)
        return {c: e / total for c, e in zip(self.classes, exp)}


@lru_cache(maxsize=1)
def load_classifier() -> IntentClassifier:
    return IntentClassifier(json.loads(MODEL_PATH.read_text(encoding="utf-8")))
