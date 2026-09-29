"""Train the /v1/recommend task classifier and export it as src/jev_systemone_local/intent_model.json.

Needs scikit-learn (training only; the served classifier is pure Python):
    uv run --with scikit-learn python eval/recommend/train_classifier.py
Trains on train_requests_{a,b,c}.py only; requests_ja.py stays a held-out evaluation set (first 6 per task = dev,
last 6 = test). The vocabulary is pruned to the K features with the largest weights and the model is refit on
that vocabulary, so the exported model is exactly what was evaluated.
"""
from __future__ import annotations

import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

here = Path(__file__).parent
root = here.parent.parent
sys.path[:0] = [str(here), str(root / "src")]
from requests_ja import REQUESTS  # noqa: E402

from jev_systemone_local.intent_classifier import IntentClassifier  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.TRAIN_REQUESTS


train = load("train_requests_a") + load("train_requests_b") + load("train_requests_c")
X, y = [r["text"] for r in train], [r["task"] for r in train]
seen = collections.defaultdict(int)
dev, test = [], []
for r in REQUESTS:
    seen[r["task"]] += 1
    (dev if seen[r["task"]] <= 6 else test).append(r)


def fit(vocabulary=None):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, min_df=2 if vocabulary is None else 1,
                          vocabulary=vocabulary)
    lr = LogisticRegression(C=10, max_iter=4000)
    lr.fit(vec.fit_transform(X), y)
    return vec, lr


def export(vec, lr):
    names = vec.get_feature_names_out()
    return {"classes": list(lr.classes_), "ngram": [1, 3], "intercept": [round(float(v), 5) for v in lr.intercept_],
            "features": {n: [round(float(vec.idf_[i]), 4)] + [round(float(c), 4) for c in lr.coef_[:, i]]
                         for i, n in enumerate(names)}}


def accuracy(model, rows):
    return sum(max(p := model.predict_proba(r["text"]), key=p.get) == r["task"] for r in rows) / len(rows)


vec, lr = fit()
print("full vocabulary:", len(vec.vocabulary_))
order = np.argsort(-np.abs(lr.coef_).max(axis=0))
names = vec.get_feature_names_out()
results = {}
for k in (1500, 3000, 6000, len(names)):
    vocab = sorted(names[order[:k]])
    v2, l2 = fit(vocab)
    model = IntentClassifier(export(v2, l2))
    results[k] = model
    print(f"K={k:6d}  dev {accuracy(model, dev):.3f}  test {accuracy(model, test):.3f}  all {accuracy(model, REQUESTS):.3f}")

K = 3000
out = root / "src/jev_systemone_local/intent_model.json"
vocab = sorted(names[order[:K]])
v2, l2 = fit(vocab)
payload = export(v2, l2)
out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
# parity: the pure-Python model must agree with scikit-learn on every training text
py = IntentClassifier(json.loads(out.read_text(encoding="utf-8")))
diff = max(abs(py.predict_proba(t)[c] - p) for t, prob in zip(X[:100], l2.predict_proba(v2.transform(X[:100])))
           for c, p in zip(l2.classes_, prob))
print(f"exported K={K}: {out.stat().st_size / 1024:.0f} KB; max |python - sklearn| probability = {diff:.4f}")
