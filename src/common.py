"""Provided utilities. No neural-network implementation is hidden here."""
import importlib.util
from pathlib import Path
import numpy as np


def load_impl(filename):
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location('lab_implementation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_params(path, params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)


def load_params(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def evaluate(impl, params, x, y, batch_size=256):
    """Use every example, including the short last batch; weight loss by count."""
    if len(x) != len(y) or not len(y) or batch_size <= 0:
        raise ValueError('Expected nonempty paired samples and positive batch size')
    total_loss, correct = 0.0, 0
    for start in range(0, len(y), batch_size):
        xb, yb = x[start:start+batch_size], y[start:start+batch_size]
        logits, _ = impl.forward(params, xb)
        loss, _ = impl.softmax_ce(logits, yb)
        total_loss += loss * len(yb)
        correct += int((logits.argmax(axis=1) == yb).sum())
    return total_loss / len(y), correct / len(y)
