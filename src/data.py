"""Provided MNIST IDX reader. Uses local .gz or uncompressed files only."""
import gzip
import struct
from pathlib import Path
import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data'


def read_idx(directory, filename, images):
    path = Path(directory) / filename
    if path.exists():
        raw = path.read_bytes()
    elif path.with_suffix(path.suffix + '.gz').exists():
        with gzip.open(str(path) + '.gz', 'rb') as handle:
            raw = handle.read()
    else:
        raise FileNotFoundError(f'Missing {path}. Ask the instructor for the data folder.')
    magic, count = struct.unpack_from('>II', raw)
    if images:
        if magic != 2051:
            raise ValueError('Wrong image IDX magic')
        rows, cols = struct.unpack_from('>II', raw, 8)
        if (rows, cols) != (28, 28) or len(raw) != 16 + count * 784:
            raise ValueError('Wrong MNIST dimensions or truncated image file')
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 784)
    if magic != 2049 or len(raw) != 8 + count:
        raise ValueError('Wrong label IDX header or truncated file')
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).astype(np.int64)
    if np.any(labels > 9):
        raise ValueError('MNIST label outside 0..9')
    return labels


def load_data(directory=DEFAULT_DATA, split='train', train_size=6000, val_size=1000):
    if split == 'test':
        x = read_idx(directory, 't10k-images-idx3-ubyte', True)
        y = read_idx(directory, 't10k-labels-idx1-ubyte', False)
        if len(x) != len(y):
            raise ValueError('Image/label count mismatch')
        return x.astype(np.float32) / 255.0, y
    x = read_idx(directory, 'train-images-idx3-ubyte', True)
    y = read_idx(directory, 'train-labels-idx1-ubyte', False)
    if len(x) != 60000 or len(x) != len(y):
        raise ValueError('Expected official 60000-example training split')
    if not 1 <= train_size <= 55000 or not 1 <= val_size <= 5000:
        raise ValueError('train_size: 1..55000; val_size: 1..5000')
    # Split seed is fixed, independent of the model/training seed.
    order = np.random.default_rng(2026).permutation(len(y))
    train_ids, val_ids = order[:train_size], order[55000:55000+val_size]
    return (x[train_ids].astype(np.float32)/255, y[train_ids],
            x[val_ids].astype(np.float32)/255, y[val_ids])
