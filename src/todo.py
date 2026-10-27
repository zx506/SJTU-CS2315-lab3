"""Please complete nine TODO blocks using Python / NumPy."""
import numpy as np


def scalar_step(w, b, x, y, lr):
    """Return old loss, dw, db, new w, new b. No NumPy is needed here."""
    # TODO: scalar
    raise NotImplementedError("scalar")


def linear_forward(x, w, b):
    """x: [B,I], w: [I,O], b: [O]; return [B,O]."""
    # TODO: linear_forward
    raise NotImplementedError("linear_forward")


def linear_backward(x, w, dout):
    """Return dx, dw, db. dout already includes loss averaging."""
    # TODO: linear_backward
    raise NotImplementedError("linear_backward")


def relu_forward(x):
    # TODO: relu_forward
    raise NotImplementedError("relu_forward")


def relu_backward(x, dout):
    """Use derivative 0 at x=0. Do not modify either input."""
    # TODO: relu_backward
    raise NotImplementedError("relu_backward")


def softmax_ce(logits, labels):
    """Return mean CE and dlogits. labels: integer [B], logits: [B,C]."""
    # TODO: softmax_ce
    raise NotImplementedError("softmax_ce")


def init_params(seed=7, dims=(784, 32, 16, 10), dtype=np.float32):
    """Provided: random weights break symmetry; biases start at zero."""
    rng = np.random.default_rng(seed)
    params = {}
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:]), 1):
        params[f'W{i}'] = (rng.standard_normal((din, dout)) * np.sqrt(2 / din)).astype(dtype)
        params[f'b{i}'] = np.zeros(dout, dtype=dtype)
    return params


def forward(params, x):
    """Return raw logits and cache: (x, z1, h1, z2, h2)."""
    # TODO: forward
    raise NotImplementedError("forward")


def backward(params, cache, dlogits):
    """Return a dict with the same keys/shapes as params. Do not update params."""
    # TODO: backward
    raise NotImplementedError("backward")


def sgd_update(params, grads, lr):
    """Update every parameter in place, only after the whole backward pass."""
    # TODO: sgd_update
    raise NotImplementedError("sgd_update")
