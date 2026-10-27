"""Run unchanged. Inspect shapes and explain each operation in words."""
import numpy as np

x = np.array([[1., 2., 3.], [4., 5., 6.]])
w = np.array([[1., 0.], [0., 1.], [1., 1.]])
b = np.array([.1, -.1])
print('X shape:', x.shape, 'W shape:', w.shape, 'b shape:', b.shape)
print('X @ W + b =\n', x @ w + b)
print('X.T shape:', x.T.shape)
print('sum over samples (axis=0):', x.sum(axis=0))
print('sum over features (axis=1):', x.sum(axis=1))
print('row max with keepdims:\n', x.max(axis=1, keepdims=True))
print('Boolean mask:\n', x > 3)
labels = np.array([2, 0])
scores = np.array([[.2, .1, .7], [.8, .1, .1]])
print('one correct-class value per row:', scores[np.arange(2), labels])
print('Exercise: explain @ versus *, axis=0 versus axis=1, and keepdims.')
