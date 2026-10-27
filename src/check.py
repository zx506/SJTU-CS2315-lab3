"""Public checks: known values, numerical gradients, stability and convergence."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
from pathlib import Path
import tempfile
import unittest
import numpy as np
from common import load_impl, save_params, load_params, evaluate

impl = None


def numeric_gradient(fn, arr, eps=1e-5):
    result = np.empty_like(arr)
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        try:
            arr[idx] = old + eps
            plus = fn()
            arr[idx] = old - eps
            minus = fn()
        finally:
            arr[idx] = old
        result[idx] = (plus - minus) / (2 * eps)
    return result


class Checks(unittest.TestCase):
    def test_scalar(self):
        np.testing.assert_allclose(impl.scalar_step(1., 0., 2., 5., .05), [9, -12, -6, 1.6, .3])
        np.testing.assert_allclose(impl.scalar_step(1.6, .3, 2., 5., .05), [2.25, -6, -3, 1.9, .45])
        np.testing.assert_allclose(impl.scalar_step(0., 0., -2., 1., .1), [1, 4, -2, -.4, .2])

    def test_linear(self):
        rng = np.random.default_rng(17)
        x, w, b, upstream = (rng.normal(size=s) for s in [(3, 2), (2, 4), (4,), (3, 4)])
        out = impl.linear_forward(x, w, b)
        self.assertEqual(out.shape, (3, 4))
        self.assertAlmostEqual(out[0, 0], sum(x[0, i]*w[i, 0] for i in range(2)) + b[0])
        fn = lambda: float((impl.linear_forward(x, w, b)*upstream).sum())
        for analytic, arr in zip(impl.linear_backward(x, w, upstream), [x, w, b]):
            self.assertEqual(analytic.shape, arr.shape)
            np.testing.assert_allclose(analytic, numeric_gradient(fn, arr), atol=1e-7, rtol=1e-5)

    def test_relu(self):
        x = np.array([[-2., 0., 3.], [4., -1., 2.]])
        dout = np.array([[2., 3., 4.], [-1., 5., 6.]])
        saved = dout.copy()
        np.testing.assert_array_equal(impl.relu_forward(x), [[0, 0, 3], [4, 0, 2]])
        np.testing.assert_array_equal(impl.relu_backward(x, dout), [[0, 0, 4], [-1, 0, 6]])
        np.testing.assert_array_equal(dout, saved)

    def test_ce(self):
        logits = np.array([[.5, -1., 2.], [3., 2., -2.]])
        labels = np.array([2, 0])
        saved = logits.copy()
        loss, grad = impl.softmax_ce(logits, labels)
        fn = lambda: impl.softmax_ce(logits, labels)[0]
        np.testing.assert_allclose(grad, numeric_gradient(fn, logits), atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(grad.sum(axis=1), 0, atol=1e-12)
        np.testing.assert_array_equal(logits, saved)
        loss2, grad2 = impl.softmax_ce(np.concatenate([logits, logits]), np.tile(labels, 2))
        self.assertAlmostEqual(loss, loss2)
        np.testing.assert_allclose(grad2[:2], grad/2)

    def test_ce_stability(self):
        scores = np.array([[1000., 0., -1000.], [-1000., 1000., 0.]])
        labels = np.array([2, 0])
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            loss, grad = impl.softmax_ce(scores, labels)
        self.assertTrue(np.isfinite(grad).all())
        self.assertAlmostEqual(loss, 2000.)
        shifted_loss, shifted_grad = impl.softmax_ce(scores + 12345., labels)
        self.assertAlmostEqual(loss, shifted_loss)
        np.testing.assert_allclose(grad, shifted_grad)

    def test_network_gradients(self):
        rng = np.random.default_rng(31)
        p = impl.init_params(11, (3, 4, 3, 3), np.float64)
        x, y = rng.normal(size=(5, 3)), np.array([0, 1, 2, 1, 0])
        saved = {k: v.copy() for k, v in p.items()}
        logits, cache = impl.forward(p, x)
        self.assertEqual(logits.shape, (5, 3))
        _, dl = impl.softmax_ce(logits, y)
        grads = impl.backward(p, cache, dl)
        self.assertEqual(set(grads), set(p))
        fn = lambda: impl.softmax_ce(impl.forward(p, x)[0], y)[0]
        self.assertGreater(min(np.min(np.abs(cache[1])), np.min(np.abs(cache[3]))), 1e-4)
        for key in p:
            np.testing.assert_array_equal(p[key], saved[key])
            np.testing.assert_allclose(grads[key], numeric_gradient(fn, p[key]), atol=2e-7, rtol=2e-4)

    def test_update(self):
        p = {'a': np.array([1., 2.]), 'b': np.array([3.])}
        g = {'a': np.array([2., -1.]), 'b': np.array([.5])}
        impl.sgd_update(p, g, .1)
        np.testing.assert_allclose(p['a'], [.8, 2.1])
        np.testing.assert_allclose(p['b'], [2.95])

    def test_small_overfit(self):
        # Artificial calibration task, not a generalization benchmark.
        rng = np.random.default_rng(9)
        x = np.repeat(np.eye(3), 8, axis=0) + rng.normal(0, .02, (24, 3))
        y = np.repeat(np.arange(3), 8)
        p = impl.init_params(3, (3, 8, 8, 3), np.float64)
        initial = evaluate(impl, p, x, y)[0]
        for _ in range(300):
            z, cache = impl.forward(p, x)
            _, dz = impl.softmax_ce(z, y)
            impl.sgd_update(p, impl.backward(p, cache, dz), .1)
        loss, acc = evaluate(impl, p, x, y)
        self.assertLess(loss, min(.1, initial * .2))
        self.assertGreater(acc, .95)

    def test_checkpoint_and_eval(self):
        rng = np.random.default_rng(8)
        x, y = rng.normal(size=(7, 3)), np.arange(7) % 3
        p = impl.init_params(3, (3, 4, 4, 3), np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'weights.npz'
            save_params(path, p)
            restored = load_params(path)
            np.testing.assert_array_equal(impl.forward(p, x)[0], impl.forward(restored, x)[0])
        saved = {k: v.copy() for k, v in p.items()}
        np.testing.assert_allclose(evaluate(impl, p, x, y, 3), evaluate(impl, p, x, y, 7))
        for key in p:
            np.testing.assert_array_equal(p[key], saved[key])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default=str(Path(__file__).with_name('todo.py')))
    
    ap.add_argument('--part', choices=['scalar', 'linear', 'relu', 'ce', 'network', 'all'], default='all')
    args = ap.parse_args()
    impl = load_impl(args.impl)
    groups = {'scalar': ['scalar'], 'linear': ['linear'], 'relu': ['relu'],
              'ce': ['ce', 'ce_stability'],
              'network': ['network_gradients', 'update', 'small_overfit', 'checkpoint_and_eval']}
    names = [n for values in groups.values() for n in values] if args.part == 'all' else groups[args.part]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Checks('test_'+n) for n in names))
    raise SystemExit(0 if result.wasSuccessful() else 1)
