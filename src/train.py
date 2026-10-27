"""Provided training driver. Students implement the functions in todo.py."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from common import load_impl, save_params, evaluate
from data import load_data, DEFAULT_DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default=str(Path(__file__).with_name('todo.py')))
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--out', type=Path, default=Path('runs/baseline'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--train-size', type=int, default=6000)
    ap.add_argument('--val-size', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--hidden', type=int, nargs=2, default=[32, 16])
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or min(args.hidden) < 1:
        ap.error('epochs, batch size, lr and hidden dimensions must be positive')
    impl = load_impl(args.impl)
    x, y, vx, vy = load_data(args.data_dir, train_size=args.train_size, val_size=args.val_size)
    params = impl.init_params(args.seed, (784, *args.hidden, 10))
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(numpy_version=np.__version__, split_seed=2026, device='CPU')
    (args.out/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    start = time.perf_counter()
    best = -1.0
    with (args.out/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'elapsed_seconds'])
        for epoch in range(args.epochs + 1):
            if epoch:
                order = rng.permutation(len(y))
                for offset in range(0, len(y), args.batch_size):
                    ids = order[offset:offset+args.batch_size]
                    logits, cache = impl.forward(params, x[ids])
                    loss, dlogits = impl.softmax_ce(logits, y[ids])
                    grads = impl.backward(params, cache, dlogits)
                    if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads.values()):
                        raise FloatingPointError('Nonfinite loss/gradient. Check CE stability and learning rate.')
                    impl.sgd_update(params, grads, args.lr)
            tl, ta = evaluate(impl, params, x, y)
            vl, va = evaluate(impl, params, vx, vy)
            row = [epoch, tl, ta, vl, va, time.perf_counter()-start]
            writer.writerow(row)
            stream.flush()
            print(f'epoch={epoch:02d} train_ce={tl:.4f} train_acc={ta:.3f} val_ce={vl:.4f} val_acc={va:.3f}', flush=True)
            if va > best:
                best = va
                save_params(args.out/'best.npz', params)
    save_params(args.out/'last.npz', params)
    print('Saved best.npz, last.npz, metrics.csv and config.json', flush=True)


if __name__ == '__main__':
    main()
