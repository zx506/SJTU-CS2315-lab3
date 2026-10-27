"""Load weights in a new process. No training or gradients are needed to predict."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import argparse
from pathlib import Path
import numpy as np
from common import load_impl, load_params, evaluate
from data import load_data, DEFAULT_DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--impl', default=str(Path(__file__).with_name('todo.py')))
    ap.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--val-size', type=int, default=1000)
    ap.add_argument('--index', type=int, default=0)
    args = ap.parse_args()
    impl, params = load_impl(args.impl), load_params(args.checkpoint)
    if args.split == 'test':
        x, y = load_data(args.data_dir, split='test')
    else:
        _, _, x, y = load_data(args.data_dir, val_size=args.val_size)
    if not 0 <= args.index < len(y):
        ap.error('index outside selected split')
    loss, acc = evaluate(impl, params, x, y)
    logits, _ = impl.forward(params, x[args.index:args.index+1])
    pred = int(logits.argmax(axis=1)[0])
    print(f'split={args.split} n={len(y)} ce={loss:.6f} accuracy={acc:.4f}')
    print(f'index={args.index} true={y[args.index]} predicted={pred}')
    shades = ' .:-=+*#%@'
    for row in x[args.index].reshape(28, 28):
        print(''.join(shades[min(9, int(float(v)*9))]*2 for v in row))


if __name__ == '__main__':
    main()
