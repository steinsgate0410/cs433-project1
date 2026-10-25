"""Reproduce AIcrowd submission #337816 (F1 0.441) with frozen V6 recipe.

Usage: python run.py
Optional: python run.py --data dataset --output submission.csv
"""
import argparse
import numpy as np
from preprocessing import BRFSSEncoder
from implementations import fit_candidate
from helpers import load_brfss, submission

LAMBDA = 0.0005
THRESHOLD = 0.208452
ITERATIONS = 1600

def fit_final(X, y, candidate, iterations=ITERATIONS):
    """Refit encoder and estimator on all provided labeled rows."""
    encoder = BRFSSEncoder(candidate['features']).fit(X)
    fitted = fit_candidate(encoder.transform(X), y, candidate, iterations=iterations)
    return encoder, fitted

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='dataset')
    parser.add_argument('--output', default='submission.csv')
    args = parser.parse_args()
    X, y, _, _ = load_brfss(args.data, sample_fraction=1.0)
    candidate = dict(name='broad_l1_v6', features='broad', model='l1', lam=LAMBDA)
    encoder, model = fit_final(X, y, candidate, iterations=ITERATIONS)
    path = submission(args.data, encoder, model, THRESHOLD, filename=args.output)
    weights = model['estimator']['w']
    print(f'Saved {path}; train={len(y):,}, dimension={len(encoder.names_)}, '
          f'nonzero={np.count_nonzero(np.abs(weights) > 1e-6)}, '
          f'lambda={LAMBDA}, threshold={THRESHOLD}, iterations={ITERATIONS}')

if __name__ == '__main__':
    main()
