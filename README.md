# EPFL CS-433 Project 1 — Heart Disease Classification

## Final submission

Reproduces **AIcrowd Submission #337816** (F1 **0.441**, accuracy **0.880**)
with **BRFSS 2015**, codebook-aware preprocessing and sparse L1 logistic regression.
Only Python standard library and NumPy are needed for the pipeline.

## Quick start

1. Download the official course datasets and put `x_train.csv`, `y_train.csv`,
   and `x_test.csv` in `dataset/` (do not commit the dataset).
2. Install `pip install -r requirements.txt`.
3. Run `python run.py`.
4. Upload the generated `submission.csv` to AIcrowd.

The final model is **broad L1**, with **lambda=0.0005**, **1600 iterations**,
**F1 threshold=0.208452**. It trains on all **328,135** labeled rows.
Survey metadata, geographic IDs and phone sampling fields are excluded;
health-related optional modules remain eligible. The encoder fits its categories,
medians, missing indicators and standardization parameters on the training partition.
Final encoded dimension in the validated run was 553; 201 coefficients were nonzero.

## Code structure

- `implementations.py`: original six required numerical methods (unchanged
  signatures) plus the extra proximal-FISTA logistic models.
- `preprocessing.py`: exact V6 BRFSS cleaning rules, feature pools and encoder.
- `helpers.py`: provided CSV utilities, V6 data loader, validation splits, metrics
  and batch test submission writer.
- `run.py`: **only** final frozen model refit and `submission.csv` generation.
- `experiments.py`: eight-candidate CV ablation and the original V6 full-data
  lambda tuning / threshold selection / convergence diagnostics.

## Reproduce experiments (optional, slower)

The original eight-model comparison on a 20% subsample can be run from Python:

```python
from helpers import load_brfss
from experiments import CANDIDATES, evaluate_pipeline
X, y, _, _ = load_brfss('dataset', sample_fraction=0.2)
result = evaluate_pipeline(X, y, candidates=CANDIDATES, folds=3,
                           iterations=600, criterion='auc', out_dir='step3_results')
```

Full-data tuning (expensive):

```bash
python experiments.py --data dataset
```

This executes 3-fold CV on the 80% development set for lambda candidates
`0.00025, 0.0005, 0.00075, 0.001`, tunes an exact F1 threshold on a
separate 20% partition, performs a convergence comparison (800 vs 1600
iterations) and refits on all labels. The resulting output is
`submission_refined_l1.csv`, distinct from `run.py`'s `submission.csv`.

**Important:** `run.py` does **not** rerun CV; it is frozen to the
parameters used for the graded submission. Do not choose model settings by
looking at the AIcrowd hidden-test labels or leaderboard scores.

## Reproducibility verification

After running `python run.py` with the same original course CSVs:

```python
import numpy as np
old = np.genfromtxt('submission_refined_l1.csv', delimiter=',', skip_header=1, dtype=int)
new = np.genfromtxt('submission.csv', delimiter=',', skip_header=1, dtype=int)
assert np.array_equal(old, new), 'Final submission differs from V6'
print('Exact reproduction passed')
```

Keep the original `submission_refined_l1.csv` locally for this comparison.
If an external BLAS environment produces boundary-point differences,
check optimizer and feature ordering before treating the output as a reproduction.

## Course grading and report

Keep all six required functions and their prescribed signatures in
`implementations.py`. Run the official public grading tests. The final
report is a separate document (at most two pages) and should identify
the team members, methods, CV comparisons and final submission ID.

## Integration notes

This combines V6 `step3_pipeline.py` and `tune_full.py` into five Python
files. Teammates' six required algorithm implementations and original CSV
helpers remain intact. An earlier `preprocessing.py` supplied optional
dataframe-like utility functions; these are kept where useful, but its
mean-imputation shortcut and the separate `data_cleaning.py` are not used
in the final model because they differ from the graded V6 encoder.

Original V6 prediction code was preserved verbatim where possible, rather
than changed to an alternative optimization implementation.
