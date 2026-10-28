# Heart Disease Classification

EPFL CS-433 Machine Learning, Project 1

Boyang Hu, Jinrun Jiang and Yunqing Pan (equal contribution)

This project uses NumPy logistic regression to classify reported heart disease from BRFSS survey data. The labeled data contain 328,135 respondents and 321 features; 8.83% of labels are positive.

## Method and results

We use a stratified split with seed 42: 75% for fitting, 15% for tuning and 10% for holdout evaluation. Preprocessing includes feature filtering, missing-value handling, categorical encoding and numeric scaling. The notebook compares three feature sets (Compact, Broad and Interactions) and three L2 regularization strengths. It selects a model and decision threshold using tuning F1.

The selected model uses Broad features, regularization `0.0001` and threshold `0.185`. On the 32,814-row holdout set, it achieves **F1 0.4385** and **accuracy 0.8667**. The same fitted model at threshold `0.5` achieves F1 0.2237 and accuracy 0.9133. The notebook then refits the selected configuration on all labeled rows and predicts the test set.

## Run the notebook

Download the course dataset from the [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) and place these files in `dataset/`:

```text
dataset/
  x_train.csv
  y_train.csv
  x_test.csv
```

The reported run used Python 3.13.15, NumPy 2.5.2, Matplotlib 3.11.0 and Jupyter Notebook 7.5.7. For example, create an environment and start the notebook from the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook run.ipynb
```

Select the Python kernel from this environment, then choose **Restart Kernel and Run All Cells**. Small numerical differences across NumPy or BLAS environments can change the selected model when tuning scores are close.

Running the notebook overwrites `submission.csv`, `results/model.npz` and the plots in `figures/`. The submission contains 109,379 predictions in the original test ID order, with labels `-1` and `+1`.

## Files

- `run.ipynb` — data exploration, preprocessing, model selection, evaluation and prediction export.
- `implementations.py` — the six required course methods. The notebook experiments use their own L-BFGS logistic optimizer.
- `helpers.py` — data loading and CSV submission functions.
- `report/report.pdf` — final two-page report; `report/report.tex` is its editable source.
- `results/model.npz` — final weights and threshold; preprocessing rules are learned again when the notebook runs.
- `figures/` — data overview and evaluation plots.
