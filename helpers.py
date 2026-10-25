"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

# Reproduction helpers from the tested V6 pipeline.
from pathlib import Path
from preprocessing import ALL_NAMES, BASE_CAT, BASE_NUM, EXTRA_CAT, EXTRA_NUM, LEAKAGE, SURVEY_METADATA
from implementations import probabilities

def _number(value):
    s = str(value).strip()
    if s.lower() in ('', 'na', 'nan', 'null', '?'):
        return np.nan
    return float(s)


def _csv_features(path, feature_names=ALL_NAMES):
    """Read only requested columns; IDs remain separate from model input."""
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = [s.strip() for s in next(reader)]
        # Raw input may contain leakage/proxy columns (e.g. HAREHAB1),
        # but they are NOT model inputs. Refuse only if caller selects them.
        forbidden = (LEAKAGE | SURVEY_METADATA).intersection(feature_names)
        if forbidden:
            raise ValueError(f'Forbidden model feature selection: {sorted(forbidden)}')
        absent = [name for name in ['Id', *(BASE_CAT+BASE_NUM+EXTRA_CAT+EXTRA_NUM)] if name not in header]
        if absent:
            raise ValueError(f'Missing required columns: {absent}')
        indices = [header.index(c) if c in header else None for c in feature_names]
        id_idx = header.index('Id')
        ids, rows = [], []
        for row in reader:
            ids.append(int(float(row[id_idx])))
            rows.append([_number(row[i]) if i is not None else np.nan for i in indices])
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate feature IDs')
    return np.asarray(rows, dtype=np.float32), ids


def load_brfss(folder='dataset', sample_fraction=0.2, seed=42):
    """Return (raw X, y in {0,1}, ids, feature names); match labels by Id."""
    folder = Path(folder)
    X, ids = _csv_features(folder / 'x_train.csv')
    with open(folder / 'y_train.csv', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if 'Id' not in reader.fieldnames or '_MICHD' not in reader.fieldnames:
            raise ValueError('Expected Id,_MICHD in y_train.csv')
        label_dict = {}
        for row in reader:
            idx = int(float(row['Id']))
            value = int(float(row['_MICHD']))
            if idx in label_dict or value not in (-1, 1):
                raise ValueError(f'Duplicate Id or invalid label for {idx}')
            label_dict[idx] = int(value == 1)
    if not all(i in label_dict for i in ids):
        raise ValueError('Some feature Ids are not found in labels')
    y = np.fromiter((label_dict[i] for i in ids), dtype=np.int8, count=len(ids))
    if not 0 < sample_fraction <= 1:
        raise ValueError('sample_fraction must be in (0,1]')
    if sample_fraction < 1:
        rng = np.random.default_rng(seed)
        keep = rng.random(len(y)) < sample_fraction
        X, y, ids = X[keep], y[keep], ids[keep]
    return X, y, ids, ALL_NAMES[:]


def stratified_split(y, ratios=(0.70, 0.15, 0.15), seed=2026):
    """Return disjoint index arrays; ratios correspond to train, threshold, test."""
    if not np.isclose(sum(ratios), 1):
        raise ValueError('split ratios must sum to 1')
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        a = int(len(indices) * ratios[0]); b = int(len(indices) * (ratios[0] + ratios[1]))
        for group, chunk in zip(parts, (indices[:a], indices[a:b], indices[b:])):
            group.extend(chunk.tolist())
    return tuple(rng.permutation(g).astype(int) for g in parts)


def stratified_folds(y, k=3, seed=2027):
    rng = np.random.default_rng(seed)
    by_class = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        by_class.append(np.array_split(idx, k))
    for fold in range(k):
        val = np.concatenate([blocks[fold] for blocks in by_class])
        train = np.setdiff1d(np.arange(len(y)), val)
        yield train, val


def roc_auc(y, p):
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg: return float('nan')
    order = np.argsort(p, kind='mergesort')
    sorted_p, sorted_y = p[order], y[order]
    edges = np.r_[0, np.flatnonzero(np.diff(sorted_p) != 0)+1, len(p)]
    ranks = np.arange(1, len(p)+1, dtype=np.float64)
    rank_pos = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        rank_pos += sorted_y[left:right].sum() * (ranks[left]+ranks[right-1])/2
    return float((rank_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg))


def average_precision(y,p):
    n_pos = int(np.sum(y))
    if n_pos == 0: return float('nan')
    order = np.argsort(-p, kind='mergesort')
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)!=0),len(p)-1]
    tp = np.cumsum(ys)[ends]
    return float(np.sum((tp/(ends+1))*np.diff(np.r_[0,tp]))/n_pos)


def metrics(y,p,threshold=.5):
    y = np.asarray(y, dtype=np.int8)
    pred = p >= threshold
    tp = int(np.sum(pred & (y==1))); fp = int(np.sum(pred & (y==0)))
    fn = int(np.sum(~pred & (y==1))); tn = int(np.sum(~pred & (y==0)))
    prec = tp/max(tp+fp,1); recall = tp/max(tp+fn,1)
    q = np.clip(p, 1e-12, 1-1e-12)
    return {'auc':roc_auc(y,p),'ap':average_precision(y,p),
            'bce':float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q))),
            'accuracy':(tp+tn)/len(y), 'precision':prec,'recall':recall,
            'f1':2*prec*recall/max(prec+recall,1e-12),
            'tp':tp,'fp':fp,'tn':tn,'fn':fn}


def select_threshold(y, p, metric='f1'):
    """Use ONLY separate threshold-tuning labels; not model-CV or test labels."""
    if metric not in ('f1','accuracy'):
        raise ValueError("threshold metric must be 'f1' or 'accuracy'")
    # Quantiles concentrate trials around observed predictions.
    values = np.unique(np.r_[0., np.quantile(p, np.linspace(0.,1.,201)), .5, 1.])
    options = []
    for t in values:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tn = len(y) - tp - fp - fn
        score = ((tp + tn) / len(y) if metric == 'accuracy'
                 else 2*tp / max(2*tp+fp+fn, 1))
        options.append((score, float(t)))
    return max(options, key=lambda z:(z[0],-abs(z[1]-.5)))[1]


def submission(folder,encoder,fitted,threshold,filename='submission.csv',batch_size=5000):
    """Write Id,Prediction (-1/+1); batch test transformation to limit RAM."""
    folder=Path(folder)
    with open(folder/'x_test.csv',encoding='utf-8-sig',newline='') as inp, open(filename,'w',newline='') as out:
        reader=csv.reader(inp);header=[s.strip() for s in next(reader)]
        indexes=[header.index(c) if c in header else None for c in ALL_NAMES]
        idx_id=header.index('Id')
        writer=csv.writer(out);writer.writerow(['Id','Prediction'])
        buffer,ids=[],[]
        def flush():
            if not buffer: return
            x=np.asarray(buffer,dtype=np.float32)
            p=probabilities(encoder.transform(x),fitted)
            writer.writerows(zip(ids,np.where(p>=threshold,1,-1)))
            buffer.clear();ids.clear()
        for row in reader:
            ids.append(int(float(row[idx_id])))
            buffer.append([_number(row[i]) if i is not None else np.nan for i in indexes])
            if len(ids)>=batch_size: flush()
        flush()
    return str(filename)
