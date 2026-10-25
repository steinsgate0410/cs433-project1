"""CS-433 experiments: eight-model comparison and full-data L1/F1 refinement.

Full-data tuning: python experiments.py --data dataset
Eight-model comparison: import evaluate_pipeline, CANDIDATES from this module.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path
import numpy as np

from preprocessing import BRFSSEncoder
from implementations import fit_candidate, probabilities
from helpers import (load_brfss, metrics, stratified_folds, stratified_split,
                     select_threshold, submission)
from run import fit_final

# Candidate configurations: feature representation and estimator jointly chosen by CV.
CANDIDATES = [
    # Retain the earlier experiments as controls; broad grows the candidate pool.
    dict(name='baseline_l2', features='base', model='l2', lam=0.005),
    dict(name='curated_l2', features='full', model='l2', lam=0.005),
    dict(name='curated_l1', features='full', model='l1', lam=0.002),
    dict(name='broad_l2', features='broad', model='l2', lam=0.005),
    dict(name='broad_l1_sparse', features='broad', model='l1', lam=0.002),
    dict(name='broad_l1', features='broad', model='l1', lam=0.001),
    dict(name='broad_elasticnet', features='broad', model='elasticnet',
         lam=0.002, l1_ratio=0.5),
    dict(name='broad_l1_weighted', features='broad', model='l1', lam=0.002,
         class_weight='balanced'),
]
# Optional extra experiment, uncomment in Notebook if needed:
OPTIONAL_REFIT = dict(name='broad_l1_l2_refit',features='broad',model='l1_l2',
                      lam=0.002,refit_lam=0.005)

def _save_rows(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _raw_selected(encoded_names, w, eps=1e-6):
    """Group nonzero encoded dummy coefficients by their ORIGINAL raw field."""
    return {n.split('==')[0].split(':')[0]
            for n,v in zip(encoded_names,w) if abs(v)>eps}


def evaluate_pipeline(X,y,candidates=CANDIDATES,folds=3,iterations=600,
                      criterion='auc',threshold_metric='f1',seed=2026,
                      auc_tolerance=0.001,out_dir=None):
    """3-stage local protocol: development CV -> separate threshold -> evaluation.

    CV preprocessor/feature selection/class weights are fit on each training fold.
    Compares broad vs curated pools on identical row splits. Saves original-field
    selection frequencies; threshold/evaluation labels are not involved in CV.
    """
    if criterion not in ('auc','ap','bce'):
        raise ValueError('criterion must be auc, ap, or bce')
    fit_idx,tune_idx,test_idx=stratified_split(y,seed=seed)
    Xdev,ydev=X[fit_idx],y[fit_idx]
    splits=list(stratified_folds(ydev,k=folds,seed=seed+1))
    cv_rows=[]; freq={c['name']:{} for c in candidates}
    # Share fitted fold preprocessors among candidates with the same pool.
    pools=list(dict.fromkeys(c['features'] for c in candidates))
    for pool in pools:
        group=[c for c in candidates if c['features']==pool]
        for fold,(tr,va) in enumerate(splits,1):
            enc=BRFSSEncoder(pool).fit(Xdev[tr])
            xtr,xva=enc.transform(Xdev[tr]),enc.transform(Xdev[va])
            for cand in group:
                model=fit_candidate(xtr,ydev[tr],cand,iterations)
                m=metrics(ydev[va],probabilities(xva,model))
                weights=model['estimator']['w']
                selected=_raw_selected(np.asarray(enc.names_)[model['mask']],weights)
                m.update(candidate=cand['name'],pool=pool,fold=fold,
                         nonzero=int(np.count_nonzero(np.abs(weights)>1e-6)),
                         selected_raw=len(selected),dim=xtr.shape[1],
                         residual=model['estimator']['prox_gradient_norm'])
                cv_rows.append(m)
                # Frequency uses groups of dummies, not individual dummies.
                for name in selected:
                    freq[cand['name']][name]=freq[cand['name']].get(name,0)+1
    summary=[]
    for cand in candidates:
        rows=[r for r in cv_rows if r['candidate']==cand['name']]
        summary.append(dict(candidate=cand['name'],pool=cand['features'],
                            model=cand['model'],class_weight=str(cand.get('class_weight')),
                            mean_auc=float(np.mean([r['auc'] for r in rows])),
                            std_auc=float(np.std([r['auc'] for r in rows],ddof=1)),
                            mean_ap=float(np.mean([r['ap'] for r in rows])),
                            mean_bce=float(np.mean([r['bce'] for r in rows])),
                            mean_nonzero=float(np.mean([r['nonzero'] for r in rows])),
                            mean_selected_raw=float(np.mean([r['selected_raw'] for r in rows])),
                            mean_dim=float(np.mean([r['dim'] for r in rows])),
                            mean_residual=float(np.mean([r['residual'] for r in rows]))))
    # Near-tied AUC/AP/BCE: prefer fewer coefficients. Heuristic, not statistical CI.
    key='mean_'+criterion
    best=(min if criterion=='bce' else max)(r[key] for r in summary)
    near=[r for r in summary if (r[key]<=best+auc_tolerance if criterion=='bce'
                                 else r[key]>=best-auc_tolerance)]
    pick=min(near,key=lambda r:(r['mean_nonzero'],
                                 -r['mean_auc'],-r['mean_ap']))
    chosen=next(c for c in candidates if c['name']==pick['candidate'])
    encoder=BRFSSEncoder(chosen['features']).fit(Xdev)
    fitted=fit_candidate(encoder.transform(Xdev),ydev,chosen,iterations)
    threshold=select_threshold(y[tune_idx],
                               probabilities(encoder.transform(X[tune_idx]),fitted),
                               metric=threshold_metric)
    final=metrics(y[test_idx],probabilities(encoder.transform(X[test_idx]),fitted),threshold)
    final.update(candidate=chosen['name'],threshold=threshold,
                 n_train=len(fit_idx),n_tune=len(tune_idx),n_test=len(test_idx))
    frequency=[dict(candidate=c['name'],raw_feature=f,fold_count=count,
                    frequency=count/folds)
               for c in candidates for f,count in sorted(freq[c['name']].items(),
                                                key=lambda item:(-item[1],item[0]))]
    if out_dir is not None:
        out=Path(out_dir);out.mkdir(exist_ok=True,parents=True)
        _save_rows(out/'cv_folds.csv',cv_rows)
        _save_rows(out/'cv_summary.csv',summary)
        _save_rows(out/'evaluation.csv',[final])
        if frequency:_save_rows(out/'feature_stability.csv',frequency)
    return dict(cv=summary,cv_folds=cv_rows,chosen=chosen,threshold=threshold,
                test_metrics=final,encoder=encoder,fitted=fitted,
                stability=frequency,test_indices=test_idx)


def selected_features(encoder, fitted, top=20, eps=1e-6):
    """List nonzero encoded columns and standardized coefficients (not causal effects)."""
    names = np.asarray(encoder.names_, dtype=object)[fitted['mask']]
    w = fitted['estimator']['w']
    entries = [(str(name), float(coef)) for name,coef in zip(names,w)
               if abs(coef) > eps]
    return sorted(entries, key=lambda pair: -abs(pair[1]))[:top]



def exact_f1_threshold(y, p):
    """Max-F1 threshold over every distinct prediction score; O(n log n)."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.all(np.isfinite(p)):
        raise ValueError('Invalid labels/probabilities for threshold tuning')
    order = np.argsort(-p, kind='mergesort')
    scores, labels = p[order], y[order]
    # Group ties: >= threshold must predict an entire tied-score block positive.
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(y)-1]
    tp = np.cumsum(labels)[ends]
    f1 = 2 * tp / np.maximum(ends + 1 + int(labels.sum()), 1)
    winners = np.flatnonzero(f1 >= f1.max() - 1e-14)
    chosen = winners[np.argmin(np.abs(scores[ends[winners]] - 0.5))]
    return float(scores[ends[chosen]]), float(f1[chosen])


def csv_rows(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lam_grid = sorted(set(args.lambdas))
    if args.check_iterations < 0 or (args.check_iterations > 0 and args.check_iterations >= args.iterations):
        raise ValueError("--check-iterations must be 0 or smaller than --iterations")
    if not lam_grid or any(lam <= 0 for lam in lam_grid):
        raise ValueError('Provide positive lambdas')

    X, y, _, _ = load_brfss(args.data, sample_fraction=1.0)
    dev, tune, unused = stratified_split(y, ratios=(0.8, 0.2, 0.0), seed=args.seed)
    assert not len(unused) and not np.intersect1d(dev, tune).size
    ydev = y[dev]
    print(f'FULL DATA: {len(y):,} rows, positive={y.mean():.2%}; '
          f'CV-development={len(dev):,}, threshold-tuning={len(tune):,}', flush=True)
    print('Model: broad L1; lambda grid:', lam_grid, flush=True)

    # All model comparisons use identical folds and the same training-fold encoder.
    out_of_fold = {lam: np.full(len(dev), np.nan, dtype=np.float64) for lam in lam_grid}
    fold_rows = []
    for fold, (tr, va) in enumerate(stratified_folds(ydev, args.folds, seed=args.seed + 1), 1):
        print(f'\nFold {fold}/{args.folds}: fit preprocessing...', flush=True)
        encoder = BRFSSEncoder('broad').fit(X[dev[tr]])
        xtr = encoder.transform(X[dev[tr]])
        xva = encoder.transform(X[dev[va]])
        print(f'  train={len(tr):,}, valid={len(va):,}, encoded dimensions={xtr.shape[1]}', flush=True)
        for lam in lam_grid:
            cand = dict(name=f'l1_{lam:g}', features='broad', model='l1', lam=float(lam))
            fitted = fit_candidate(xtr, ydev[tr], cand, iterations=args.iterations)
            pred = probabilities(xva, fitted)
            out_of_fold[lam][va] = pred
            m = metrics(ydev[va], pred)
            state = fitted['estimator']
            row = dict(fold=fold, lam=lam, auc=m['auc'], ap=m['ap'], bce=m['bce'],
                       nonzero=int(np.count_nonzero(np.abs(state['w']) > 1e-6)),
                       dim=xtr.shape[1], iterations=state['iterations'],
                       residual=state['prox_gradient_norm'])
            fold_rows.append(row)
            print(f"  λ={lam:g} AUC={m['auc']:.5f} AP={m['ap']:.5f} "
                  f"BCE={m['bce']:.5f} nonzero={row['nonzero']} "
                  f"iter={row['iterations']} residual={row['residual']:.3g}", flush=True)
        del encoder, xtr, xva
        csv_rows(outdir / 'lambda_cv_folds.csv', fold_rows)

    summaries = []
    for lam in lam_grid:
        oof = out_of_fold[lam]
        if not np.all(np.isfinite(oof)):
            raise RuntimeError('Some development rows have no out-of-fold predictions')
        threshold, f1 = exact_f1_threshold(ydev, oof)
        m = metrics(ydev, oof, threshold)
        related = [r for r in fold_rows if r['lam'] == lam]
        summaries.append(dict(lam=lam, cv_f1=f1, cv_threshold=threshold,
                              cv_auc=m['auc'], cv_ap=m['ap'], cv_bce=m['bce'],
                              mean_nonzero=float(np.mean([r['nonzero'] for r in related])),
                              max_residual=max(r['residual'] for r in related)))
    # F1 first (leaderboard metric); AP breaks ties, fewer columns breaks further ties.
    selected = max(summaries, key=lambda r:(r['cv_f1'], r['cv_ap'], -r['mean_nonzero']))
    csv_rows(outdir/'lambda_cv_summary.csv', summaries)
    print('\nOOF-CV summary (F1 optimized on development OOF; selection estimate, not a test score):')
    for r in summaries:
        print(f"  λ={r['lam']:<7g} F1={r['cv_f1']:.5f} AP={r['cv_ap']:.5f} "
              f"AUC={r['cv_auc']:.5f} BCE={r['cv_bce']:.5f} "
              f"nonzero={r['mean_nonzero']:.1f}", flush=True)
    lam = float(selected['lam'])
    candidate = dict(name='broad_l1_tuned', features='broad', model='l1', lam=lam)
    print(f'\nChosen λ={lam:g} (development CV only).', flush=True)

    # The threshold partition is excluded from every fold and from λ selection.
    print('Fit on CV-development partition; optimize F1 threshold separately...', flush=True)
    encoder_dev = BRFSSEncoder('broad').fit(X[dev])
    xdev = encoder_dev.transform(X[dev])
    fitted_dev = fit_candidate(xdev, ydev, candidate, iterations=args.iterations)

    # Compare two iteration budgets on EXACTLY the same training matrix.
    # No tune labels or leaderboard score are used for this convergence check.
    convergence = None
    if args.check_iterations:
        print(f'Convergence check: {args.check_iterations} vs {args.iterations} iterations...', flush=True)
        short_fit = fit_candidate(xdev, ydev, candidate, iterations=args.check_iterations)
        short = short_fit['estimator']; long = fitted_dev['estimator']
        probe = xdev[:min(len(xdev), 2048)]
        p_short = probabilities(probe, short_fit)
        p_long = probabilities(probe, fitted_dev)
        convergence = dict(
            lambda_=lam,
            short_budget=args.check_iterations, long_budget=args.iterations,
            short_iters=short['iterations'], long_iters=long['iterations'],
            short_objective=short['objective'], long_objective=long['objective'],
            short_residual=short['prox_gradient_norm'],
            long_residual=long['prox_gradient_norm'],
            weight_l2_change=float(np.linalg.norm(long['w']-short['w'])),
            intercept_abs_change=abs(float(long['b']-short['b'])),
            max_prob_change=float(np.max(np.abs(p_long-p_short))),
            mean_prob_change=float(np.mean(np.abs(p_long-p_short))),
        )
        csv_rows(outdir/'convergence_check.csv', [convergence])
        print(f"  objective: {short['objective']:.8f} -> {long['objective']:.8f}; "
              f"stationarity residual: {short['prox_gradient_norm']:.3g} -> "
              f"{long['prox_gradient_norm']:.3g}; "
              f"max Δp (first {len(probe):,} development rows)="
              f"{convergence['max_prob_change']:.5g}", flush=True)
        del short_fit, probe, p_short, p_long

    tune_prob = probabilities(encoder_dev.transform(X[tune]), fitted_dev)
    threshold, tune_f1 = exact_f1_threshold(y[tune], tune_prob)
    tune_metrics = metrics(y[tune], tune_prob, threshold)
    print(f'F1-tuned threshold={threshold:.6f}; tuning-set F1={tune_f1:.5f} '
          f"precision={tune_metrics['precision']:.5f} "
          f"recall={tune_metrics['recall']:.5f}", flush=True)
    del encoder_dev, xdev, fitted_dev, tune_prob

    config = {'candidate': candidate, 'threshold': threshold,
              'selected_by': f'pooled {args.folds}-fold OOF F1 (dev only)',
              'threshold_selected_by': '20% disjoint label-tuning partition',
              'n_labeled': int(len(y)), 'cv_development': int(len(dev)),
              'threshold_tuning': int(len(tune)), 'folds': args.folds,
              'iterations': args.iterations, 'check_iterations': args.check_iterations,
              'convergence_check': convergence, 'seed': args.seed,
              'cv_selection': selected, 'threshold_tuning_metrics': tune_metrics}
    (outdir/'best_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    print('\nRefit selected recipe on ALL labeled rows...', flush=True)
    enc_final, fitted_final = fit_final(X, y, candidate, iterations=args.iterations)
    final_state = fitted_final['estimator']
    print(f"  final dimensions={len(enc_final.names_)}, "
          f"nonzero={int(np.count_nonzero(np.abs(final_state['w']) > 1e-6))}, "
          f"iterations={final_state['iterations']}, "
          f"residual={final_state['prox_gradient_norm']:.3g}", flush=True)
    if max(selected['max_residual'], final_state['prox_gradient_norm']) > 1e-3:
        print('WARNING: optimizer stationarity residual > 1e-3; check convergence before submitting.', flush=True)
    # Save the entire fitted encoder and estimator for reproducible inference.
    # Only unpickle checkpoint files you created and trust.
    with (outdir / 'best_model.pkl').open('wb') as fh:
        pickle.dump({'encoder': enc_final, 'model': fitted_final,
                     'threshold': threshold, 'candidate': candidate},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    del X, y

    test_file = Path(args.data) / 'x_test.csv'
    if test_file.exists():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission(args.data, enc_final, fitted_final, threshold,
                   filename=str(output_path))
        print(f'AIcrowd submission saved: {output_path}', flush=True)
    else:
        print(f'No {test_file}; tuning/config completed but cannot generate submission.', flush=True)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='dataset', help='Directory containing x_train.csv, y_train.csv and x_test.csv')
    p.add_argument('--lambdas', type=float, nargs='+', default=[0.00025,0.0005,0.00075,0.001])
    p.add_argument('--folds', type=int, default=3)
    p.add_argument('--iterations', type=int, default=1600)
    p.add_argument('--check-iterations', type=int, default=800,
                   help='Compare selected model at this iteration budget vs --iterations; 0 disables')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--outdir', default='refined_tuning_results')
    p.add_argument('--output', default='submission_refined_l1.csv')
    args = p.parse_args()
    if args.folds < 2 or args.iterations < 1:
        p.error('At least 2 CV folds and 1 iteration are required')
    run(args)


if __name__ == '__main__':
    main()
