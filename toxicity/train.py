"""Local-only retrospective evaluation and trusted model scoring."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .data import FEATURES, audit, read_data
from .evaluation import error_slice, group_folds, metrics, select_threshold

CONFIG = dict(objective='binary', learning_rate=.025, num_leaves=31,
              min_child_samples=25, subsample=.85, subsample_freq=1,
              colsample_bytree=.55, reg_alpha=.2, reg_lambda=2.,
              n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def estimator(kind, seed, iterations=1200):
    model = (LGBMClassifier(**CONFIG, n_estimators=iterations, random_state=seed)
             if kind == 'lightgbm' else LogisticRegression(C=.25, solver='liblinear', max_iter=1000, random_state=seed))
    steps = [('variance', VarianceThreshold())]
    if kind == 'logistic':
        steps.append(('scale', MaxAbsScaler()))
    return Pipeline(steps+[('model', model)])


def inner_policy(X, y, groups, kind, seed, folds, cap):
    """Outer data must never be passed to this selection routine."""
    probability = np.zeros(len(y))
    iterations = []
    for fold, (train, valid) in enumerate(group_folds(y, groups, folds, seed)):
        pipe = estimator(kind, seed+fold, cap)
        # Fit preprocessing on inner TRAIN only, then transform inner validation.
        Xt = pipe[:-1].fit_transform(X[train], y[train])
        Xv = pipe[:-1].transform(X[valid])
        model = pipe.named_steps['model']
        if kind == 'lightgbm':
            model.fit(Xt, y[train], eval_X=Xv, eval_y=y[valid], eval_metric='binary_logloss',
                      callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
            iterations.append(int(model.best_iteration_ or cap))
        else:
            model.fit(Xt, y[train])
        probability[valid] = model.predict_proba(Xv)[:, 1]
    return select_threshold(y, probability), int(np.median(iterations)) if iterations else cap


def nested_evaluate(frame, groups, kind, *, outer=5, inner=3, cap=1200, seed=42):
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame.label.to_numpy(dtype=np.int8)
    probability = np.zeros(len(y))
    predicted = np.zeros(len(y), dtype=np.int8)
    high_weight = np.zeros(len(y), dtype=bool)
    fold_report = []
    for fold, (train, valid) in enumerate(group_folds(y, groups, outer, seed)):
        threshold, iterations = inner_policy(X[train], y[train], groups[train], kind, seed+100+fold, inner, cap)
        pipe = estimator(kind, seed+1000+fold, iterations)
        # Fixed iteration refit: outer validation is NOT an early-stopping eval_set.
        pipe.fit(X[train], y[train])
        probability[valid] = pipe.predict_proba(X[valid])[:, 1]
        predicted[valid] = probability[valid] >= threshold
        high_weight[valid] = frame.MolWt.to_numpy()[valid] > np.quantile(frame.MolWt.to_numpy()[train], .75)
        fold_report.append({'fold': fold, 'train_rows': len(train), 'valid_rows': len(valid),
                            'group_overlap': 0, 'threshold_from_inner_only': threshold,
                            'iterations_from_inner_only': iterations if kind == 'lightgbm' else None,
                            **metrics(y[valid], probability[valid], predicted[valid])})
        print(json.dumps({'model': kind, 'completed_outer_fold': fold+1, 'total': outer}), flush=True)
    result = {'model': kind, 'pooled_outer_oof': metrics(y, probability, predicted),
              'fixed_threshold_0_5_comparison': metrics(y, probability, (probability >= .5).astype(int)),
              'fold_f1_non_toxic_std_descriptive_not_confidence_interval': float(np.std([f['f1_non_toxic_label1'] for f in fold_report])),
              'folds': fold_report,
              'error_slices': {'high_MolWt_above_outer_train_q75': error_slice(y, predicted, high_weight),
                               'other_MolWt': error_slice(y, predicted, ~high_weight)}}
    return result


def run(data, output, prediction=None, *, outer=5, inner=3, cap=1200):
    if output.exists():
        raise FileExistsError('Use a new output directory; old results are never overwritten')
    frame = read_data(data)
    prediction_frame = read_data(prediction, labeled=False) if prediction else None
    quality, groups = audit(frame, prediction_frame)
    # Validate all outer/inner partitions BEFORE creating output or starting model fits.
    y = frame.label.to_numpy(dtype=np.int8)
    splits = group_folds(y, groups, outer, 42)
    for fold, (train, _) in enumerate(splits):
        group_folds(y[train], groups[train], inner, 42+100+fold)
    group_folds(y, groups, inner, 2042)
    output.mkdir(parents=True)
    report = {'status': 'running', 'data_origin': 'user_supplied_previously_explored_course_data',
              'evaluation': 'retrospective_nested_group_cv_not_new_independent_test',
              'label_definition': {'0': 'toxic', '1': 'non-toxic'},
              'protocol': {'outer_folds': outer, 'inner_folds': inner, 'seed': 42,
                           'grouping': 'union_of_chirality_free_Murcko_scaffold_and_exact_original_features',
                           'acyclic_fallback': 'exact_chirality_free_connectivity',
                           'primary_model_preselected': 'lightgbm_original',
                           'baseline': 'logistic_original',
                           'no_model_selection_using_outer_results': True,
                           'threshold_objective': 'F1 non-toxic label1; grid .20:.005:.80',
                           'lightgbm_config': CONFIG, 'iteration_cap': cap,
                           'label_selected_Morgan8_features_used': False},
              'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'rdkit', 'joblib']}},
              'data_quality': quality, 'models': []}
    write_json(output/'report.json', report)
    for kind in ['lightgbm', 'logistic']:
        report['models'].append(nested_evaluate(frame, groups, kind, outer=outer, inner=inner, cap=cap))
        write_json(output/'report.json', report)
    # Independently derive FINAL artifact policy from full data inner CV; do not
    # present outer OOF scores as a measurement of this all-data fitted artifact.
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    threshold, iterations = inner_policy(X, y, groups, 'lightgbm', 2042, inner, cap)
    pipe = estimator('lightgbm', 3042, iterations).fit(X, y)
    bundle = {'schema_version': 1, 'pipeline': pipe, 'features': FEATURES,
              'threshold': threshold, 'label_definition': {'0': 'toxic', '1': 'non-toxic'}}
    joblib.dump(bundle, output/'model.joblib')
    loaded = joblib.load(output/'model.joblib')
    difference = float(np.max(np.abs(pipe.predict_proba(X[:32])-loaded['pipeline'].predict_proba(X[:32]))))
    report['final_artifact'] = {'model': 'lightgbm_original', 'threshold': threshold, 'iterations': iterations,
                                'training_rows': len(y), 'roundtrip_max_abs_probability_error': difference,
                                'independently_tested_final_artifact': False}
    model = pipe.named_steps['model']
    names = np.asarray(FEATURES)[pipe.named_steps['variance'].get_support()]
    importance = model.booster_.feature_importance(importance_type='gain')
    top = np.argsort(-importance)[:20]
    report['descriptive_training_gain_not_causal_explanation'] = [{'feature': str(names[i]), 'gain': float(importance[i])} for i in top]
    report['status'] = 'complete'
    write_json(output/'report.json', report)
    # File digests remain local provenance; report contains no row-level values.
    write_json(output/'provenance.json', {'training_sha256': hashlib.sha256(Path(data).read_bytes()).hexdigest()})
    if prediction:
        score(prediction, output/'model.joblib', output/'predictions.csv')
    return report


def score(data, model, output):
    if output.exists():
        raise FileExistsError('Prediction output already exists')
    # joblib can execute code: caller must supply an artifact they created/trust.
    bundle = joblib.load(model)
    if bundle.get('schema_version') != 1 or bundle.get('features') != FEATURES or bundle.get('label_definition') != {'0': 'toxic', '1': 'non-toxic'}:
        raise ValueError('Unsupported model schema or label definition')
    if not np.isfinite(bundle['threshold']) or not 0 <= bundle['threshold'] <= 1:
        raise ValueError('Invalid threshold')
    frame = read_data(data, labeled=False)
    probability = bundle['pipeline'].predict_proba(frame[FEATURES].to_numpy(dtype=np.float64))[:, 1]
    if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid predicted probabilities')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({'SMILES': frame.SMILES, 'output': (probability >= bundle['threshold']).astype(np.int8)})
    with output.open('x') as stream:
        result.to_csv(stream, index=False)
    return {'rows': len(result), 'label_counts': {str(k): int(v) for k, v in result.output.value_counts().items()}}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    auditing = sub.add_parser('audit')
    auditing.add_argument('--data', required=True, type=Path)
    auditing.add_argument('--prediction-data', type=Path)
    auditing.add_argument('--output', required=True, type=Path)
    train = sub.add_parser('train')
    train.add_argument('--data', required=True, type=Path)
    train.add_argument('--prediction-data', type=Path)
    train.add_argument('--output', required=True, type=Path)
    scoring = sub.add_parser('score')
    for name in ['data', 'model', 'output']:
        scoring.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'audit':
        if args.output.exists():
            raise FileExistsError('Audit output already exists')
        frame = read_data(args.data)
        prediction = read_data(args.prediction_data, labeled=False) if args.prediction_data else None
        report, _ = audit(frame, prediction)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    elif args.command == 'train':
        run(args.data, args.output, args.prediction_data)
    else:
        print(json.dumps(score(args.data, args.model, args.output)))


if __name__ == '__main__':
    main()
