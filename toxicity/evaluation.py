"""Threshold selection and explicit toxicity-aware evaluation."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold


def select_threshold(y, probability):
    thresholds = np.round(np.arange(.20, .801, .005), 3)
    scores = np.asarray([f1_score(y, probability >= t, pos_label=1, zero_division=0) for t in thresholds])
    # Ties: closest to 0.5, then smaller threshold. Frozen policy, not outer-score tuning.
    return float(min(thresholds[scores == scores.max()], key=lambda t: (abs(t-.5), t)))


def metrics(y, probability, predicted):
    return {
        'accuracy': float(accuracy_score(y, predicted)),
        'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
        'f1_non_toxic_label1': float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        'precision_non_toxic_label1': float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        'recall_non_toxic_label1': float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        'f1_toxic_label0': float(f1_score(y, predicted, pos_label=0, zero_division=0)),
        'precision_toxic_label0': float(precision_score(y, predicted, pos_label=0, zero_division=0)),
        'recall_toxic_label0': float(recall_score(y, predicted, pos_label=0, zero_division=0)),
        'roc_auc_non_toxic_label1': float(roc_auc_score(y, probability)),
        'average_precision_toxic_label0': float(average_precision_score(1-y, 1-probability)),
        'confusion_matrix_rows_actual_columns_predicted_0_1': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        'toxic_missed_as_non_toxic': int(((y == 0) & (predicted == 1)).sum()),
    }


def group_folds(y, groups, count, seed):
    splits = list(StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    coverage = np.zeros(len(y), dtype=int)
    for train, valid in splits:
        if set(groups[train]) & set(groups[valid]):
            raise ValueError('Group overlap')
        if set(y[train]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError('Both labels are required in every fold; use more data or fewer folds')
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError('Invalid OOF coverage')
    return splits


def error_slice(y, predicted, mask):
    toxic = mask & (y == 0)
    n = int(toxic.sum())
    return {'rows': int(mask.sum()), 'toxic_rows': n,
            'toxic_missed_as_non_toxic': int((toxic & (predicted == 1)).sum()),
            'recall_toxic_label0': float((predicted[toxic] == 0).mean()) if n else None}
