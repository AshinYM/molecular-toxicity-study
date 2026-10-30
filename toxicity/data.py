"""Strict input schema and label-free molecular grouping; no row values in reports."""
from collections import Counter
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

FEATURES = [f'{prefix}_{i}' for prefix in ('ecfp', 'fcfp', 'ptfp') for i in range(1024)] + ['MolWt', 'clogp', 'sa_score', 'qed']
FINGERPRINTS = FEATURES[:-4]


def read_data(path, *, labeled=True):
    # Detect duplicate CSV headers before pandas silently renames them.
    import csv
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError('Duplicate input columns')
    frame = pd.read_csv(path)
    expected = {'SMILES', *FEATURES} | ({'label'} if labeled else set())
    if set(frame.columns) != expected or len(frame) == 0:
        raise ValueError('Input must have SMILES, the documented features, and label only for training')
    if frame.SMILES.isna().any() or not frame.SMILES.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise ValueError('SMILES must be non-empty strings')
    try:
        values = frame[FEATURES].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError('Features must be numeric') from exc
    if not np.isfinite(values).all():
        raise ValueError('Missing or non-finite feature values')
    if not np.isin(values[:, :3072], [0, 1]).all():
        raise ValueError('Fingerprint values must be binary')
    if labeled and (frame.label.isna().any() or set(frame.label.unique()) != {0, 1}):
        raise ValueError('Training requires both labels: 0=toxic, 1=non-toxic')
    # Check SMILES without printing proprietary molecular strings on parse failure.
    with rdBase.BlockLogs():
        valid = [Chem.MolFromSmiles(s) is not None for s in frame.SMILES]
    if not all(valid):
        raise ValueError(f'Invalid SMILES rows: {len(valid)-sum(valid)}')
    return frame


def molecule_keys(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    stereo = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    Chem.RemoveStereochemistry(mol)
    connectivity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return stereo, connectivity, scaffold or 'ACYCLIC:'+connectivity


def feature_digests(frame):
    values = np.ascontiguousarray(frame[FEATURES].to_numpy(dtype='<f8'))
    values[values == 0] = 0  # Normalize signed zero for exact feature identity.
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in values]


def grouped_identity(frame):
    """Union scaffolds AND identical model inputs, including transitive relations.

    Acyclic fallback only groups exact connectivity, not all acyclic similarity.
    Does not merge salts/tautomers or claim chemical equivalence beyond RDKit keys.
    """
    keys = [molecule_keys(s) for s in frame.SMILES]
    digests = feature_digests(frame)
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, (key, digest) in enumerate(zip(keys, digests)):
        for token in [('scaffold', key[2]), ('features', digest)]:
            if token in seen:
                a, b = find(i), find(seen[token])
                parent[a] = b
            else:
                seen[token] = i
    groups = np.asarray([find(i) for i in range(len(frame))])
    return groups, keys, digests


def collision_summary(keys, labels):
    grouped = {}
    for key, label in zip(keys, labels):
        grouped.setdefault(key, []).append(int(label))
    repeated = [v for v in grouped.values() if len(v) > 1]
    conflicting = [v for v in grouped.values() if len(set(v)) > 1]
    return {'duplicate_groups': len(repeated), 'duplicate_rows': sum(map(len, repeated)),
            'conflicting_label_groups': len(conflicting), 'conflicting_label_rows': sum(map(len, conflicting))}


def audit(frame, prediction=None):
    groups, keys, digests = grouped_identity(frame)
    values = frame[FEATURES].to_numpy(dtype=float)
    result = {
        'rows': len(frame), 'features': len(FEATURES),
        'label_counts': {str(k): int(v) for k, v in Counter(frame.label).items()},
        'missing_cells': int(frame.isna().sum().sum()),
        'constant_features_descriptive_only': int((np.ptp(values, axis=0) == 0).sum()),
        'duplicate_smiles_extra': int(frame.SMILES.duplicated().sum()),
        'canonical_stereo_identity': collision_summary([k[0] for k in keys], frame.label),
        'connectivity_identity': collision_summary([k[1] for k in keys], frame.label),
        'exact_model_inputs': collision_summary(digests, frame.label),
        'scaffold_groups': len(set(k[2] for k in keys)),
        'combined_groups': len(set(groups)),
        'largest_combined_group_rows': max(Counter(groups).values()),
        'acyclic_rows': sum(k[2].startswith('ACYCLIC:') for k in keys),
        'descriptor_iqr_outliers_not_removed': {},
    }
    for name in FEATURES[-4:]:
        q1, q3 = frame[name].quantile([.25, .75])
        result['descriptor_iqr_outliers_not_removed'][name] = int(((frame[name] < q1-1.5*(q3-q1)) | (frame[name] > q3+1.5*(q3-q1))).sum())
    if prediction is not None:
        pred_keys = [molecule_keys(s) for s in prediction.SMILES]
        pred_hashes = feature_digests(prediction)
        result['prediction_overlap'] = {
            'rows': len(prediction),
            'canonical_stereo_rows': sum(k[0] in {x[0] for x in keys} for k in pred_keys),
            'connectivity_rows': sum(k[1] in {x[1] for x in keys} for k in pred_keys),
            'scaffold_rows': sum(k[2] in {x[2] for x in keys} for k in pred_keys),
            'exact_model_input_rows': sum(h in set(digests) for h in pred_hashes),
            'prediction_labels_available': False,
        }
    return result, groups
