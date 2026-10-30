import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from toxicity.data import FEATURES, audit, grouped_identity, molecule_keys, read_data
from toxicity.evaluation import group_folds, metrics, select_threshold
from toxicity.train import estimator, inner_policy, nested_evaluate, score


def fixture():
    values = np.zeros((8, len(FEATURES)))
    values[:, :8] = np.eye(8)
    values[:, -4:] = np.arange(32).reshape(8, 4)+1
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, 'SMILES', ['CCO', 'OCC', 'C[C@H](O)F', 'C[C@@H](O)F',
                              'c1ccccc1', 'Cc1ccccc1', 'CCN', 'CCC'])
    frame['label'] = [0, 0, 0, 1, 1, 0, 1, 1]
    return frame


class DataTests(unittest.TestCase):
    def test_schema_missing_extra_nonfinite_binary_labels_and_smiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.csv'
            original = fixture()
            original.to_csv(path, index=False)
            self.assertEqual(len(read_data(path)), 8)
            cases = [original.drop(columns=['qed']), original.assign(unknown=1),
                     original.assign(qed=np.inf), original.assign(ecfp_0=2),
                     original.assign(label=1), original.assign(SMILES='invalid')]
            for frame in cases:
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    read_data(path)
            original.drop(columns='label').to_csv(path, index=False)
            self.assertEqual(len(read_data(path, labeled=False)), 8)
            original.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                read_data(path, labeled=False)

    def test_duplicate_header_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.csv'
            path.write_text('SMILES,SMILES\nCCO,CCO\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_data(path)

    def test_chirality_connectivity_and_scaffold_keys(self):
        a, b = molecule_keys('C[C@H](O)F'), molecule_keys('C[C@@H](O)F')
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(a[1:], b[1:])
        self.assertNotEqual(molecule_keys('CCN')[2], molecule_keys('CCO')[2])
        self.assertEqual(molecule_keys('c1ccccc1')[2], molecule_keys('Cc1ccccc1')[2])

    def test_transitive_union_blocks_same_input_and_scaffold_overlap(self):
        frame = fixture()
        # Acyclic CCO and aromatic benzene have identical supplied features;
        # benzene and toluene share a scaffold, so all must stay in one group.
        frame.loc[4, FEATURES] = frame.loc[0, FEATURES].to_numpy()
        groups, _, _ = grouped_identity(frame)
        self.assertEqual(len(set(groups[[0, 1, 4, 5]])), 1)
        self.assertEqual(groups[2], groups[3])
        changed = frame.assign(label=1-frame.label)
        np.testing.assert_array_equal(groups, grouped_identity(changed)[0])

    def test_conflicts_are_reported_not_dropped_and_no_molecule_values_leak(self):
        frame = fixture()
        report, groups = audit(frame, frame.drop(columns='label'))
        self.assertEqual(report['rows'], len(frame))
        self.assertEqual(report['connectivity_identity']['conflicting_label_groups'], 1)
        self.assertEqual(report['prediction_overlap']['canonical_stereo_rows'], 8)
        payload = json.dumps(report)
        self.assertNotIn('C[C@H]', payload)
        self.assertNotIn('SMILES', payload)


class EvaluationTests(unittest.TestCase):
    def test_label_zero_is_toxic_and_confusion_orientation(self):
        y = np.array([0, 0, 1, 1])
        result = metrics(y, np.array([.1, .9, .8, .7]), np.array([0, 1, 1, 1]))
        self.assertEqual(result['recall_toxic_label0'], .5)
        self.assertEqual(result['toxic_missed_as_non_toxic'], 1)
        self.assertEqual(result['confusion_matrix_rows_actual_columns_predicted_0_1'], [[1, 1], [0, 2]])

    def test_threshold_tie_policy_and_no_mutation(self):
        y, p = np.array([0, 1]), np.array([.1, .9])
        self.assertEqual(select_threshold(y, p), .5)
        np.testing.assert_array_equal(p, [.1, .9])

    def test_group_disjointness_and_single_oof_coverage(self):
        y = np.tile([0, 1], 30)
        groups = np.repeat(np.arange(30), 2)
        coverage = np.zeros(len(y))
        for train, valid in group_folds(y, groups, 5, 42):
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            coverage[valid] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(y)))

    def test_variance_and_scaler_fit_on_training_only(self):
        X = np.array([[0., 1.], [1., 1.], [0., 1.], [1., 1.]])
        pipe = estimator('logistic', 42).fit(X, [0, 1, 0, 1])
        np.testing.assert_array_equal(pipe.named_steps['variance'].get_support(), [True, False])
        # A new value in training-constant column must not add a feature.
        self.assertEqual(pipe[:-1].transform([[100., 999.]]).shape, (1, 1))
        self.assertEqual(pipe.named_steps['scale'].max_abs_[0], 1.)

    def test_outer_validation_never_enters_inner_selection_or_refit(self):
        frame = fixture()
        groups = np.arange(len(frame))
        partitions = [(np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
                      (np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3]))]
        class Fake:
            def fit(self, X, y):
                self.trained = set(X[:, -1])
                return self
            def predict_proba(self, X):
                assert not self.trained & set(X[:, -1])
                return np.tile([.4, .6], (len(X), 1))
        selected = []
        def policy(X, y, g, *args):
            selected.append(set(g))
            return .5, 5
        with patch('toxicity.train.group_folds', return_value=partitions), \
             patch('toxicity.train.inner_policy', side_effect=policy), \
             patch('toxicity.train.estimator', side_effect=lambda *a: Fake()):
            nested_evaluate(frame, groups, 'lightgbm', outer=2)
        self.assertEqual(selected, [{0, 1, 2, 3}, {4, 5, 6, 7}])

    def test_real_inner_lightgbm_policy_and_scoring_roundtrip(self):
        import joblib
        rng = np.random.default_rng(42)
        X = rng.normal(size=(90, 5))
        y = np.tile([0, 1], 45)
        threshold, count = inner_policy(X, y, np.repeat(np.arange(45), 2), 'lightgbm', 42, 3, 5)
        self.assertTrue(.2 <= threshold <= .8)
        self.assertTrue(1 <= count <= 5)
        frame = fixture()
        pipe = estimator('logistic', 42).fit(frame[FEATURES].to_numpy(), frame.label)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame.drop(columns='label').to_csv(root/'input.csv', index=False)
            bundle = {'schema_version': 1, 'features': FEATURES, 'pipeline': pipe,
                      'threshold': .5, 'label_definition': {'0': 'toxic', '1': 'non-toxic'}}
            joblib.dump(bundle, root/'model.joblib')
            self.assertEqual(score(root/'input.csv', root/'model.joblib', root/'result.csv')['rows'], 8)
            output = pd.read_csv(root/'result.csv')
            self.assertEqual(output.columns.tolist(), ['SMILES', 'output'])
            self.assertEqual(output.SMILES.tolist(), frame.SMILES.tolist())
            with self.assertRaises(FileExistsError):
                score(root/'input.csv', root/'model.joblib', root/'result.csv')
            bundle['label_definition'] = {'0': 'non-toxic', '1': 'toxic'}
            joblib.dump(bundle, root/'model.joblib')
            with self.assertRaises(ValueError):
                score(root/'input.csv', root/'model.joblib', root/'other.csv')


if __name__ == '__main__':
    unittest.main()
