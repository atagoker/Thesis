"""Paderborn protocol, input schema, and leakage checks."""

import copy
import csv
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, BearingCNN, read_json
from thesislib.paderborn import bearing_metadata, make_windows, read_vibration, validate_protocol
from thesislib.paderborn_ml import aggregate_recordings


class PaderbornChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(ROOT / "configs/paderborn_baseline_v1.json")

    def test_all_bearings_partitioned_without_overlap(self):
        validate_protocol(self.cfg)
        self.assertEqual(sum(map(len, self.cfg["bearing_groups"].values())), 32)

    def test_overlap_is_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["bearing_groups"]["evaluation"][0] = "K001"
        with self.assertRaises(ValueError):
            validate_protocol(cfg)

    def test_combined_faults_are_not_mislabeled_healthy(self):
        for bearing in ["KB23", "KB24", "KB27"]:
            self.assertEqual(bearing_metadata(bearing)["class_id"], 1)
            self.assertEqual(bearing_metadata(bearing)["damage_family"], "combined_inner_outer")

    def test_binary_head_preserves_four_class_default(self):
        x = torch.zeros(2, 1, 2048)
        self.assertEqual(BearingCNN()(x).shape, (2, 4))
        self.assertEqual(BearingCNN(num_classes=2)(x).shape, (2, 2))

    def test_recording_aggregation_does_not_depend_on_window_order(self):
        p = np.array([[.1, .9], [.8, .2], [.3, .7]])
        records, means, labels = aggregate_recordings(p, ["a:0", "b:0", "a:2048"], np.array([1, 0, 1]))
        np.testing.assert_array_equal(records, ["a", "b"])
        np.testing.assert_allclose(means, [[.2, .8], [.8, .2]])
        np.testing.assert_array_equal(labels, [1, 0])
        with self.assertRaises(ValueError):
            aggregate_recordings(p, ["a:0", "b:0", "a:2048"], np.array([1, 0, 0]))

    def test_real_signal_and_nonoverlapping_windows(self):
        path = ROOT / "data/raw/paderborn/K001/N15_M07_F10_K001_1.mat"
        raw, quality = read_vibration(path)
        x = make_windows(raw, self.cfg)
        self.assertGreater(quality["samples"], 200000)
        self.assertEqual(x.shape[1:], (1, 2048))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(x.mean(axis=-1), 0, atol=1e-6)
        np.testing.assert_allclose(x.std(axis=-1), 1, atol=1e-6)

    def test_target_and_test_archives_do_not_contain_labels(self):
        paths = list((ROOT / "data/processed/paderborn_v1").glob("*_unlabeled.npz"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"x", "ids"})

    def test_recordings_never_cross_bearing_groups(self):
        with (ROOT / "data/manifests/paderborn_v1_records.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2560)
        seen = {}
        for row in rows:
            seen.setdefault(row["bearing_id"], set()).add(row["bearing_group"])
        self.assertTrue(all(len(roles) == 1 for roles in seen.values()))
        for row in rows:
            self.assertIn(row["bearing_id"], self.cfg["bearing_groups"][row["bearing_group"]])

    def test_training_guard_denies_target_and_evaluator_reads(self):
        from scripts.train_paderborn_baseline import deny_target_access
        for path in ["data/processed/paderborn_v1/evaluator/a.npz",
                     "data/processed/paderborn_v1/target_adaptation_N09_M07_F10_unlabeled.npz",
                     "data/processed/paderborn_v1/evaluation_N09_M07_F10_unlabeled.npz",
                     "data/manifests/paderborn_v1_records.csv"]:
            with self.assertRaises(PermissionError):
                deny_target_access("open", (path, "r", 0))
        deny_target_access("open", ("data/processed/paderborn_v1/source_fit.npz", "r", 0))


if __name__ == "__main__":
    unittest.main()
