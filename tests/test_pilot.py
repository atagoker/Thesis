"""Checks for scientific failure modes, not merely code coverage."""

import csv
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, BearingCNN, ema_update, masked_pseudo_loss, normalize_windows


class MethodChecks(unittest.TestCase):
    def test_only_accepted_pseudo_labels_supply_gradients(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        loss = masked_pseudo_loss(logits, torch.tensor([1, 2]), torch.tensor([True, False]))
        loss.backward()
        self.assertLess(float(logits.grad[0, 1]), 0)
        self.assertGreater(float(logits.grad[0, 0]), 0)
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros(4)))

    def test_empty_confidence_mask_is_finite_zero_with_gradients(self):
        logits = torch.randn(5, 4, requires_grad=True)
        loss = masked_pseudo_loss(logits, torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.bool))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_teacher_is_independent_and_follows_ema(self):
        student, teacher = BearingCNN(), BearingCNN()
        with torch.no_grad():
            for p in student.parameters():
                p.fill_(4)
            for p in teacher.parameters():
                p.fill_(2)
        ema_update(teacher, student, 0.9)
        for t, s in zip(teacher.parameters(), student.parameters()):
            self.assertTrue(torch.allclose(t, torch.full_like(t, 2.2)))
            self.assertNotEqual(t.data_ptr(), s.data_ptr())

    def test_constant_signal_normalization_is_finite(self):
        actual = normalize_windows(np.ones((3, 1024)))
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(float(np.abs(actual).max()), 0)

    def test_raw_intervals_and_filter_regions_do_not_cross_splits(self):
        path = ROOT / "data/manifests/pilot_v1_windows.csv"
        self.assertTrue(path.exists(), "Run prepare_cwru.py before the data-integrity checks.")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            lo, hi = int(row["raw_start"]), int(row["raw_end_exclusive"])
            self.assertGreaterEqual(lo, int(row["raw_region_start"]))
            self.assertLessEqual(hi, int(row["raw_region_end_exclusive"]))
            groups.setdefault(row["record_id"], []).append(row)
        for record_rows in groups.values():
            intervals = sorted(record_rows, key=lambda r: int(r["raw_start"]))
            for a, b in zip(intervals, intervals[1:]):
                self.assertLessEqual(int(a["raw_end_exclusive"]), int(b["raw_start"]))
                if a["split"] != b["split"]:
                    self.assertLess(int(a["raw_region_end_exclusive"]), int(b["raw_region_start"]))

    def test_target_training_archives_have_no_labels(self):
        for load in [1, 2, 3]:
            for split in ["fit", "validation", "test"]:
                with np.load(ROOT / f"data/processed/pilot_v1/load{load}_{split}_unlabeled.npz") as data:
                    self.assertEqual(set(data.files), {"x", "ids"})
                    self.assertTrue(np.isfinite(data["x"]).all())


if __name__ == "__main__":
    unittest.main()
