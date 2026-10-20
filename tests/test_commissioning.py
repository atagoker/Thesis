"""Scientific safeguards for the new commissioning extension (synthetic inputs)."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from thesislib.common import BearingCNN, ROOT, ema_update, masked_pseudo_loss, read_json, set_seed
from thesislib.commissioning import (CONFIG_PATH, TrainingAccessGuard, load_archive, make_teacher,
                                    source_loader, train_run, views)
from thesislib.commissioning_evaluation import metrics
from scripts.analyze_paderborn_commissioning import describe_trace


class CommissioningChecks(unittest.TestCase):
    def setUp(self):
        self.cfg = read_json(CONFIG_PATH)
        set_seed(3)

    def test_guard_allows_only_selected_unlabeled_target_and_source(self):
        guard = TrainingAccessGuard()
        guard.enabled = True
        allowed = ROOT / "data/processed/paderborn_v1/source_fit.npz"
        guard("open", (str(allowed), "rb", 0))
        target = ROOT / "data/processed/paderborn_v1/target_adaptation_N09_M07_F10_unlabeled.npz"
        with self.assertRaises(PermissionError):
            guard("open", (str(target), "rb", 0))
        guard.target_condition = "N09_M07_F10"
        guard("open", (str(target), "rb", 0))
        for name in ["data/processed/paderborn_v1/evaluator/target_adaptation_N09_M07_F10_labels.npz",
                     "data/processed/paderborn_v1/evaluation_N09_M07_F10_unlabeled.npz",
                     "data/processed/paderborn_v1/target_adaptation_N15_M01_F10_unlabeled.npz",
                     "data/manifests/paderborn_v1_records.csv", "reports/test.md",
                     "results/paderborn_baseline_v1/predictions/example.npz",
                     "data/raw/paderborn/K001/example.mat"]:
            with self.assertRaises(PermissionError, msg=name):
                guard("open", (str(ROOT / name), "rb", 0))

    def test_target_schema_rejects_labels_even_when_hash_is_valid(self):
        from thesislib.common import sha256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.npz"
            np.savez(path, x=np.zeros((2, 1, 2048)), ids=["a", "b"], y=[0, 1])
            cfg = {**self.cfg, "prepared_directory": directory}
            prepared = {"prepared_sha256": {"target.npz": sha256(path)}}
            with self.assertRaisesRegex(ValueError, "schema"):
                load_archive(cfg, prepared, "target", labeled=False)

    def test_source_order_does_not_depend_on_global_or_augmentation_rng(self):
        x, y = torch.randn(13, 1, 2048), np.array([0] * 3 + [1] * 10)
        a = source_loader(x, y, 4, 7)
        first = torch.cat([row[2] for row in a])
        torch.randn(100)
        views(x, self.cfg, torch.Generator().manual_seed(55))
        b = source_loader(x, y, 4, 7)
        self.assertTrue(torch.equal(first, torch.cat([row[2] for row in b])))

    def test_teacher_is_independent_frozen_and_ema_follows_updated_student(self):
        student = BearingCNN(num_classes=2)
        teacher = make_teacher(student)
        initial = copy.deepcopy(teacher.state_dict())
        with torch.no_grad():
            for parameter in student.parameters():
                parameter.add_(2.)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(all(torch.equal(initial[k], v) for k, v in teacher.state_dict().items()))
        ema_update(teacher, student, .75)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, initial[key] + .5)

    def test_masked_loss_full_denominator_and_empty_mask_gradients(self):
        logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        labels = torch.tensor([0, 0])
        loss = masked_pseudo_loss(logits, labels, torch.tensor([True, False]))
        expected = torch.nn.functional.cross_entropy(logits[:1], labels[:1]) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.)
        logits.grad = None
        zero = masked_pseudo_loss(logits, labels, torch.zeros(2, dtype=torch.bool))
        zero.backward()
        self.assertEqual(float(zero.detach()), 0.)
        self.assertEqual(float(logits.grad.abs().sum()), 0.)

    def test_zero_target_weight_reproduces_both_source_control_networks(self):
        cfg = {**self.cfg, "batch_size": 4, "adaptation_epochs": 2, "target_loss_weight": 0.}
        x, y = torch.randn(9, 1, 2048), np.array([0] * 4 + [1] * 5)
        ids = np.asarray([f"r{i}:0" for i in range(9)])
        state = BearingCNN(num_classes=2).state_dict()
        control = train_run(cfg, 7, state, (x, y), (x, y, ids), verbose=False)
        adapted = train_run(cfg, 7, state, (x, y), (x, y, ids), torch.randn(5, 1, 2048), verbose=False)
        self.assertEqual(control["source_sequence_sha256"], adapted["source_sequence_sha256"])
        self.assertEqual(control["updates"], 6)
        for network in ["student", "teacher"]:
            for key in control[network]:
                self.assertTrue(torch.equal(control[network][key], adapted[network][key]), key)
        self.assertEqual(set(adapted["trace"]), {"epoch", "index", "pseudo", "confidence", "accepted", "student_prediction"})
        self.assertLess(int(adapted["trace"]["index"].max()), 5)

    def test_imbalanced_all_faulty_classifier_cannot_pass_balanced_metrics(self):
        y = np.array([0] * 20 + [1] * 100)
        result = metrics(y, np.tile([.1, .9], (120, 1)))
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)
        self.assertAlmostEqual(result["macro_f1"], 5 / 11)
        self.assertEqual(result["healthy_false_positive_rate"], 1.)
        self.assertEqual(result["fault_recall"], 1.)

    def test_posthoc_trace_counts_repeated_exposures_and_empty_acceptance(self):
        trace = {"epoch": np.array([1, 1, 1, 2]), "index": np.array([0, 1, 1, 0]),
                 "pseudo": np.array([0, 0, 1, 0]), "accepted": np.array([True, True, True, False]),
                 "confidence": np.array([.95, .96, .97, .6])}
        row = describe_trace(trace, np.array([0, 1]), 1)
        self.assertEqual(row["exposures"], 3)
        self.assertEqual(row["true_faulty_exposures"], 2)
        self.assertAlmostEqual(row["accepted_precision"], 2 / 3)
        self.assertAlmostEqual(row["accepted_healthy_precision"], .5)
        self.assertEqual(row["accepted_wrong"], 1)
        empty = describe_trace(trace, np.array([0, 1]), 2)
        self.assertIsNone(empty["accepted_precision"])
        self.assertIsNone(empty["true_faulty_coverage"])


if __name__ == "__main__":
    unittest.main()
