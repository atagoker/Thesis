"""Desktop event and data-boundary checks without a display server."""

import unittest

import torch

from thesisui.data import checkpoint_choices, decode_event, encode_event
from thesisui.training import TrainingCancelled, observe_training


class DesktopChecks(unittest.TestCase):
    def test_event_protocol_handles_kind_fields_and_non_json_output(self):
        message = decode_event(encode_event("result", kind="predict", predicted_class="healthy"))
        self.assertEqual(message["event"], "result")
        self.assertEqual(message["kind"], "predict")
        self.assertEqual(decode_event("ordinary output")["event"], "log")
        self.assertEqual(decode_event("[]")["event"], "log")

    def test_model_catalog_excludes_diagnostic_students(self):
        paths = [str(path) for _, path in checkpoint_choices(7, "N09_M07_F10")]
        self.assertEqual(len(paths), 4)
        self.assertFalse(any("student_diagnostic" in p for p in paths))
        self.assertTrue(any("teacher_student" in p for p in paths))

    def test_observer_emits_completed_steps_and_epoch_without_changing_result(self):
        def function():
            cfg = {"batch_size": 2, "adaptation_epochs": 2}
            loader = [1, 2]
            history = []
            source_loss, target_loss = torch.tensor(2.), torch.tensor(.1)
            for epoch in [1, 2]:
                totals = {"source_seen": 0}
                totals["source_seen"] += 2
                totals["source_seen"] += 2
                history.append({"epoch": epoch, "source_loss": 2.})
            return 17
        events = []
        with observe_training(function, lambda kind, **payload: events.append((kind, payload))):
            value = function()
        self.assertEqual(value, 17)
        self.assertEqual([event[0] for event in events], ["batch", "batch", "epoch"] * 2)
        self.assertEqual([(p["epoch"], p["step"]) for e, p in events if e == "batch"],
                         [(1, 1), (1, 2), (2, 1), (2, 2)])
        with self.assertRaises(TrainingCancelled):
            with observe_training(function, lambda *args, **kwargs: None, lambda: True):
                function()


if __name__ == "__main__":
    unittest.main()
