"""Persist regression tests and cross-check new reports against frozen evidence."""

import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, read_json, sha256, write_json
from thesislib.commissioning_evaluation import verify_frozen_run


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    cfg, _, _, out, frozen = verify_frozen_run()
    baseline = csv_rows((ROOT / cfg["baseline_completion"]).parent / "results_by_seed.csv")
    old = {(r["condition"], r["seed"], r["level"]): r for r in baseline}
    rows = csv_rows(out / "results_by_seed.csv")
    target = [r for r in rows if r["scope"] == "target" and r["level"] == "recording"]
    if len(target) != 36 or len(rows) != 144:
        raise AssertionError("Expected 36 primary method/task/seed results and 144 total level/scope rows.")
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "fault_recall",
                    "healthy_false_positive_rate", "roc_auc", "fault_average_precision", "tn", "fp", "fn", "tp"]
    checked = 0
    for r in rows:
        if r["method"] == "source_only":
            previous = old[(r["evaluation_condition"], r["seed"], r["level"])]
            for key in metric_names:
                if abs(float(r[key]) - float(previous[key])) > 1e-12:
                    raise AssertionError(f"Initial baseline changed: {r['evaluation_condition']} {r['seed']} {key}")
            checked += 1
    replay = read_json(out / "reproducibility_seed7_N09_M07_F10.json")
    if not replay["all_passed"]:
        raise AssertionError("Exact replay failed.")
    analysis = read_json(out / "analysis.json")
    if not analysis["all_trace_histories_reconciled"] or len(csv_rows(out / "pseudo_labels_by_epoch.csv")) != 90:
        raise AssertionError("Pseudo-label trace reconciliation incomplete.")
    # Exercise a real intercepted open in a fresh process, without ever reading the file.
    program = ("import sys; from thesislib.common import ROOT; "
               "from thesislib.commissioning import TrainingAccessGuard; "
               "g=TrainingAccessGuard(); g.enabled=True; g.target_condition='N09_M07_F10'; "
               "sys.addaudithook(g); "
               "open(ROOT/'data/processed/paderborn_v1/evaluator/target_adaptation_N09_M07_F10_labels.npz','rb')")
    guarded = subprocess.run([sys.executable, "-c", program], cwd=ROOT, capture_output=True, text=True, check=False)
    guard_passed = guarded.returncode != 0 and "Commissioning trainer cannot read" in guarded.stderr
    if not guard_passed:
        raise AssertionError("Actual evaluator file open was not intercepted.")
    buffer = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    print(buffer.getvalue())
    evidence = {"verified_utc": datetime.now(timezone.utc).isoformat(),
                "verification_script_sha256": sha256(Path(__file__)),
                "training_artifact_hashes_verified": True, "original_baseline_metric_rows_matched": checked,
                "primary_method_task_seed_rows": len(target), "total_result_rows": len(rows),
                "source_update_budget_per_run": sorted({r["updates"] for r in frozen["runs"]}),
                "all_source_sequences_matched": True, "exact_replay_passed": replay["all_passed"],
                "real_evaluator_open_blocked": guard_passed, "pseudo_trace_epochs_reconciled": 90,
                "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                "tests_passed": result.wasSuccessful(), "test_output": buffer.getvalue(),
                "test_file_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "tests").glob("test*.py"))}}
    write_json(out / "verification.json", evidence)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(f"VERIFIED: {checked} unchanged baseline rows; 36 primary comparisons; 90 epoch traces; {result.testsRun} tests.")


if __name__ == "__main__":
    main()
