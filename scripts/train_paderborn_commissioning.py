"""Run the declared Paderborn commissioning study, without target reference labels."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, read_json, sha256, write_json
from thesislib.commissioning import (CONFIG_PATH, PROTOCOL_PATH, IMPLEMENTATION_PATHS, TrainingAccessGuard,
                                    load_archive, train_run, validate_provenance, write_csv)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_run(out, result, cfg, baseline, seed, condition, origin_digest):
    folder = out / f"seed_{seed}" / (condition or "source_control")
    folder.mkdir(parents=True, exist_ok=True)
    roles = {"student": "continued_source", "teacher": "source_ema"} if condition is None else {
        "student": "student_diagnostic", "teacher": "teacher_student"}
    for network, role in roles.items():
        torch.save({"state_dict": result[network], "classes": cfg["classes"],
                    "config": baseline, "commissioning_config": cfg,
                    "role": f"paderborn_binary_{role}", "seed": seed, "target_condition": condition,
                    "source_checkpoint_sha256": origin_digest,
                    "selected_epoch": cfg["adaptation_epochs"], "selection_rule": cfg["checkpoint_selection"],
                    "target_labels_used": False, "updates": result["updates"]}, folder / f"{role}.pt")
    write_csv(folder / "history.csv", result["history"])
    if result["trace"]:
        np.savez_compressed(folder / "pseudo_trace.npz", **result["trace"])
    return {"seed": seed, "condition": condition or "source_control", "updates": result["updates"],
            "source_sequence_sha256": result["source_sequence_sha256"], "seconds": result["seconds"],
            "final_acceptance": result["history"][-1]["acceptance"],
            "source_checkpoint_sha256": origin_digest}


def compare_replay(reference, replay):
    comparisons = []
    for path in sorted(replay.glob("seed_*/*/*.pt")):
        relative = path.relative_to(replay)
        old = torch.load(reference / relative, map_location="cpu", weights_only=True)["state_dict"]
        new = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        exact = old.keys() == new.keys() and all(torch.equal(old[k], new[k]) for k in old)
        comparisons.append({"checkpoint": relative.as_posix(), "exact_tensor_match": exact,
                            "max_absolute_difference": max(float((old[k] - new[k]).abs().max()) for k in old)})
    traces_exact = True
    for path in replay.glob("seed_*/*/pseudo_trace.npz"):
        with np.load(path, allow_pickle=False) as new, np.load(reference / path.relative_to(replay), allow_pickle=False) as old:
            traces_exact &= set(new.files) == set(old.files) and all(np.array_equal(new[k], old[k]) for k in old.files)
    histories_exact = all(sha256(path) == sha256(reference / path.relative_to(replay))
                          for path in replay.glob("seed_*/*/history.csv"))
    evidence = {"checkpoints": comparisons, "pseudo_traces_exact": bool(traces_exact),
                "histories_exact": histories_exact, "all_passed": bool(comparisons) and traces_exact
                and histories_exact and all(row["exact_tensor_match"] for row in comparisons),
                "scope": "Same installed software, GPU, inputs and configuration; not a cross-device guarantee.",
                "target_reference_labels_access_blocked": True}
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-seed", type=int, choices=[7, 17, 27])
    parser.add_argument("--replay-condition", choices=["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"])
    args = parser.parse_args()
    if (args.replay_seed is None) != (args.replay_condition is None):
        parser.error("Replay needs both --replay-seed and --replay-condition.")
    cfg = read_json(CONFIG_PATH)
    baseline, frozen, prepared = validate_provenance(cfg)
    original = ROOT / cfg["output_directory"]
    out = original if args.replay_seed is None else original / f"verification_replay_{args.replay_seed}_{args.replay_condition}"
    implementation = {p: sha256(ROOT / p) for p in IMPLEMENTATION_PATHS}
    if args.replay_seed is not None:
        complete = read_json(original / "training_complete.json")
        if complete["config_sha256"] != sha256(CONFIG_PATH) or complete["implementation_sha256"] != implementation:
            raise ValueError("Configuration/code differs from the run being replayed.")
    if (out / "training_started.json").exists() or (out / "training_complete.json").exists():
        raise RuntimeError("A started/completed run exists. Preserve it; do not overwrite experimental evidence.")
    if not torch.cuda.is_available():
        raise RuntimeError("This recorded experiment requires the verified CUDA GPU.")
    declaration = {"started_utc": utc_now(), "protocol_id": cfg["protocol_id"],
                   "config_sha256": sha256(CONFIG_PATH), "protocol_sha256": sha256(PROTOCOL_PATH),
                   "implementation_sha256": implementation,
                   "prepared_summary_sha256": sha256(ROOT / cfg["prepared_summary"]),
                   "baseline_completion_sha256": sha256(ROOT / cfg["baseline_completion"]),
                   "target_labels_used_for_training_or_selection": False,
                   "baseline_test_results_previously_inspected": True,
                   "experiment_design": "exploratory; fixed adaptation settings before this run",
                   "python": sys.version, "torch": str(torch.__version__),
                   "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "replay": args.replay_seed is not None}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "training_started.json", declaration)
    write_json(out / "config_snapshot.json", cfg)
    guard = TrainingAccessGuard()
    sys.addaudithook(guard)
    guard.enabled = True
    x, y, _ = load_archive(cfg, prepared, "source_fit", labeled=True)
    validation = load_archive(cfg, prepared, "source_validation", labeled=True)
    runs = []
    input_hashes = {name: prepared["prepared_sha256"][name] for name in ["source_fit.npz", "source_validation.npz"]}
    print(f"GPU: {torch.cuda.get_device_name(0)}; source windows={len(y)}; labels hidden for target inputs", flush=True)
    for seed in cfg["seeds"] if args.replay_seed is None else [args.replay_seed]:
        guard.target_condition = None
        source_path = (ROOT / cfg["baseline_completion"]).parent / f"seed_{seed}/source_best.pt"
        checkpoint = torch.load(source_path, map_location="cuda", weights_only=True)
        if checkpoint["classes"] != cfg["classes"] or checkpoint["seed"] != seed:
            raise ValueError("Unexpected source checkpoint metadata.")
        origin_digest = sha256(source_path)
        control = train_run(cfg, seed, checkpoint["state_dict"], (x, y), validation)
        runs.append(save_run(out, control, cfg, baseline, seed, None, origin_digest))
        sequence = control["source_sequence_sha256"]
        for index, condition in enumerate(cfg["target_conditions"]):
            if args.replay_condition is not None and args.replay_condition != condition:
                continue
            guard.target_condition = condition
            name = f"target_adaptation_{condition}_unlabeled"
            target, target_labels, _ = load_archive(cfg, prepared, name, labeled=False)
            assert target_labels is None
            input_hashes[f"{name}.npz"] = prepared["prepared_sha256"][f"{name}.npz"]
            result = train_run(cfg, seed, checkpoint["state_dict"], (x, y), validation, target, index)
            if result["source_sequence_sha256"] != sequence or result["updates"] != control["updates"]:
                raise RuntimeError("Source update sequence/budget is not matched to the control.")
            runs.append(save_run(out, result, cfg, baseline, seed, condition, origin_digest))
    guard.enabled = False
    if sha256(CONFIG_PATH) != declaration["config_sha256"] or sha256(PROTOCOL_PATH) != declaration["protocol_sha256"]:
        raise RuntimeError("Protocol/configuration changed during training.")
    for relative, digest in implementation.items():
        if sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Code changed during training: {relative}")
    artifacts = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.glob("seed_*/*/*")) if p.is_file()}
    write_json(out / "training_complete.json", {**declaration, "completed_utc": utc_now(), "runs": runs,
               "input_sha256": input_hashes, "artifact_sha256": artifacts,
               "checkpoint_sha256": {k: v for k, v in artifacts.items() if k.endswith(".pt")},
               "source_sequences_and_update_counts_matched": True,
               "audit_guard_enabled_during_all_training": True, "target_evaluation_performed": False})
    if args.replay_seed is not None:
        evidence = compare_replay(original, out)
        evidence.update({"seed": args.replay_seed, "condition": args.replay_condition})
        write_json(original / f"reproducibility_seed{args.replay_seed}_{args.replay_condition}.json", evidence)
        if not evidence["all_passed"]:
            raise RuntimeError("Replay did not match. Inspect reproducibility evidence.")
        print("REPLAY PASSED: exact model tensors, histories and pseudo-label traces.", flush=True)
    else:
        print("COMMISSIONING COMPLETE: all models frozen; evaluator labels have not been loaded.", flush=True)


if __name__ == "__main__":
    main()
