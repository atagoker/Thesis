"""Isolated desktop jobs. stdout is newline-delimited real progress events."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesisui.data import ROOT, encode_event, source_checkpoint

SESSION = None


def emit(event_type, **fields):
    line = encode_event(event_type, **fields)
    print(line, flush=True)
    if SESSION is not None:
        with (SESSION / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def session_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to((ROOT / "outputs/desktop_sessions").resolve()):
        raise ValueError("Desktop job output must be inside outputs/desktop_sessions.")
    path.mkdir(parents=True, exist_ok=True)
    if (path / "job.json").exists():
        raise ValueError("A desktop job already exists in this directory.")
    return path


def health():
    emit("stage", message="Checking PyTorch and CUDA…")
    import torch
    emit("health", python=sys.version.split()[0], torch=str(torch.__version__),
         cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
         vram_total_mb=round(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else 0)


def prepare_signal(args, checkpoint=None):
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    from thesislib.common import normalize_windows, read_json
    from thesislib.paderborn import read_vibration, make_windows
    emit("stage", message="Reading the MATLAB vibration recording…")
    if args.dataset == "paderborn":
        cfg = checkpoint["config"] if checkpoint else read_json(ROOT / "configs/paderborn_baseline_v1.json")
        raw, quality = read_vibration(args.file)
        x = make_windows(raw, cfg)
        original_fs, processed_fs, channel = 64000, cfg["processed_fs_hz"], "vibration_1"
    else:
        cfg = checkpoint["config"] if checkpoint else read_json(ROOT / "configs/pilot_v1.json")
        data = loadmat(args.file)
        if args.channel not in data:
            keys = [k for k in data if not k.startswith("__")]
            raise ValueError(f"Channel '{args.channel}' was not found. Available keys: {', '.join(keys)}")
        raw = np.asarray(data[args.channel], dtype=np.float64).reshape(-1)
        if not np.isfinite(raw).all() or raw.size < 8192 or raw.std() < 1e-10:
            raise ValueError("Signal is invalid, constant or too short.")
        original_fs, processed_fs, channel = args.sampling_rate, cfg["processed_sampling_rate_hz"], args.channel
        if original_fs <= 0:
            raise ValueError("A positive original sampling rate is required.")
        ratio = Fraction(processed_fs, original_fs)
        processed = resample_poly(raw, ratio.numerator, ratio.denominator)
        trim, length = cfg["processed_edge_trim_samples"], cfg["window_length"]
        if trim:
            processed = processed[trim:-trim]
        windows = [processed[s:s+length] for s in range(0, len(processed)-length+1, cfg["stride"])]
        if not windows:
            raise ValueError("Recording is too short for the selected model.")
        x = normalize_windows(np.stack(windows))[:, None, :]
        quality = {"samples": len(raw), "rms_recorded_units": float(np.sqrt(np.mean(raw**2))),
                   "nominal_fs_hz": original_fs, "sampling_rate_policy": "explicit rate supplied by the user"}
    # Peak-preserving envelope: each bin supplies its minimum and maximum.
    bins = min(2500, len(raw) // 2)
    width = len(raw) // bins
    chunks = raw[:bins * width].reshape(bins, width)
    mins, maxs = chunks.argmin(1), chunks.argmax(1)
    positions = np.sort(np.stack([np.arange(bins)*width + mins, np.arange(bins)*width + maxs], axis=1), axis=1).reshape(-1)
    spectrum = np.abs(np.fft.rfft(x[0, 0] * np.hanning(x.shape[-1])))
    frequency = np.fft.rfftfreq(x.shape[-1], d=1 / processed_fs)
    payload = {"file": str(args.file.resolve()), "dataset": args.dataset, "channel": channel,
               "raw_samples": len(raw), "windows": len(x), "original_fs": original_fs,
               "processed_fs": processed_fs, "window_length": x.shape[-1], "duration": len(raw) / original_fs,
               "wave_t": (positions / original_fs).tolist(), "wave_y": raw[positions].tolist(),
               "first_window": x[0, 0].tolist(), "fft_hz": frequency.tolist(), "fft_amplitude": spectrum.tolist(),
               "quality": quality}
    emit("signal", **payload)
    return x, payload


def inspect_signal(args):
    from thesislib.common import write_json
    _, payload = prepare_signal(args)
    write_json(SESSION / "signal.json", payload)
    emit("result", kind="inspect", report=str(SESSION / "signal.json"), message="Signal loaded and checked.")


def predict(args):
    emit("stage", message="Loading the selected PyTorch model…")
    import numpy as np
    import torch
    from thesislib.common import BearingCNN, sha256, write_json
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint.get("classes", [])
    if args.dataset == "paderborn":
        roles = {"paderborn_binary_source_only", "paderborn_binary_continued_source", "paderborn_binary_source_ema",
                 "paderborn_binary_teacher_student"}
        if classes != ["healthy", "faulty"] or checkpoint.get("role") not in roles:
            raise ValueError("Choose a supported binary Paderborn checkpoint.")
    elif classes != ["healthy", "inner_race", "rolling_element", "outer_race"]:
        raise ValueError("Choose a four-class CWRU checkpoint.")
    x, signal_info = prepare_signal(args, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BearingCNN(num_classes=len(classes)).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    emit("stage", message=f"Predicting {len(x)} windows on {device}…")
    result = []
    with torch.no_grad():
        for start in range(0, len(x), 4):
            result.append(model(torch.from_numpy(x[start:start+4]).to(device)).softmax(1).cpu().numpy())
            p = np.concatenate(result)
            emit("prediction_progress", completed=len(p), total=len(x), faulty_scores=(1 - p[:, 0]).tolist())
    p = np.concatenate(result)
    mean = p.mean(0)
    decision = int(mean[1] >= .5) if len(classes) == 2 else int(mean.argmax())
    report = {"input_file": str(args.file.resolve()), "input_sha256": sha256(args.file),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "model_role": checkpoint["role"], "classes": classes, "dataset": args.dataset,
              "adaptation_target_condition": checkpoint.get("target_condition"),
              "predicted_class": classes[decision], "mean_probabilities": {c: float(v) for c, v in zip(classes, mean)},
              "mean_faulty_score": float(1 - mean[0]), "window_count": len(x), "device": str(device),
              "input_quality": signal_info["quality"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "note": "Model scores are not calibrated failure risks. A known dataset recording is a demonstration, not a new independent test."}
    write_json(SESSION / "prediction.json", report)
    np.savez_compressed(SESSION / "window_predictions.npz", probabilities=p)
    emit("result", kind="predict", report=str(SESSION / "prediction.json"), **report)


def train(args):
    import numpy as np
    import torch
    from thesislib.common import read_json, sha256, write_json
    from thesislib.commissioning import (CONFIG_PATH, IMPLEMENTATION_PATHS, TrainingAccessGuard, load_archive,
                                        train_run, validate_provenance, write_csv)
    from thesisui.training import observe_training
    emit("stage", message="Verifying frozen source models and prepared inputs…")
    original = read_json(CONFIG_PATH)
    baseline, _, prepared = validate_provenance(original)
    cfg = {**original, "adaptation_epochs": args.epochs}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = source_checkpoint(args.seed)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    write_json(SESSION / "run_manifest.json", {"kind": "interactive desktop commissioning",
               "status": "started", "config": cfg, "source_checkpoint_sha256": sha256(path),
               "implementation_sha256": {p: sha256(ROOT / p) for p in IMPLEMENTATION_PATHS},
               "observer_sha256": sha256(ROOT / "thesisui/training.py"), "seed": args.seed, "target": args.target,
               "torch": str(torch.__version__), "device": str(device), "target_labels_used": False,
               "note": "Separate interactive run. No target evaluation, no replacement of frozen thesis evidence."})
    guard = TrainingAccessGuard()
    sys.addaudithook(guard)
    guard.enabled, guard.target_condition = True, args.target
    x, y, _ = load_archive(original, prepared, "source_fit", labeled=True)
    validation = load_archive(original, prepared, "source_validation", labeled=True)
    target, truth, _ = load_archive(original, prepared, f"target_adaptation_{args.target}_unlabeled", labeled=False)
    assert truth is None
    emit("training_started", source_windows=len(y), target_windows=len(target), epochs=args.epochs,
         steps_per_epoch=(len(y) + cfg["batch_size"] - 1) // cfg["batch_size"], device=str(device), seed=args.seed,
         target=args.target)
    emit("stage", message="Learning from source labels and unlabeled target windows…")
    with observe_training(train_run, emit, lambda: (SESSION / "cancel.flag").exists()):
        result = train_run(cfg, args.seed, checkpoint["state_dict"], (x, y), validation, target,
                           original["target_conditions"].index(args.target), verbose=False)
    guard.enabled = False
    for network, role in [("teacher", "paderborn_binary_teacher_student"), ("student", "paderborn_binary_student_diagnostic")]:
        torch.save({"state_dict": result[network], "classes": cfg["classes"], "config": baseline,
                    "commissioning_config": cfg, "seed": args.seed, "target_condition": args.target,
                    "role": role, "selected_epoch": args.epochs, "desktop_session": True,
                    "target_labels_used": False}, SESSION / f"{network}.pt")
    write_csv(SESSION / "history.csv", result["history"])
    np.savez_compressed(SESSION / "pseudo_trace.npz", **result["trace"])
    final = {"kind": "interactive desktop commissioning", "status": "complete", "seed": args.seed,
             "target": args.target, "epochs": args.epochs, "updates": result["updates"],
             "source_sequence_sha256": result["source_sequence_sha256"], "seconds": result["seconds"],
             "target_labels_used": False, "teacher_sha256": sha256(SESSION / "teacher.pt"),
             "final": result["history"][-1]}
    write_json(SESSION / "completed.json", final)
    emit("result", kind="train", report=str(SESSION / "completed.json"), checkpoint=str(SESSION / "teacher.pt"),
         message="Interactive run completed. Teacher checkpoint and histories saved.", **{k: v for k, v in final.items() if k != "kind"})


def main():
    global SESSION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["health", "inspect", "predict", "train"])
    parser.add_argument("--session", type=Path)
    parser.add_argument("--dataset", choices=["paderborn", "cwru"], default="paderborn")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--channel", default="X138_DE_time")
    parser.add_argument("--sampling-rate", type=int, default=48000)
    parser.add_argument("--seed", type=int, choices=[7, 17, 27], default=7)
    parser.add_argument("--target", choices=["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"], default="N09_M07_F10")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30:
        parser.error("Epochs must be between 1 and 30.")
    if args.action != "health" and args.session is None:
        parser.error("A unique --session directory is required.")
    if args.action in {"inspect", "predict"} and (args.file is None or not args.file.is_file()):
        parser.error("Choose an existing MATLAB file.")
    if args.action == "predict" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("Choose an existing model checkpoint.")
    if args.session is not None:
        SESSION = session_path(args.session)
        (SESSION / "job.json").write_text(json.dumps({"action": args.action, "status": "started"}) + "\n", encoding="utf-8")
    try:
        emit("stage", message="Starting the local Python worker…")
        {"health": health, "inspect": lambda: inspect_signal(args), "predict": lambda: predict(args), "train": lambda: train(args)}[args.action]()
        emit("done")
        if SESSION:
            (SESSION / "job.json").write_text(json.dumps({"action": args.action, "status": "complete"}) + "\n", encoding="utf-8")
    except Exception as error:
        from thesisui.training import TrainingCancelled
        cancelled = isinstance(error, TrainingCancelled)
        emit("cancelled" if cancelled else "error", message=str(error), detail=traceback.format_exc())
        if SESSION:
            (SESSION / "job.json").write_text(json.dumps({"action": args.action, "status": "cancelled" if cancelled else "failed"}) + "\n", encoding="utf-8")
        raise SystemExit(2 if cancelled else 1)


if __name__ == "__main__":
    main()
