"""Saved desktop-session index and independent, portable result exports."""

import csv
from dataclasses import dataclass
from datetime import datetime
import io
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
from zipfile import BadZipFile, ZipFile, ZIP_DEFLATED

from thesisui.branding import APP_NAME, APP_VERSION
from thesisui.data import ROOT, TARGET_NAMES, read_json

SESSION_ROOT = ROOT / "outputs/desktop_sessions"
EXPORT_ROOT = ROOT / "outputs/report_exports"
ARTIFACTS = {"predict": ("prediction.json", "window_predictions.npz"),
             "train": ("completed.json", "run_manifest.json", "history.csv", "teacher.pt"),
             "inspect": ("signal.json",)}


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def valid_result(action, result):
    if action == "predict":
        return (finite_number(result.get("mean_faulty_score"))
                and isinstance(result.get("input_file"), str)
                and type(result.get("window_count")) is int and result["window_count"] > 0)
    if action == "train":
        final = result.get("final")
        return (result.get("status") == "complete" and result.get("target_labels_used") is False
                and result.get("target") in tuple(TARGET_NAMES)
                and isinstance(final, dict) and all(finite_number(final.get(k))
                    for k in ("acceptance", "source_val_teacher_macro_f1")))
    return action == "inspect" and isinstance(result.get("file"), str)


@dataclass(frozen=True)
class Session:
    folder: Path
    action: str
    status: str
    result: dict

    @property
    def teacher(self):
        if self.action == "train" and self.status == "complete":
            return self.folder / "teacher.pt"
        return None

    @property
    def description(self):
        if self.action == "train" and self.result:
            target = self.result.get("target")
            return f"{TARGET_NAMES.get(target, target)} · seed {self.result.get('seed')} · {self.result.get('epochs')} epochs"
        path = self.result.get("input_file") if self.action == "predict" else self.result.get("file")
        if isinstance(path, str):
            return Path(path).name
        return "No completed result"


def read_session(folder):
    """Do not deserialize models or expose outputs from uncommitted jobs."""
    def read_file(name):
        path = folder / name
        if not path.is_file() or not path.resolve().is_relative_to(folder.resolve()):
            return {}
        value = read_json(path)
        return value if isinstance(value, dict) else {}

    try:
        job = read_file("job.json")
    except (OSError, ValueError):
        job = {}
    action = job.get("action")
    if not isinstance(action, str) or action not in ARTIFACTS:
        action = next((key for key in ARTIFACTS if f"_{key}_" in folder.name), "unknown")
    raw_status = job.get("status")
    status = raw_status if raw_status in ("complete", "cancelled", "failed") else "incomplete"
    result = {}
    if status == "complete":
        names = ARTIFACTS.get(action, ())
        try:
            if names and all((folder / name).is_file() and (folder / name).stat().st_size > 0
                             and (folder / name).resolve().is_relative_to(folder.resolve()) for name in names):
                result = read_file(names[0])
        except (OSError, ValueError):
            pass
        if not valid_result(action, result):
            status, result = "incomplete", {}
    return Session(folder, action, status, result)


def list_sessions(root=SESSION_ROOT):
    root = Path(root).resolve()
    if not root.is_dir():
        return []
    return [read_session(folder) for folder in sorted(root.iterdir(), reverse=True)
            if folder.is_dir() and folder.resolve().parent == root and folder.resolve() == folder]


def report_text(session):
    data = session.result
    lines = [f"{APP_NAME} {APP_VERSION} | Saved {session.action} session",
             f"Session: {session.folder.name}", f"Status: {session.status}", ""]
    if session.action == "predict" and data:
        lines += [f"Recording: {data.get('input_file')}", f"Dataset: {data.get('dataset')}",
                  f"Model: {data.get('checkpoint')}", f"Model role: {data.get('model_role')}",
                  f"Model adaptation condition: {data.get('adaptation_target_condition') or 'Not declared'}",
                  f"Device: {data.get('device')}", f"Decision: {data.get('predicted_class')}",
                  f"Windows: {data.get('window_count')}", f"Mean fault-class score: {data['mean_faulty_score']:.4f}",
                  f"Decision rule: {'faulty at score ≥ 0.50' if data.get('dataset') == 'paderborn' else 'highest mean class score'}", "",
                  "Scores are model outputs, not calibrated failure risks.",
                  "A demonstration recording does not establish diagnostic accuracy.", "",
                  "See prediction.json for input/model hashes and window_scores.csv for individual scores."]
    elif session.action == "train" and data:
        metrics = data.get("final", {})
        lines += [session.description, f"Training updates: {data.get('updates')}",
                  "Target reference labels used: no", f"Teacher SHA-256: {data.get('teacher_sha256')}",
                  f"Last-epoch pseudo-label acceptance: {metrics.get('acceptance', 0):.1%}",
                  f"Last-epoch source-validation macro F1: {metrics.get('source_val_teacher_macro_f1', 0):.1%}", "",
                  "Acceptance is not accuracy. Source validation is not target evaluation.",
                  "This interactive run does not replace the saved thesis study.", "",
                  "See history.csv for epoch metrics and run_manifest.json for the configuration.",
                  "The teacher model stays in the original desktop session folder."]
    elif session.action == "inspect" and data:
        lines += [f"Recording: {data.get('file')}", f"Dataset: {data.get('dataset')}",
                  f"Channel: {data.get('channel')}", f"Samples: {data.get('raw_samples')}",
                  f"Nominal input rate: {data.get('original_fs')} Hz", f"Windows: {data.get('windows')}", "",
                  "Signal inspection only. No fault prediction was run.", "See signal.json for signal and preprocessing details."]
    return "\n".join(lines) + "\n"


def export_session(session, destination=EXPORT_ROOT):
    """Export only completed results, never raw datasets or model weights."""
    session = read_session(session.folder)
    if session.status != "complete":
        raise ValueError("Only completed sessions can be exported.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = destination / f"{APP_NAME}_{session.action}_{time}_{uuid4().hex[:8]}.zip"
    with NamedTemporaryFile(dir=destination, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("summary.txt", report_text(session))
            for name in ARTIFACTS[session.action]:
                if Path(name).suffix in {".json", ".csv"}:
                    archive.write(session.folder / name, name)
            if session.action == "predict":
                import numpy as np
                try:
                    with np.load(session.folder / "window_predictions.npz", allow_pickle=False) as saved:
                        values = saved["probabilities"]
                except (BadZipFile, KeyError, TypeError, ValueError) as error:
                    raise ValueError("The saved window prediction file is invalid.") from error
                classes = session.result["classes"]
                if (classes not in [["healthy", "faulty"], ["healthy", "inner_race", "rolling_element", "outer_race"]]
                        or values.shape != (session.result["window_count"], len(classes))
                        or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all()):
                    raise ValueError("Saved window scores do not match the prediction report.")
                output = io.StringIO(newline="")
                writer = csv.writer(output)
                writer.writerow(["window", "fault_class_score", *classes])
                for index, row in enumerate(values, 1):
                    writer.writerow([index, 1 - float(row[0]), *map(float, row)])
                archive.writestr("window_scores.csv", output.getvalue())
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    return out
