"""Small desktop metadata helpers; no raw/evaluator labels are loaded here."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
TARGET_NAMES = {"N09_M07_F10": "Reduced speed · 900 rpm", "N15_M01_F10": "Reduced torque · 0.1 Nm",
                "N15_M07_F04": "Reduced force · 400 N"}
METHOD_NAMES = {"source_only": "Initial source", "continued_source": "Extra source",
                "source_ema": "Source EMA", "teacher_student": "Teacher–student"}


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def evidence_root():
    # Published snapshots work even before the runtime outputs have been restored.
    runtime = ROOT / "outputs/paderborn_commissioning_v1"
    return runtime if (runtime / "evaluation.json").is_file() else ROOT / "results/paderborn_commissioning_v1"


def summary():
    return read_json(evidence_root() / "evaluation.json", {})


def dataset_summary():
    return read_json(ROOT / "data/manifests/paderborn_v1_summary.json", {})


def source_checkpoint(seed):
    relative = Path(f"paderborn_baseline_v1/seed_{seed}/source_best.pt")
    for parent in [ROOT / "outputs", ROOT / "results"]:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Source checkpoint is missing. Restore published results first.")


def checkpoint_choices(seed=7, target=TARGETS[0]):
    options = []
    try:
        options.append(("Initial source · seed " + str(seed), source_checkpoint(seed)))
    except FileNotFoundError:
        pass
    folder = evidence_root() / f"seed_{seed}"
    for method, group in [("continued_source", "source_control"), ("source_ema", "source_control"),
                          ("teacher_student", target)]:
        path = folder / group / f"{method}.pt"
        if path.is_file():
            options.append((f"{METHOD_NAMES[method]} · seed {seed}", path))
    return options


def make_session(kind):
    # Every GUI action writes to a unique directory; existing frozen runs are never destinations.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = ROOT / "outputs/desktop_sessions" / f"{stamp}_{kind}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def encode_event(event_type, **fields):
    return json.dumps({"event": event_type, **fields}, allow_nan=False)


def decode_event(line):
    try:
        item = json.loads(line)
    except (ValueError, TypeError):
        return {"event": "log", "message": str(line)}
    if not isinstance(item, dict) or not isinstance(item.get("event"), str):
        return {"event": "log", "message": str(line)}
    return item
