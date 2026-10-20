"""Package small reproducibility evidence; never include raw data or environments."""

import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, sha256, write_json


def main():
    files = set()
    for directory in ["thesislib"]:
        files.update((ROOT / directory).glob("*.py"))
    files.add(ROOT / "tests/test_paderborn.py")
    for name in ["README.md", "START_HERE.md", "requirements.txt", "requirements-torch.txt", "requirements-lock.txt",
                 "configs/paderborn_baseline_v1.json", "reports/PADERBORN_PROTOCOL_V1.md",
                 "reports/PADERBORN_BASELINE_RESULTS.md", "reports/PADERBORN_START_HERE.md", "reports/THESIS_PROGRESS.md",
                 "data/manifests/paderborn_import.json", "data/manifests/paderborn_v1_summary.json",
                 "data/manifests/paderborn_v1_records.csv"]:
        files.add(ROOT / name)
    for name in ["import_paderborn_archives.py", "inspect_paderborn_import.py", "prepare_paderborn.py",
                 "train_paderborn_baseline.py", "evaluate_paderborn_baseline.py", "predict_paderborn.py",
                 "archive_paderborn_evidence.py"]:
        files.add(ROOT / "scripts" / name)
    for directory in ["outputs/paderborn_baseline_v1", "outputs/paderborn_demo"]:
        files.update(path for path in (ROOT / directory).rglob("*") if path.is_file()
                     and not any(part.startswith("verification_replay_") for part in path.parts))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    index = {str(path.relative_to(ROOT)).replace("\\", "/"):
             {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)}
    out = ROOT / "outputs/evidence"
    out.mkdir(exist_ok=True, parents=True)
    archive = out / "paderborn_baseline_v1_evidence.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(files):
            zipped.write(path, str(path.relative_to(ROOT)))
        zipped.writestr("EVIDENCE_INDEX.json", json.dumps(index, indent=2))
    with zipfile.ZipFile(archive) as zipped:
        bad = zipped.testzip()
        if bad:
            raise RuntimeError(f"Archive integrity failure: {bad}")
    write_json(out / "paderborn_baseline_v1_index.json", {
        "archive": str(archive), "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
        "included_files": index, "excludes": ["raw vibration data", "RAR archives", "Python environment", "duplicate replay checkpoints"],
        "note": "No raw/processed arrays are included. Recreate them from the original archives before running data-dependent tests."})
    print(f"Evidence archive: {archive}; bytes={archive.stat().st_size}; files={len(files)}; CRC check passed")


if __name__ == "__main__":
    main()
