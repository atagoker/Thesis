"""Archive code, manifests, and measured evidence without raw CWRU data."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, sha256, write_json


def main():
    selected = set()
    for directory in ["scripts", "thesislib", "tests", "configs", "reports", "data/manifests", ".vscode"]:
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                selected.add(path)
    for name in ["README.md", "START_HERE.md", "requirements.txt", "requirements-torch.txt", "requirements-lock.txt", ".gitignore"]:
        selected.add(ROOT / name)
    for path in (ROOT / "outputs/pilot_v1").rglob("*"):
        if path.is_file() and "verification_replay" not in path.parts:
            selected.add(path)
    for name in ["LICENSE", "models/CNN_1.py"]:
        selected.add(ROOT / "external/CAT" / name)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Code, settings, provenance, checkpoints, predictions, figures, and reports; no raw CWRU data or Python environment",
        "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(selected)],
    }
    evidence = ROOT / "outputs/evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    index = evidence / "pilot_v1_index.json"
    write_json(index, manifest)
    archive_path = evidence / "pilot_v1_evidence.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
        archive.write(index, arcname="evidence_index.json")
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Archive verification failed: {bad}")
    print(f"Evidence archive: {archive_path} ({archive_path.stat().st_size/1e6:.2f} MB)", flush=True)
    print(f"SHA256: {sha256(archive_path)}", flush=True)


if __name__ == "__main__":
    main()
