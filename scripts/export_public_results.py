"""Publish small experiment evidence without raw data, archives, or environments.

Copies byte-for-byte snapshots. Does not change original experiment records.
"""

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    selected = []
    for name in ["pilot_v1", "paderborn_baseline_v1", "paderborn_commissioning_v1", "paderborn_demo", "paderborn"]:
        for path in (ROOT / "outputs" / name).rglob("*"):
            if (path.is_file() and path.suffix in {".json", ".csv", ".png", ".pt", ".npz"}
                    and not any(part.startswith("verification_replay") for part in path.parts)):
                selected.append(path)
    for name in ["setup/setup_check.json", "setup/synthetic_signal.png", "demo/prediction.json",
                 "demo/cwru_regression_after_paderborn.json"]:
        path = ROOT / "outputs" / name
        if path.exists():
            selected.append(path)
    output = ROOT / "results"
    entries = []
    for source in sorted(set(selected)):
        if source.stat().st_size > 10 * 1024 * 1024:
            raise ValueError(f"Unexpected large public evidence file: {source}")
        relative = source.relative_to(ROOT / "outputs")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_digest = digest(source)
        if destination.exists() and digest(destination) != source_digest:
            raise RuntimeError(f"Published evidence already differs; review before replacing: {destination}")
        shutil.copy2(source, destination)
        if digest(destination) != source_digest:
            raise RuntimeError(f"Snapshot checksum failure: {destination}")
        entries.append({"published_path": destination.relative_to(ROOT).as_posix(),
                        "restore_path": source.relative_to(ROOT).as_posix(),
                        "bytes": source.stat().st_size, "sha256": source_digest})
    manifest = {"scope": "Byte-identical small experimental evidence; no raw signals or environments",
                "files": entries, "total_bytes": sum(row["bytes"] for row in entries),
                "notes": ["Training/evaluation metadata retain original local paths for provenance.",
                          "Prediction files contain reference labels for offline evaluation, not adaptation inputs.",
                          "Checkpoints are trusted project-generated PyTorch files; load with weights_only=True."]}
    (output / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(entries)} verified files ({manifest['total_bytes']/1e6:.2f} MB) to {output}")


if __name__ == "__main__":
    main()
