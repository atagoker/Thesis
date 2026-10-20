"""Restore published result/model snapshots into ignored runtime outputs.

Never overwrite a differing local file. This restores no raw/processed datasets.
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


def contained(relative, parent):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / parent).resolve()):
        raise ValueError(f"Manifest path escapes expected directory: {relative}")
    return path


def main():
    manifest = json.loads((ROOT / "results/PUBLICATION_MANIFEST.json").read_text(encoding="utf-8"))
    operations = []
    # Validate the whole operation before copying anything.
    for row in manifest["files"]:
        source = contained(row["published_path"], "results")
        destination = contained(row["restore_path"], "outputs")
        if digest(source) != row["sha256"]:
            raise ValueError(f"Published evidence checksum mismatch: {source}")
        if destination.exists() and digest(destination) != row["sha256"]:
            raise RuntimeError(f"Will not overwrite a different local result: {destination}")
        operations.append((source, destination))
    copied = 0
    for source, destination in operations:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != digest(source):
                raise RuntimeError(f"Restored file verification failed: {destination}")
            copied += 1
    print(f"Verified {len(operations)} evidence files; restored {copied}; existing matches preserved.")
    print("Raw datasets and prepared training/evaluation inputs are still separate; see results/README.md.")


if __name__ == "__main__":
    main()
