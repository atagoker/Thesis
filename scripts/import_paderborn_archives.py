"""Safely unpack the user's Paderborn archives and record import evidence.

No model training, label mapping, or modification of existing CWRU data occurs.
7-Zip checks archive CRCs while extracting. Local SHA256 values identify the
archives; they are NOT checksums supplied by the dataset publisher.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ARCHIVES = set("""K001 K002 K003 K004 K005 K006
KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30
KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21""".split())
SOURCE = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command):
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"7-Zip failed ({result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def entries(seven_zip, archive):
    listing = run([seven_zip, "l", "-slt", "-sccUTF-8", str(archive)])
    if "----------" not in listing:
        raise RuntimeError(f"No file listing in {archive}")
    files = []
    for block in re.split(r"\r?\n\s*\r?\n", listing.split("----------", 1)[1].strip()):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        if "Path" not in fields:
            continue
        path = PureWindowsPath(fields["Path"])
        if (path.is_absolute() or path.drive or ".." in path.parts
                or not path.parts or path.parts[0] != archive.stem
                or any(":" in part for part in path.parts)):
            raise RuntimeError(f"Unsafe/unexpected archive entry: {fields['Path']}")
        if fields.get("Encrypted") == "+" or any("Link" in key for key in fields):
            raise RuntimeError(f"Encrypted/linked entry not supported: {path}")
        if fields.get("Folder") == "-":
            files.append({"relative_path": path.as_posix(), "bytes": int(fields["Size"]),
                          "archive_crc32": fields.get("CRC")})
    if not files:
        raise RuntimeError(f"Empty archive: {archive}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=ROOT / "rar")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/paderborn")
    args = parser.parse_args()
    seven_zip = shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("7-Zip is required; no archives have been changed.")
    archives = sorted(args.archives.resolve().glob("*.rar"))
    if not archives:
        raise RuntimeError("No .rar archives found.")
    actual = {path.stem for path in archives}
    unexpected = actual - OFFICIAL_ARCHIVES
    if unexpected:
        raise RuntimeError(f"Unexpected archive names: {sorted(unexpected)}")
    output = args.output.resolve()
    manifest = ROOT / "data/manifests/paderborn_import.json"
    if manifest.exists():
        raise RuntimeError(f"Import evidence already exists: {manifest}; inspect before rerunning.")
    catalog = [(archive, entries(seven_zip, archive)) for archive in archives]
    needed = sum(entry["bytes"] for _, files in catalog for entry in files)
    if shutil.disk_usage(ROOT).free < needed + 1024 ** 3:
        raise RuntimeError("Insufficient free disk space for extraction plus 1 GiB reserve.")
    for archive, _ in catalog:
        if (output / archive.stem).exists():
            raise RuntimeError(f"Will not overwrite existing folder: {output / archive.stem}")
    output.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "official_catalog_url": SOURCE, "archive_directory": str(args.archives.resolve()),
              "extracted_directory": str(output), "status": "in_progress",
              "missing_official_archives": sorted(OFFICIAL_ARCHIVES - actual),
              "archive_count": len(archives), "expected_extracted_bytes": needed,
              "archive_hash_note": "Locally computed; not publisher-provided checksums.",
              "records": []}
    print(f"Archives: {len(archives)}; compressed: {sum(a.stat().st_size for a in archives)/1e9:.2f} GB; "
          f"extracted: {needed/1e9:.2f} GB", flush=True)
    for index, (archive, files) in enumerate(catalog, 1):
        digest = sha256(archive)
        extraction = run([seven_zip, "x", "-y", "-aos", "-bb0", "-bd", "-sccUTF-8",
                          f"-o{output}", str(archive)])
        if "Everything is Ok" not in extraction:
            raise RuntimeError(f"No successful integrity confirmation for {archive.name}")
        for entry in files:
            path = output / entry["relative_path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"Extracted file missing or wrong size: {path}")
        report["records"].append({"archive": archive.name, "archive_bytes": archive.stat().st_size,
                                   "archive_sha256": digest, "extraction_crc_check": "passed",
                                   "mat_file_count": sum(f["relative_path"].endswith(".mat") for f in files),
                                   "files": files})
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[{index}/{len(catalog)}] {archive.name}: extracted and CRC checked; "
              f"{report['records'][-1]['mat_file_count']} MATLAB recordings", flush=True)
    report["status"] = "complete"
    report["total_mat_files"] = sum(record["mat_file_count"] for record in report["records"])
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"IMPORT COMPLETE: {report['total_mat_files']} MATLAB recordings. Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
