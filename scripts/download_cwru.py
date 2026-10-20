"""Download the official CWRU 48 kHz DE subset and preserve provenance.

Run: .venv/Scripts/python.exe scripts/download_cwru.py
No raw data are included in Git. URLs are discovered from official tables.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, CLASSES, read_json, sha256, write_json

FAULT_PAGE = "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data"
NORMAL_PAGE = "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.label = ""
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = ""

    def handle_data(self, data):
        if self.href:
            self.label += data

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            self.links.append((self.href, self.label.strip()))
            self.href = None


def official_catalog():
    result = []
    for page in [NORMAL_PAGE, FAULT_PAGE]:
        request = Request(page, headers={"User-Agent": "BearingThesis/1.0"})
        with urlopen(request, timeout=45) as response:
            html = response.read().decode("utf-8")
        parser = Links()
        parser.feed(html)
        for href, label in parser.links:
            healthy = re.fullmatch(r"Normal_([0-3])", label)
            fault = re.fullmatch(r"(IR|B|OR)(007|014|021)(@6)?_([0-3])", label)
            if page == NORMAL_PAGE and healthy:
                cls, load, size = 0, int(healthy[1]), None
            elif page == FAULT_PAGE and fault:
                if fault[1] == "OR" and fault[3] != "@6":
                    continue
                cls = {"IR": 1, "B": 2, "OR": 3}[fault[1]]
                load, size = int(fault[4]), int(fault[2]) / 1000
            else:
                continue
            url = urljoin(page, href)
            if urlparse(url).netloc != "engineering.case.edu" or not url.endswith(".mat"):
                raise ValueError(f"Unexpected download URL: {url}")
            filename = Path(urlparse(url).path).name
            result.append({
                "record_id": f"cwru48_{filename[:-4]}", "name": label,
                "filename": filename, "class_id": cls, "class_name": CLASSES[cls],
                "load_hp": load, "defect_size_inch": size,
                "outer_race_clock": 6 if cls == 3 else None,
                "nominal_fs_hz": 48000,
                "sampling_rate_evidence": "official 48k fault table" if cls else
                    "48k normal-baseline convention; normal table and MAT lack explicit sample-rate field",
                "sampling_rate_status": "documented_family" if cls else "assumed_nominal",
                "source_page": page, "download_url": url,
            })
    result.sort(key=lambda r: (r["load_hp"], r["class_id"], r["defect_size_inch"] or 0))
    if len(result) != 40 or len({r["filename"] for r in result}) != 40:
        raise RuntimeError(f"Expected 40 unique records, found {len(result)}; inspect official tables.")
    return result


def validate_record(path, record):
    data = loadmat(path)
    keys = [key for key in data if key.endswith("_DE_time")]
    expected = f"X{int(Path(record['filename']).stem):03d}_DE_time"
    flags = []
    if expected in keys:
        selected = expected
    elif len(keys) == 1:
        selected = keys[0]
        flags.append(f"filename/channel-prefix mismatch: selected sole key {selected}")
    else:
        raise ValueError(f"Expected explicitly named channel {expected} in {path.name}, got {keys}")
    if len(keys) > 1:
        flags.append("additional DE arrays in MAT; selected filename-matching key")
    x = np.asarray(data[selected], dtype=np.float64).reshape(-1)
    if x.size < 20000 or not np.isfinite(x).all() or x.std() <= 0:
        raise ValueError(f"Invalid signal: {path}")
    rpm_key = selected.replace("_DE_time", "RPM")
    rpm = float(np.asarray(data[rpm_key]).reshape(-1)[0]) if rpm_key in data else None
    return dict(record, channel_key=selected, all_de_channel_keys=keys, samples=int(x.size),
                quality_flags=flags,
                rpm_recorded=rpm,
                bytes=path.stat().st_size, sha256=sha256(path),
                raw_path=path.relative_to(ROOT).as_posix())


def download_one(record, existing):
    path = ROOT / "data/raw/cwru48" / record["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verified = validate_record(path, record)
        if record["record_id"] in existing and verified["sha256"] != existing[record["record_id"]]["sha256"]:
            raise RuntimeError(f"Existing raw file differs from saved checksum: {path}")
        return verified
    temporary = path.with_suffix(".mat.part")
    for attempt in range(3):
        try:
            request = Request(record["download_url"], headers={"User-Agent": "BearingThesis/1.0"})
            with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            validate_record(temporary, record)
            temporary.replace(path)
            return validate_record(path, record)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = ROOT / "data/manifests/cwru48_records.json"
    existing = {r["record_id"]: r for r in read_json(manifest)["records"]} if manifest.exists() else {}
    records = official_catalog()
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(download_one, record, existing) for record in records]
        for job in as_completed(jobs):
            item = job.result()
            completed.append(item)
            print(f"[{len(completed):02d}/40] {item['name']}: {item['samples']:,} DE samples, {item['bytes']/1e6:.1f} MB", flush=True)
    completed.sort(key=lambda r: (r["load_hp"], r["class_id"], r["defect_size_inch"] or 0))
    write_json(manifest, {
        "created_utc": datetime.now(timezone.utc).isoformat(), "classes": CLASSES,
        "raw_file_count": len(completed), "records": completed,
        "limitations": ["Normal-file 48 kHz is a recorded assumption, not embedded MAT metadata.",
                        "Source and target loads may reuse physical bearings.",
                        "Clock position is preserved as @6 without interpreting the load-zone convention."],
    })
    print(f"Saved provenance manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
