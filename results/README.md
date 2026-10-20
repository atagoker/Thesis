# Published experimental evidence

This folder makes the completed work inspectable directly on GitHub, without
requiring the multi-gigabyte raw datasets or a Python environment.

- `pilot_v1/`: four-class CWRU source-only, continued-source, custom EMA
  teacher-student and source-only EMA control results, histories, predictions,
  figures, model checkpoints, and frozen execution evidence.
- `paderborn_baseline_v1/`: binary healthy/faulty source-only models for three
  seeds, bearing-disjoint evaluation, per-bearing results, figures and replay
  evidence. Paderborn teacher-student adaptation is not implemented yet.
- `paderborn_demo/`: healthy/faulty command-line demonstration reports.
- `paderborn/`: initial archive-import inspection.
- `setup/`: synthetic software/GPU check; not a fault-detection result.
- `demo/`: CWRU inference demonstration and compatibility regression output.

`PUBLICATION_MANIFEST.json` identifies every snapshot by SHA256 and original
runtime path. These are byte-identical copies; the original experiment records
have not been rewritten to look more portable. Some JSON metadata intentionally
retain original Windows paths. `.gitattributes` disables line-ending conversion
so the frozen hashes survive cloning.

## Restore checkpoint/report paths after cloning

After installing the environment described in `START_HERE.md`, run:

```powershell
.\.venv\Scripts\python.exe scripts\restore_published_results.py
```

The script verifies hashes, restores files to `outputs/`, and refuses to replace
a differing local file. Restoring published results does **not** recreate raw
datasets or prepared signal arrays. Inference still needs the input MATLAB file;
full evaluation needs the prepared arrays and separate evaluator labels.

## Data access and exclusions

Original CWRU/Paderborn signals, RAR archives, prepared signal arrays, the virtual
environment and vendored third-party source are intentionally excluded. The
Paderborn extracted data alone occupy about 22.37 GB. Obtain data from the
publishers and follow their terms:

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file).
- [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter).
  The publisher specifies CC BY-NC 4.0 and attribution; do not assume commercial
  permission from this repository's public visibility.

The small provenance, split and quality manifests are under `data/manifests/`.
Published prediction archives may contain reference labels for completed offline
evaluation. They are not the training-facing unlabeled adaptation archives.

## Scientific status

This is research/prototype evidence, not a certified diagnostic product. Read
the reports for negative results, controls, seed variability, limited healthy
bearing counts, sampling-rate caveats and the unreadable Paderborn file. No claim
of teacher-student superiority or industrial deployment readiness is made.
