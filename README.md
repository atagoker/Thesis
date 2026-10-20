# Bearing fault diagnosis thesis

Working Python/PyTorch prototype for **Implementation of Label-Free Teacher–Student Commissioning for Vibration-Signal-Based Fault Detection**.

Repository: [atagoker/Thesis](https://github.com/atagoker/Thesis), default branch
`master`. This is a BSc research prototype, not a production diagnostic system.

## Browse or reproduce the work

- [Published results, figures and checkpoints](results/README.md) are included in
  this repository. Large original datasets and the Python environment are not.
- [Thesis progress](reports/THESIS_PROGRESS.md) distinguishes completed work from
  remaining research; percentages of the entire thesis are not asserted.
- [Detailed English research companion](docs/research/THESIS_RESEARCH_COMPANION_EN_2026-09-06.md)
  and [master guide](THESIS_MASTER_GUIDE.md) explain the background and plan.
- [Initial student-supplied presentation](docs/presentations/initial_thesis_presentation.pptx)
  is retained unchanged as historical context, not a presentation of current results.

For a fresh checkout:

```powershell
git clone https://github.com/atagoker/Thesis.git
cd Thesis
```

Install dependencies using [START_HERE.md](START_HERE.md), then restore the small
published runtime artifacts with:

```powershell
.\.venv\Scripts\python.exe scripts\restore_published_results.py
```

Download/prepare the original datasets separately before running data-dependent
tests or evaluation. Frozen experiments deliberately refuse to overwrite completed
runs. To conduct new training, declare a new experiment version or use a separate
fresh working copy without restored completion markers.

Public availability does not grant a blanket source/data license. No project-wide
source-code license has been selected. Dataset and referenced-code rights remain
with their respective publishers; see [data and evidence notes](results/README.md).

## New: Paderborn baseline

The downloaded 32-bearing collection is now connected to a **binary healthy/faulty,
bearing-disjoint source-only baseline**. All original recordings were audited;
one unreadable MATLAB file outside the selected partitions is documented.
Three GPU training seeds and a separate frozen-checkpoint evaluation are complete.
Paderborn teacher-student adaptation is **not yet implemented**.

Start with [the beginner run guide](reports/PADERBORN_START_HERE.md),
[measured results](reports/PADERBORN_BASELINE_RESULTS.md), and
[the fixed protocol](reports/PADERBORN_PROTOCOL_V1.md). The CWRU experiment below
is separate and remains unchanged in scope.

The project now includes real CWRU data preparation, source training, custom EMA teacher–student commissioning, separate evaluation, and a command-line predictor. Read [the measured pilot report](reports/PILOT_RESULTS.md) first, then [the implemented protocol](reports/EXPERIMENT_PROTOCOL_PILOT_V1.md).

## Current scientific result

The pilot evaluates four classes, three training seeds, and source load 0 to target loads 1, 2, and 3 hp. Teacher–student improves mean macro F1 over the initial source-only model, but does not outperform matched extra-source and source-only EMA controls. This is an initial finding, not evidence of general adaptation superiority.

The split uses disjoint temporal blocks within recordings. It is not independent-bearing validation. The normal-baseline nominal sample rate is an explicitly recorded assumption. Full published CAT training remains unfinished.

## Open the completed work in VS Code

```powershell
code .
```

Select `.venv\Scripts\python.exe` as the interpreter. See [START_HERE.md](START_HERE.md) for environment details. All commands below run from this project folder and use the explicit interpreter so activation is optional.

## View or regenerate the reports

```powershell
.\.venv\Scripts\python.exe scripts\make_pilot_report.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Figures are in `outputs/pilot_v1/figures/`. Original data and model artifacts are intentionally ignored by Git. The small provenance and window manifests are in `data/manifests/`.

## Reproduce the pipeline from a fresh project copy

Install the environment using `START_HERE.md`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\download_cwru.py
.\.venv\Scripts\python.exe scripts\prepare_cwru.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\run_pilot.py
.\.venv\Scripts\python.exe scripts\run_source_ema_control.py
.\.venv\Scripts\python.exe scripts\evaluate_pilot.py
.\.venv\Scripts\python.exe scripts\analyze_pseudolabels.py
.\.venv\Scripts\python.exe scripts\verify_pilot_replay.py
.\.venv\Scripts\python.exe scripts\make_pilot_report.py
```

`run_pilot.py` refuses to overwrite a completed pilot. On the current working copy, use the stored outputs and the replay verifier. For a new full experiment, preserve this run and create a new protocol/output version rather than changing these evaluated settings silently.

## Run a prototype prediction

The following is a demonstration on a known collection member, not a new independent test:

```powershell
.\.venv\Scripts\python.exe scripts\predict_record.py --file data\raw\cwru48\138.mat --channel X138_DE_time --sampling-rate 48000 --checkpoint outputs\pilot_v1\seed_7\teacher_target3.pt
```

It writes `outputs/demo/prediction.json`. For a different measurement, provide its actual channel and sampling rate. Model confidence is not a validated industrial failure probability.

## Public code reference

The reference repository is [EPFL-IMOS/CAT](https://github.com/EPFL-IMOS/CAT), pinned in the compatibility report to commit `354cf94aef4f0afd1f81d7ef645bd40ffc68d8d5`. It was cloned into `external/CAT`, which is Git-ignored and retains the original MIT license.

On a fresh copy, obtain and check it with:

```powershell
git clone https://github.com/EPFL-IMOS/CAT.git external/CAT
git -C external/CAT switch --detach 354cf94aef4f0afd1f81d7ef645bd40ffc68d8d5
.\.venv\Scripts\python.exe external\CAT\train_base.py --help
.\.venv\Scripts\python.exe scripts\check_cat_reference.py
```

The custom experiment uses its own model and training loop. The public CNN check only establishes forward/backward/optimizer compatibility on real CWRU inputs.

## Project map

```text
configs/pilot_v1.json       declared pilot settings
thesislib/common.py         model and shared numerical utilities
scripts/                   preparation, training, inference, evaluation
tests/test_pilot.py         checks for scientific failure modes
data/manifests/             provenance, assumptions, splits, counts
data/raw/                  downloaded immutable data (ignored)
data/processed/             generated input/evaluator archives (ignored)
outputs/pilot_v1/           checkpoints, predictions, metrics, figures (ignored)
reports/                   readable method, results, learning notes
external/CAT/              original reference code (ignored)
```

The next step is understanding the implemented pipeline, then selecting a stronger evaluation or method extension with the supervisor. Slides are deferred until requested.
