# Bearing fault diagnosis thesis

Working Python/PyTorch prototype for **Implementation of Label-Free Teacher–Student Commissioning for Vibration-Signal-Based Fault Detection**.

Repository: [atagoker/Thesis](https://github.com/atagoker/Thesis), default branch
`master`. This is a BSc research prototype, not a production diagnostic system.

## AtaScope — native desktop UI

Double-click **[Launch_AtaScope.bat](Launch_AtaScope.bat)** to open AtaScope, a
native Windows/Qt interface with a dark background. It shows real signal plots,
GPU predictions, live batch/epoch progress, teacher-student diagnostics and saved
experiment comparisons. It runs locally as a desktop window; no browser or web
server is used.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\.venv\Scripts\python.exe scripts\desktop_app.py
```

See [the desktop guide](reports/DESKTOP_UI.md) for controls, cancellation, session
outputs and end-to-end verification. Interactive runs save to separate
`outputs/desktop_sessions/` folders and preserve the frozen research evidence.

## Browse or reproduce the work

- [Published results, figures and checkpoints](results/README.md) are included in
  this repository. Large original datasets and the Python environment are not.
- [Thesis progress](reports/THESIS_PROGRESS.md) distinguishes completed work from
  remaining research; percentages of the entire thesis are not asserted.
- [English thesis draft](thesis/THESIS_DRAFT.md) contains the introduction,
  literature, data protocol, methods, implementation, measured results and discussion.
- [New commissioning results](reports/PADERBORN_COMMISSIONING_RESULTS.md) compare
  all four methods; [Turkish milestone guide](reports/COMMISSIONING_MILESTONE_TR.md)
  explains what was completed and how to demonstrate it.
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

## New: Paderborn teacher-student commissioning

The downloaded 32-bearing collection is now connected to a **binary healthy/faulty,
bearing-disjoint source-only baseline**. All original recordings were audited;
one unreadable MATLAB file outside the selected partitions is documented.
The baseline and a separate teacher-student commissioning study are complete.
The new study compares initial source-only, continued-source, source-only EMA and
target-adapted EMA teachers over three seeds and three target conditions. All
source update sequences are matched; target reference labels are accessed only
by the separate evaluator after model freezing. Exact local replay and 23 tests
pass. The experiment is exploratory because baseline test results were already
known before this extension was designed.

Mean target recording macro F1 is **56.34% initial source, 65.42% continued
source, 66.84% source EMA, 66.38% teacher-student**. Unlabeled-target learning
does not demonstrate an average benefit beyond source EMA in this protocol.
Under reduced radial force, accepted pseudo-label precision is only 20.29–28.15%
in the final epoch despite high confidence. Read the
[new protocol](reports/PADERBORN_COMMISSIONING_PROTOCOL_V1.md) and
[measured report](reports/PADERBORN_COMMISSIONING_RESULTS.md) for every run and limitation.

After restoring published artifacts and preparing the original data:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_paderborn_commissioning.py
.\.venv\Scripts\python.exe scripts\analyze_paderborn_commissioning.py
.\.venv\Scripts\python.exe scripts\verify_commissioning_evidence.py
```

Start with [the beginner run guide](reports/PADERBORN_START_HERE.md),
[measured results](reports/PADERBORN_BASELINE_RESULTS.md), and
[the fixed protocol](reports/PADERBORN_PROTOCOL_V1.md). The CWRU experiment below
is separate and remains unchanged in scope.

The project now includes real CWRU data preparation, source training, custom EMA teacher–student commissioning, separate evaluation, and a command-line predictor. Read [the measured pilot report](reports/PILOT_RESULTS.md) first, then [the implemented protocol](reports/EXPERIMENT_PROTOCOL_PILOT_V1.md).

## Earlier CWRU scientific result

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

Figures are in each experiment's `outputs/.../figures/` and published under
`results/`. Raw data and runtime outputs are ignored by Git; selected small
checkpoint/report snapshots are included under `results/`. Provenance and window
manifests are in `data/manifests/`.

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

The next step is reviewing the English thesis draft and measured failure cases,
then agreeing a stronger confirmation/ablation study with the supervisor. Slides
remain deferred until requested.
