# Paderborn bearing-disjoint baseline protocol

This protocol is declared before the first baseline test evaluation. It is a
second dataset experiment, not an extension of the CWRU four-class test set.

## Research question

Can a model trained on one operating condition detect faults on previously unseen
bearings under different operating conditions? This stage establishes the
source-only baseline. It does not yet test teacher-student adaptation.

## Labels and provenance

Use two classes: `healthy` (0), `faulty` (1). K001-K006 are healthy. The remaining
26 documented bearings are faulty, including combined inner/outer-ring defects.
No rolling-element-only class is fabricated to match CWRU. The official paper's
tables 4, 5 and 7 supply the mapping; section 4.1 documents nominal 64 kHz vibration
sampling. MATLAB input is selected by `Y.Name == vibration_1`; its one-based
`XIndex` identifies the embedded time axis.

Source: Lessmeier, Kimotho, Zimmer and Sextro (2016),
[official benchmark paper](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf).
Data: [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter).
Dataset use is subject to the publisher's CC BY-NC 4.0 terms and attribution.

## Fixed split

The exact bearing lists are in `configs/paderborn_baseline_v1.json`. Every one of
the 32 bearings belongs to exactly one group:

| Role | Healthy bearings | Faulty bearings | Conditions used |
|---|---:|---:|---|
| Source fit | 3 | 12 | N15_M07_F10 |
| Source validation | 1 | 4 | N15_M07_F10 |
| Unlabeled target adaptation, reserved | 1 | 5 | Each of the three target conditions |
| Evaluation | 1 | 5 | Source condition and each target condition |

The groups are bearing-disjoint. All windows and repetitions of a bearing remain
in its group. The adaptation and evaluation bearings are also disjoint. This is
stricter than randomly dividing windows from one bearing into train/test sets.
Only the predeclared conditions are selected: having downloaded every recording
does not mean every recording should be used for training. Unselected recordings
are retained unchanged and are numerically audited.

Conditions: N15_M07_F10 is 1500 rpm / 0.7 Nm / 1000 N. Targets change speed to
900 rpm (N09_M07_F10), torque to 0.1 Nm (N15_M01_F10), or radial force to 400 N
(N15_M07_F04). This is not a cross-machine industrial deployment test.

## Signal processing

Audit every original MATLAB file for finite, nonconstant vibration values and a
finite, increasing matching time axis; preserve its SHA256 and signal statistics.
Unreadable or invalid recordings are excluded with their exact path and error
recorded, never silently repaired. Halt if over five percent fail. The first
identified unreadable file is `N15_M01_F10_KA08_2.mat`; its bearing/condition is
not selected for any model partition under this frozen split. This quality rule
was added during preparation, before training or any baseline evaluation.
Use the paper's nominal 64 kHz rate, antialias-filter/resample by 1/4 to nominal
16 kHz, trim 128 output samples at each recording edge, form nonoverlapping
2048-sample windows, and z-score each window independently. There is no fitted
target normalization or statistics transfer between recordings.

Important limitation: embedded HostService timestamps and file lengths are not
always exactly consistent with a 64 kHz grid over four seconds. Preparation
reports the discrepancy for every file. The baseline uses the documented nominal
rate without silently substituting timestamp-derived rates. Physical frequency
claims require further acquisition/timing investigation. A software pass does
not resolve that measurement issue.

## Training and evaluation

Use the existing small 1D CNN with a two-output head (the CWRU default remains
four outputs). Train from scratch, not from the CWRU checkpoint. Use seeds 7, 17,
27; Adam, learning rate .001, weight decay .00001; batch size 128; 20 epochs.
Class-balanced sampling uses source training labels only. Select the checkpoint
with greatest source-validation recording-level macro F1; preserve the earliest
tie. Aggregate windows by averaging their softmax probabilities per recording.
Use a fixed .5 faulty-probability threshold. Do not select settings or checkpoints
using target labels. Reserved adaptation archives contain only inputs and IDs;
true target and test labels live in separate evaluator files.

The training program receives only source fit and source validation arrays. The
separate evaluator runs after all checkpoint hashes are frozen. Report recording-
level accuracy, balanced accuracy, macro F1, fault recall, healthy false-positive
rate, ROC AUC and average precision; also preserve window-level metrics and
per-bearing breakdowns. Source-condition tests use the same held-out bearing group
as target-condition tests, which helps describe the added condition shift.

## Interpretation limits

- There are six evaluation bearings but only one is healthy. Hundreds of windows
  do not create hundreds of independent healthy bearings.
- Training-seed standard deviations describe optimization variability on this
  fixed split, not population uncertainty or a statistical superiority claim.
- The task is binary; results cannot be directly compared with four-class CWRU
  macro F1 as though the problems were identical.
- Baseline results cannot demonstrate teacher-student benefit. That requires a
  later frozen adaptation protocol plus continued-source and source-EMA controls.
- No target-label-based tuning after evaluation is permitted under this version.
  Any method changes must be declared as a new experiment.

## Commands

Run from the project folder. Preparation and training refuse to overwrite a
completed version; prediction and evaluation reports can be regenerated.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_paderborn.py
.\.venv\Scripts\python.exe scripts\train_paderborn_baseline.py
.\.venv\Scripts\python.exe scripts\evaluate_paderborn_baseline.py
```
