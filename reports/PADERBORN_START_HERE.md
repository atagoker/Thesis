# Start here: run and understand your Paderborn baseline

## What works now

You have a real-data, GPU-trained diagnostic prototype. It can read a Paderborn
MATLAB recording, extract its vibration channel, preprocess the samples, run a
neural network, and save a binary prediction report. Three training seeds have
been evaluated on bearings excluded from training, validation and adaptation.
The baseline was trained only on source data. Paderborn teacher-student adaptation
is the next research stage, not something already achieved by these results.

You do not need to send the data again. Your original RAR archives remain in
`rar/`; extracted originals are in `data/raw/paderborn/`. The CWRU experiment is
separate. Do not mix its four-class checkpoints with the two-class Paderborn model.

## First action: make a prediction

Open the Thesis folder in VS Code. Choose Terminal > New Terminal. Run:

```powershell
cd "C:\Users\TUF Gaming\Documents\Thesis"
.\.venv\Scripts\python.exe scripts\predict_paderborn.py --file data\raw\paderborn\K006\N15_M07_F10_K006_1.mat
```

This uses the saved seed-7 source-only model. It does not train anything. The
input is a healthy held-out bearing recording from the existing evaluation set.
This is an operating demonstration, not a fresh independent performance test.

The terminal prints a class, a faulty model score, a window count, and a report
path. Read the report with:

```powershell
Get-Content .\outputs\paderborn_demo\prediction.json
```

For a faulty recording at the source operating condition, use a separate report:

```powershell
.\.venv\Scripts\python.exe scripts\predict_paderborn.py --file data\raw\paderborn\KA22\N15_M07_F10_KA22_1.mat --output outputs\paderborn_demo\faulty_example.json
```

You can then change only the input filename to another Paderborn MATLAB file.
Keep the exact original filename. A corrupt or incompatible file should produce
an error, not a fabricated prediction. This loader is not a general CSV importer.

## Read the output correctly

- `predicted_class`: the recording-level healthy/faulty decision.
- `mean_probabilities`: mean neural-network softmax scores across the windows.
  They sum approximately to one but are NOT validated industrial failure risks.
- `window_prediction_counts`: how many windows fall on each side of the fixed
  decision threshold. A recording decision uses mean scores, not majority voting.
- `window_count`: the number of complete nonoverlapping 2048-sample windows.
- `input_quality`: signal statistics and the documented nominal sampling policy.
- `checkpoint`: the saved model used; another seed may produce different scores.
- `warning`: scope and interpretation limits.

The decision rule is faulty if mean faulty score is at least .5; otherwise healthy.
For example, a known faulty signal scoring .4 is a missed fault under this rule.
The fact that software returns a score does not make the decision correct.
An unknown type of machine is not automatically covered by these two known classes.

## What a recording and a window mean

A recording is one original measurement file, approximately four seconds long.
A window is a short section cut from that file for the neural network. We use
the official nominal 64 kHz acquisition rate, antialias-resample to nominal
16 kHz, trim filter edges, then cut nonoverlapping 2048-sample windows (nominal
0.128 seconds each). Each window is normalized using its own mean and standard
deviation. This does not require a fault label.

The model sees a tensor shaped `batch x 1 x 2048`. It outputs two scores per
window. Averaging them gives one recording-level decision. Twenty recordings
from one bearing are repeated measurements, not twenty independent bearings.

Important: embedded timestamps do not always match an exact 64 kHz grid.
The nominal rate is documented by the official paper, while the timestamp
discrepancies are retained in the quality manifest. This must be investigated
before making precise physical-frequency claims. We have not silently "fixed"
the original recordings.

## What was trained

- Source condition: N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N).
- Training bearings: 3 healthy and 12 faulty.
- Source validation bearings: 1 healthy and 4 faulty, all different from training.
- Reserved target adaptation bearings: 1 healthy and 5 faulty, never used by this baseline.
- Evaluation bearings: 1 healthy and 5 faulty, different from all the groups above.
- Targets change speed, load torque or radial force.
- Three seeds, 20 training epochs each; checkpoint selection uses source validation only.

All 32 bearings have a group, but not every operating-condition recording is used
in every group. This deliberately prevents leakage. No target labels are used
for training or model selection. The evaluator knows the reference labels only
to measure performance after all models have been frozen.

## Which results matter

Open `reports/PADERBORN_BASELINE_RESULTS.md` and use the Markdown preview in VS
Code (Ctrl+Shift+V). The primary results are recording-level, not window-level.

- Accuracy: proportion of correct recording decisions. Misleading alone when
  there are many more faults than healthy examples.
- Balanced accuracy: average of healthy recall and faulty recall; an always-one-
  class classifier has 50% balanced accuracy when both classes are present.
- Macro F1: average F1 of the two classes, giving each class equal importance.
- Fault recall: fraction of faulty recordings correctly detected. Low recall
  means missed faults even if healthy false-positive rate is zero.
- Healthy false-positive rate: fraction of healthy recordings incorrectly flagged.
  Here it represents only ONE independent healthy evaluation bearing.
- ROC AUC: ranking quality across thresholds. High AUC does not guarantee the
  fixed .5 decision threshold works. Do not optimize that threshold on test labels.

Seed variation can be large. Report all three seeds rather than choosing the
best-looking one. It describes training variability, not a population confidence
interval. Do not directly compare binary Paderborn macro F1 with four-class CWRU
macro F1 as though they were the same task.

## Commands for checking, not overwriting

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\evaluate_paderborn_baseline.py
```

The first command runs tests; the second regenerates evaluation outputs from
frozen checkpoints. Training and preparation refuse to overwrite a completed
experiment. This is intentional. A new experiment needs a new declared protocol
and output version, not deletion of the current evidence.

A deterministic seed-7 replay has a separate command and folder:

```powershell
.\.venv\Scripts\python.exe scripts\train_paderborn_baseline.py --replay-seed 7
```

Once completed, the replay folder is also protected against overwriting. Read
`outputs/paderborn_baseline_v1/reproducibility_seed_7.json` for the recorded check.
It concerns this installed environment, not a guarantee across every GPU/version.

## What to send when asking for help

Paste the exact command and complete terminal output. For a prediction, give the
JSON report path. For an error, include the full traceback. Do not paste raw
vibration arrays or send your `.venv` directory. Screenshots help with the VS Code
interface; text is better for code errors. I can read local files directly.

## Next research stage

Predeclare an EMA teacher-student protocol before adaptation runs. Use the
reserved unlabeled pools, weak/strong signal perturbations, confidence-gated
pseudo-labels, source supervision and fixed stopping rules. Include matched
continued-source and source-only EMA controls. Do not assume adaptation will
help; confidently wrong pseudo-labels can reinforce errors.

Because baseline test results are now known, subsequent methods are an exploratory
development stage on this benchmark. Stronger confirmation needs additional
predeclared bearing splits or genuinely untouched external data. We should first
understand the missed faults and validation weaknesses, using source validation
and unlabeled diagnostics without tuning directly against target test labels.
