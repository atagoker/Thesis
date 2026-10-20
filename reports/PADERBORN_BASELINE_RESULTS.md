# Paderborn source-only baseline: measured results

This is a binary healthy/faulty experiment, not a teacher-student result.

## Recording-level results

Mean +/- sample standard deviation across three training seeds. Same fixed held-out bearings in all runs.

| Condition | Accuracy (%) | Balanced accuracy (%) | Macro F1 (%) | Fault recall (%) | Healthy FPR (%) |
|---|---:|---:|---:|---:|---:|
| N15_M07_F10 | 85.00 +/- 25.98 | 91.00 +/- 15.59 | 84.26 +/- 27.26 | 82.00 +/- 31.18 | 0.00 +/- 0.00 |
| N09_M07_F10 | 51.94 +/- 31.36 | 71.17 +/- 18.82 | 48.19 +/- 30.00 | 42.33 +/- 37.63 | 0.00 +/- 0.00 |
| N15_M01_F10 | 91.67 +/- 13.72 | 95.00 +/- 8.23 | 89.68 +/- 16.62 | 90.00 +/- 16.46 | 0.00 +/- 0.00 |
| N15_M07_F04 | 32.22 +/- 14.68 | 59.33 +/- 8.81 | 31.14 +/- 15.57 | 18.67 +/- 17.62 | 0.00 +/- 0.00 |

## What was actually done

- Attempted numerical audit of 2560 original recordings; 2559 readable/valid; 1 excluded with logged reasons.
- Selected 1240 recordings under the predefined bearing/condition split.
- Trained only on source fit; selected checkpoints only on source validation.
- Source-only training blocks file access to prepared target/evaluator archives with a Python audit hook.
- Froze and hashed all three checkpoints before opening evaluator reference labels.
- Reserved target-adaptation inputs contain x and IDs only and were not used by this baseline.

## How to interpret this

Each condition has 120 evaluation recordings: 20 healthy and 100 faulty, from six bearings. An always-faulty classifier would achieve 83.33% accuracy, 50% balanced accuracy, and 45.45% macro F1. Accuracy alone is therefore inadequate. There is only ONE independent healthy test bearing. Repetitions and windows are correlated; the seed standard deviation is not a population confidence interval.

The source-condition row tests unseen bearings without changing the nominal operating condition. Target rows additionally change operating conditions. This is not a clean estimate of condition shift alone on source training bearings, nor an industrial deployment validation.

ROC AUC is much higher than fixed-threshold performance for several tasks (see results_summary.csv). A high ranking metric does not mean a .5 threshold works: many fault scores fall below that threshold. This suggests a score/decision shift worth investigating, not proof that calibration alone solves cross-bearing generalization. Never choose a new threshold using these target reference labels.

These frozen test results are now observed. Do not tune v1 settings against them; declare a new experiment for any changes. Teacher-student adaptation still needs implementation and matched controls.

## Preparation limitations

The embedded-axis mean-rate diagnostic (assuming seconds) differs from nominal by over 1% in 138 recordings; maximum absolute discrepancy is 14.00%. This is a metadata inconsistency, not a verified alternative acquisition rate.

- Published nominal 64 kHz used despite embedded HostService timing discrepancies; physical frequency calibration unresolved.
- One fixed bearing split; only one healthy test bearing, so false-positive generalization is poorly estimated.
- Target evaluation changes both bearing identity and operating condition relative to source training.
- Windows and repetitions from the same bearing are correlated, not independent experiments.
- Binary healthy/faulty; not a reproduction of the earlier CWRU four-class task.
- Target adaptation pools are mixed-condition benchmarks, not healthy-only commissioning.

## Excluded files

- `data\raw\paderborn\KA08\N15_M01_F10_KA08_2.mat`: TypeError: Expecting matrix here; selected by protocol: False

## Evidence

- Protocol: `reports/PADERBORN_PROTOCOL_V1.md`.
- Exact configuration: `configs/paderborn_baseline_v1.json`.
- Per-record raw checksums and quality: `data/manifests/paderborn_v1_records.csv`.
- Per-seed window and recording metrics: `outputs/paderborn_baseline_v1/results_by_seed.csv`.
- Per-bearing breakdown: `outputs/paderborn_baseline_v1/per_bearing_results.csv`.
- Frozen model evidence: `outputs/paderborn_baseline_v1/training_complete.json`.
- Official data reference: [Lessmeier et al. (2016)](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf).
