# Measured CWRU pilot results

Date: 19 October 2026. Status: preliminary pilot.

This report comes from saved prediction and metric files. It does not claim final thesis completion or industrial validation.

## Main finding

Across the nine task/seed pairs, teacher–student changed macro F1 by **+1.41 percentage points** relative to source-only, **-1.14 points** relative to extra source training, and **-0.05 points** relative to the source-only EMA control.

The pilot therefore does not demonstrate an additional benefit from unlabeled target learning. Control runs reproduce the apparent gains without target inputs. The source-only EMA control matches teacher–student test macro F1 in eight of nine task/seed pairs, with a small difference in the remaining pair.

Teacher–student deteriorated relative to its paired source-only model in 2 of 9 pairs. These failures remain in the result tables.

## Target test macro F1 (%)

Mean ± sample standard deviation over three training seeds. This is optimization variability on fixed temporal blocks, not an interval for independent machines.

| Transfer | Source only | Extra source training | Teacher–student | Source-only EMA control* |
|---|---:|---:|---:|---:|
| 0 → 1 hp | 85.16 ± 1.62 | 90.10 ± 3.16 | 87.68 ± 1.08 | 87.68 ± 1.08 |
| 0 → 2 hp | 85.13 ± 5.79 | 87.52 ± 3.84 | 86.81 ± 4.66 | 86.95 ± 4.62 |
| 0 → 3 hp | 83.67 ± 2.46 | 83.99 ± 1.26 | 83.71 ± 2.48 | 83.71 ± 2.48 |

*The source-only EMA control was added after inspecting the initial pilot. It uses the same source batches, optimizer settings, update count, and EMA coefficient, with no target inputs. No target-informed hyperparameter changes were made. Original three-method results remain in the `_initial` artifacts.

## Average across the three transfers and three seeds

Each of the nine task/seed scores has equal weight. This is not a pooled accuracy over every prediction.

| Method | Accuracy (%) | Macro F1 (%) |
|---|---:|---:|
| Source only | 81.95 | 84.65 |
| Extra source training | 84.80 | 87.20 |
| Teacher–student | 83.35 | 86.07 |
| Source-only EMA control* | 83.40 | 86.11 |

## Detection versus diagnosis

The source-only and teacher–student models achieved 100% faulty-window recall and 0% healthy-window false positives in all evaluated target task/seed pairs, using the multiclass argmax collapsed to healthy versus faulty. Their remaining errors are among fault types. This small binary detection problem is already saturated; multiclass diagnosis is the harder task here.

These figures do not establish field sensitivity or a real-time false-alarm rate. The test collection is small, derives from the same recording families, and has laboratory fault prevalence.

## What was actually built

- 40 original MATLAB recordings with source URLs, checksums, selected channel keys, and metadata flags.
- 3,926 windows from disjoint raw time regions; nominal 48 kHz to 12 kHz conversion and 1,024-sample windows.
- A 15,940-parameter custom 1D CNN with GroupNorm.
- Source training, matched extra-source optimization, EMA teacher–student commissioning, and an additional source-only EMA control.
- Three source training seeds, three directed target transfers, saved checkpoints, and separate evaluation.
- Label-free target archives and a replay with evaluator-label access blocked.
- A command-line predictor for explicitly identified MATLAB channels.
- Public CAT source code downloaded, entry-point import checked, and its CNN exercised on real CWRU data with a GPU optimizer step.

## Window counts

| Load | Fit | Validation | Test |
|---|---:|---:|---:|
| 0 hp | 319 | 91 | 101 |
| 1 hp | 685 | 215 | 219 |
| 2 hp | 701 | 220 | 228 |
| 3 hp | 700 | 220 | 227 |

Only load-0 validation labels select source checkpoints. Target validation partitions are unused. Target test labels are confined to retrospective evaluation.

## Data quality findings

- 174.mat (IR014_0): filename/channel-prefix mismatch: selected sole key X173_DE_time.
- 175.mat (IR014_1): additional DE arrays in MAT; selected filename-matching key.
- 99.mat (Normal_2): additional DE arrays in MAT; selected filename-matching key.
- 217.mat (IR021_3): additional DE arrays in MAT; selected filename-matching key.
- Shortest selected record: IR014_0, 63,788 samples, nominal 1.329 seconds.

## What the evidence supports

The complete custom software pipeline works on real public vibration recordings and can adapt without receiving true target labels. It produces reproducible checkpoints and reports. Under this specific pilot, confident target pseudo-labels do not outperform matched source-only controls.

## Limits and unfinished research

- The healthy files' nominal 48 kHz sampling rate is explicitly assumed because their official baseline table and MAT contents lack an explicit rate field. Corroborate it before final frequency-based conclusions.
- Temporal blocks share recordings, and load transfers may share physical bearing specimens.
- Only source load 0 and three target loads were tested, with three seeds and one fixed hyperparameter configuration.
- Three defect sizes are merged into broad fault-location classes; healthy has fewer windows than the pooled fault classes.
- Target commissioning data include mixed conditions. Healthy-only commissioning, unknown faults, and changing online streams are untested.
- The model is a custom EMA baseline. Calibration, adversarial alignment, and full CAT reproduction remain future work.
- Further method development must distinguish these inspected pilot tasks from fresh final evaluation.

## Why pseudo-label learning may be limited

Retrospective diagnostics on clean commissioning inputs from the fixed final teachers show strongly unequal coverage across true classes. The class labels used here were unavailable to adaptation.

| True class | Mean accepted fraction (%) | Mean correctness among accepted labels (%) |
|---|---:|---:|
| healthy | 74.76 | 100.00 |
| inner_race | 80.37 | 99.92 |
| rolling_element | 6.61 | 29.41 |
| outer_race | 59.47 | 99.90 |

Coverage averages the nine task/seed fractions. Correctness averages nonempty accepted subsets rather than pooling every prediction. These final-model clean-input diagnostics are not the augmented training trajectory.

Rolling-element windows receive little accepted supervision, and many accepted labels for that true class are incorrect. This is consistent with confidence filtering concentrating on easier examples and retaining some confident errors. Calibration or alternative reliability measures are hypotheses for future testing, not demonstrated solutions here.

## Files to inspect

- `outputs/pilot_v1/results_by_seed.csv`: every measured target result.
- `outputs/pilot_v1/results_summary.csv`: task-level means and sample standard deviations.
- `outputs/pilot_v1/paired_changes.csv`: paired gains and failures.
- `outputs/pilot_v1/predictions/`: saved test probabilities with IDs and evaluator labels.
- `outputs/pilot_v1/figures/`: real signals, comparison, confusion matrices, and learning/coverage plots.
- `outputs/pilot_v1/reproducibility_check.json`: identical replay with evaluator access blocked.
- `outputs/pilot_v1/cat_compatibility.json`: exact scope of the public-code check.
- `reports/EXPERIMENT_PROTOCOL_PILOT_V1.md`: the implemented method and information rules.

## A defensible progress statement

> I have implemented the data pipeline and a source-access teacher–student prototype, completed a three-seed cross-load pilot, and added controls for extra training and EMA. The initial results show that target pseudo-label learning adds little beyond those controls in this setup. I am now improving the evaluation design and investigating the adaptation mechanism.
