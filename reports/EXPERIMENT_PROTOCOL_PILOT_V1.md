# Pilot protocol: CWRU bearing commissioning

Date: 19 October 2026. This is an implemented pilot protocol, not final independent-bearing validation.

## Research question and setting

Does an EMA teacher trained with confident pseudo-labels from unlabeled target vibration windows improve a small source-trained bearing classifier? Labeled source recordings remain available during adaptation. This is source-access unsupervised domain adaptation.

The primary task is four-class diagnosis: healthy, inner-race fault, rolling-element fault, and outer-race fault. Binary detection derives from healthy versus any fault. Results for the two tasks must remain distinct.

## Data selection

Download 40 original MATLAB files directly from the official CWRU tables: four healthy baselines and 36 drive-end fault recordings. The fault recordings come from the 48 kHz family, with three defect diameters (0.007, 0.014, 0.021 inches), four loads (0, 1, 2, 3 hp), and three fault locations. Outer-race data use the explicit @6 clock-position entries.

The normal-baseline table and MATLAB files do not explicitly encode their sampling rate. This pilot adopts the conventional nominal 48 kHz normal-baseline rate and marks it as an assumption in every corresponding manifest row. It remains a metadata item to corroborate before final physical-frequency claims. The fault-table sampling family is directly documented.

Sources:

- [Official 48 kHz fault table](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)
- [Official normal table](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
- [Official apparatus](https://engineering.case.edu/bearingdatacenter/apparatus-and-procedures)

Keep original files unchanged. Store URL, SHA-256 checksum, byte count, exact selected MATLAB key, signal length, recorded RPM where available, fault metadata, and quality flags in `data/manifests/cwru48_records.json`.

Some files contain mismatched or additional channel prefixes. If the filename-matching DE key exists, choose it explicitly. Otherwise accept a sole DE channel and flag the mismatch. Multiple DE channels without the expected key cause an error. This rule was fixed during data preparation before model evaluation.

## Partitioning and preprocessing

For each raw recording of N samples, define boundaries at floor(0.6N) and floor(0.8N). Remove 2,048 raw samples on each side of each boundary, giving a 4,096-sample unused gap between adjacent regions. Use the early region for fit, middle region for validation, and final region for test.

Resample each region separately from nominal 48 kHz to 12 kHz with `scipy.signal.resample_poly(up=1, down=4)`. Trim 128 processed samples from both ends of each region. This prevents the resampling filter from reading observations in another partition and removes boundary transients from retained windows.

Make complete nonoverlapping windows of 1,024 processed samples with stride 1,024. Their nominal duration is about 85.3 ms. Standardize each window using its own mean and standard deviation, with a numerical floor for the denominator. This suppresses absolute amplitude differences and does not use labels. Raw-amplitude inspection plots precede standardization.

The 4,096-sample gap was chosen before training so that the short IR014_0 recording retains at least one validation window. It prevents direct sample overlap; it does not establish statistical independence.

## Domains and allowed information

Source: load 0 hp. Targets: loads 1, 2, and 3 hp, evaluated as three separate transfers. Source fit and validation archives contain labels. Target fit archives contain only arrays and window IDs; true target labels reside in an evaluator directory.

The target fit pool contains mixed conditions. It is not a healthy-only commissioning experiment. Target validation windows remain unused. Test windows are withheld until every initial checkpoint has been fixed.

Source and target measurements can share physical bearing specimens across loads. Fit, validation, and test regions also share their parent recording. This pilot therefore studies temporal holdout within cross-load tasks, not independent specimens or machines.

## Model and source training

Use a custom CNN with three Conv1D blocks (16, 32, 64 channels), GroupNorm, ReLU, pooling, and a four-output classifier. It contains 15,940 trainable parameters. GroupNorm avoids adapting hidden running batch statistics during the comparison.

Source training uses Adam, learning rate 0.001, weight decay 0.00001, batch size 64, and 30 epochs. A replacement sampler balances source classes using source labels. Choose the earliest checkpoint achieving the highest source-validation macro F1. Seeds are 7, 17, and 27. Use the same chosen source checkpoint for all paired comparisons within a seed.

## Initial comparisons, fixed before target evaluation

1. **Source-only:** the source-validation-selected checkpoint.
2. **Continued source training:** 10 additional source-only epochs with Adam at 0.0002 and the same weight decay and batch size. Report the final student.
3. **EMA teacher–student:** the same extra source-update budget, plus unlabeled target inputs. Teacher weights begin at the selected source checkpoint and follow the student by EMA with alpha 0.99. Report the final teacher.

At each adaptation step, the teacher predicts a weak view with Gaussian noise standard deviation 0.01. Accept maximum softmax probability at least 0.90. The student predicts a stronger view with gain sampled uniformly in [0.9, 1.1] and noise standard deviation 0.05. Compute masked target cross entropy with the full target-batch denominator. Add it to source cross entropy with a weight ramping linearly from 0.05 to 0.5 over 10 epochs.

The source loader defines each adaptation epoch (five updates in this dataset). Each method has 50 additional source optimizer updates. Teacher and student share architecture but have independent parameters. Teacher gradients are disabled. Target data receive no condition labels. There is no calibration, domain-adversarial loss, adaptive threshold, or source-free setting in this pilot.

## Additional control after the initial pilot

After inspecting the initial three-method result, add a source-only EMA control. Replay exactly the continued-source training and average its student weights with alpha 0.99. This uses no target inputs or labels. The replayed student parameters matched the original continued-source checkpoints exactly in all three seeds.

This is an additional methodological control, not a member of the original predeclared method set. The original three-method tables and evaluation remain preserved with `_initial` filenames. No threshold, augmentation, learning rate, epoch budget, or other hyperparameter was tuned using the target results.

## Evaluation and interpretation

Run a separate evaluator only after saving checkpoints and their checksums. Report macro F1 as primary, with accuracy, balanced accuracy, per-class reports, confusion matrices, fault recall, healthy false-positive rate, ROC AUC, negative log likelihood, and ECE as supporting evidence.

Report every seed/task pair. Show mean and sample standard deviation over three training seeds. These describe training variability on fixed observations, not uncertainty across machines. Compare paired macro F1 differences in percentage points.

Calculate pseudo-label precision retrospectively from the fixed final model on clean commissioning inputs. This diagnostic is different from the sequence of augmented pseudo-labels used during training. It must not influence checkpoint selection.

## Implementation verification

The checks cover nonoverlapping raw regions/windows, finite normalization, target archives without labels, EMA arithmetic, accepted-label gradients, and finite zero loss for an empty acceptance mask. An adaptation replay with an audit hook blocking evaluator-label file access produced identical teacher parameters on this machine.

The CAT reference repository was downloaded at commit `354cf94aef4f0afd1f81d7ef645bd40ffc68d8d5`. Its source-training entry point imports successfully, and its public CNN completed a GPU forward/backward/optimizer update on a real prepared CWRU batch. Full published CAT training has not been reproduced. The custom pilot must not be described as full CAT.

## Work remaining before final thesis conclusions

Corroborate the normal-file sample-rate metadata; verify independent-bearing split feasibility; reserve fresh evaluation tasks for further method development; add direct pseudo-label and confidence-filter ablations; investigate why target learning adds little in this pilot; reproduce the complete chosen reference method; and extend to another dataset or stronger grouping if feasible.
