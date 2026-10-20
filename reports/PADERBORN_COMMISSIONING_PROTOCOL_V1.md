# Paderborn teacher-student commissioning protocol v1

Declared on 20 October 2026, before this adaptation experiment is trained or
evaluated. The already published Paderborn baseline test results have been
inspected. This extension is therefore **exploratory**, not an untouched
confirmatory experiment. No claim of preregistration with an external registry
is made. A timestamped local declaration hashes the configuration, this protocol,
and implementation before training. All original baseline evidence is retained.

## Question and comparison

Does learning from the reserved, unlabeled target windows add value beyond
continued supervised source training and temporal weight averaging? The primary
comparison is the final target-adapted EMA teacher against a source-only EMA
with the same initial weights, source batches, optimizer settings, update count
and EMA coefficient. Initial source-only and continued-source student models
provide the other two comparisons. Three seeds (7, 17, 27) and three targets give
nine paired task/seed comparisons. The source-only controls are trained once per
seed and evaluated on each target; these are not nine independent control fits.

## Data and task

Reuse the entire frozen [baseline protocol](PADERBORN_PROTOCOL_V1.md): binary
healthy/faulty classification, 15 source-fit bearings, 5 source-validation
bearings, 6 adaptation bearings, 6 evaluation bearings. Groups remain disjoint.
Source is N15_M07_F10; targets are N09_M07_F10, N15_M01_F10, N15_M07_F04.
Each target is adapted separately from the same seed's selected source model.
No adaptation is chained across conditions. No CWRU-to-Paderborn transfer is
performed. Nominal 64-to-16 kHz resampling, 2048-sample windows and per-window
z-scoring remain unchanged. Existing timing and single-healthy-test-bearing
limitations apply in full.

The trainer loads source fit/validation labels, but target archives contain only
signals and IDs. Target IDs are not decoded into class labels or passed as model
features. A scoped Python audit hook blocks evaluator archives, evaluation
inputs, raw data, curator label manifests, published results and report reads
during training. The source control additionally cannot load target inputs. This
is an accidental-access guard in this program, not a general security sandbox.
Metadata may describe dataset counts; no per-window target reference labels are
used to train, balance target batches, select settings or stop training.

## Fixed algorithm

Start from each existing source-validation-selected baseline checkpoint. Reset
Adam for every run; learning rate 0.0002, weight decay 0.00001, batch size 128.
Run 10 epochs of the source loader, with class-balanced source sampling. Each
source step uses one shuffled target batch, cycling through the target loader
when exhausted. The last short batches are retained. Source control and target
runs have identical source index sequences, checked by a hash. Augmentation and
target-shuffle random generators are independent of source sampling.

For target window x, teacher sees x + Gaussian noise with standard deviation
0.01. Student sees g*x + Gaussian noise with standard deviation 0.05, where
g is uniform on [0.9, 1.1]. These are perturbations of already standardized
signals; they are engineering assumptions about label preservation, not a
validated simulation of acquisition noise. The teacher softmax maximum gives
confidence and its argmax gives a hard pseudo-label. Accept confidence >= 0.9.

The student objective is supervised source cross entropy plus target weight
times masked pseudo-label cross entropy. Target weight is 0.5*epoch/10. The
target term is averaged over **all** windows in the target batch, so rejected
windows contribute zero and low coverage reduces the effective target weight.
No accepted windows means a finite zero target loss. Hard pseudo-label agreement
between weak and stronger views is the implemented consistency objective; there
is no additional KL/MSE consistency term.

The teacher is an independent, initially identical copy without gradients. After
each student optimizer step, teacher weights become 0.99*old_teacher +
0.01*new_student. Its buffers are copied from the student. GroupNorm does not
use running target batch statistics. The final epoch teacher is the reported
adapted model. The final student is saved only as a diagnostic. The control
produces both its final supervised student and its EMA from the same trajectory.

All adaptation settings are inherited from the previous CWRU pilot, not selected
by a Paderborn target score. Source validation is logged each epoch for student
and teacher but cannot select an adaptation checkpoint. This is source-access
unsupervised domain adaptation, not source-free adaptation and not a reproduction
of the complete Calibrated Adaptive Teacher method.

## Freeze, evaluation and analysis

Freeze all checkpoints and training artifacts before loading any evaluator
labels. Verify input, code and checkpoint hashes on evaluation. Fixed threshold
0.5, mean window softmax per recording, and primary recording macro F1 apply to
all four methods. Report accuracy, balanced accuracy, fault recall, healthy
false-positive rate, ROC AUC, average precision, confusion counts, window-level
metrics and per-bearing results. Also evaluate each adapted model at the source
condition as a forgetting diagnostic; it cannot influence checkpoint selection.

Report all seeds, mean and sample SD across three seeds per condition. Compare
paired macro-F1 differences against all controls, count negative differences,
and show all nine differences against source EMA. These are descriptive results
on correlated repeated conditions/bearings, not nine independent trials or
confidence intervals for an industrial bearing population. Do not choose the
best seed or target for the headline.

Log losses, confidence, accepted fraction, accepted pseudo-label counts by
predicted class, teacher/student disagreement and source validation. Preserve
the actual weak-view teacher decisions and window indices for every training
exposure. Only after freezing, join those indices to evaluator labels to measure
accepted pseudo-label precision and true-class coverage per epoch. Repeated
exposures are not independent samples. These retrospective labels cannot feed
back into training. End-of-epoch class concentration is a warning sign, not by
itself proof of erroneous collapse when a target population is imbalanced.

## Verification and stopping

Test label access boundaries, target schema, gradient masking/empty masks,
teacher independence, deterministic source sampling and a zero-target-weight
equivalence to the supervised control. Run the complete existing regression
suite. Replay seed 7 / N09_M07_F10 and its control with label access blocked;
require identical saved parameter tensors and pseudo-label traces on this
machine. Record software/GPU, update counts and provenance.

No target-driven rerun or hyperparameter search belongs to this version. Poor
results are a result to report. Any correction to a scientific bug after
evaluation must invalidate that run transparently and use a new version. Further
calibration/augmentation ablations and untouched confirmation are future work.

## Commands

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\train_paderborn_commissioning.py
.\.venv\Scripts\python.exe scripts\train_paderborn_commissioning.py --replay-seed 7 --replay-condition N09_M07_F10
.\.venv\Scripts\python.exe scripts\evaluate_paderborn_commissioning.py
.\.venv\Scripts\python.exe scripts\analyze_paderborn_commissioning.py
```

Training refuses to replace a started/completed run. Published results can be
restored for inspection without retraining. Original datasets and prepared
arrays are still needed for new training, replay and full evaluation.
