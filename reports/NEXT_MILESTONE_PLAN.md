# Agreed next thesis milestone

Recorded: 20 October 2026. The original plan below is retained for provenance.
The four items were subsequently implemented in this session; see the completion
record at the end and `THESIS_PROGRESS.md` for the current status.

The student wants a defensible intermediate milestone corresponding informally
to about 60% progress. No objective percentage or supervisor approval is claimed.

## Four priorities to retain

1. Implement Paderborn teacher-student commissioning using the reserved
   unlabeled target pools, source supervision, confidence-filtered pseudo-labels,
   prediction consistency and an EMA teacher. Do not assume it improves results.
2. Compare source-only, matched extra-source training, source-only EMA and
   teacher-student adaptation across three target conditions and three seeds.
   Report macro F1, balanced accuracy, fault recall, healthy false-positive rate,
   variability and negative transfer. Keep target labels out of training and
   model/hyperparameter selection.
3. Analyze adaptation behavior: pseudo-label acceptance, class imbalance/collapse,
   confidence and consistency histories, source-validation behavior and failure
   cases. Calibration is a later hypothesis, not an already demonstrated fix.
4. Draft the actual thesis: introduction/research questions, literature,
   datasets/protocol, implemented methods, followed by measured preliminary
   results and limitations. A research companion is not a finished thesis.

## Boundaries

- CWRU already has a custom EMA teacher-student pilot and matched controls;
  it did not demonstrate improvement beyond those simpler controls.
- Paderborn currently has only the source-only binary baseline. Its adaptation
  and comparison stages are planned, not completed.
- The existing Paderborn test results have been inspected. Subsequent development
  must be declared exploratory on this benchmark; do not tune against target test
  labels or describe previously inspected results as an untouched final test.
- Preserve current checkpoints, data splits and reports. Declare new experiment
  versions rather than silently replacing frozen baseline evidence.
- A graphical interface and additional dataset downloads are secondary priorities.
- This note alone did not authorize automatic commits or pushes of future changes.
  The user subsequently explicitly requested implementation of all four items and
  publication of the finished work to GitHub.

## Completion record

1. Implemented `paderborn_commissioning_v1`, preserving the frozen baseline files.
2. Ran three source controls and nine target adaptations; evaluated four methods
   across all three targets and seeds with fixed thresholds/final checkpoints.
3. Analyzed actual accepted pseudo-label exposures, consistency, validation,
   negative transfer and bearing-specific failures. Six measured figures and a
   full result report are available. No calibration fix is claimed.
4. Wrote `thesis/THESIS_DRAFT.md` with introduction, literature, datasets, methods,
   implementation, real results, discussion, conclusions, references and appendices.

Teacher mean macro F1 is 66.38% versus source EMA 66.84%; the implementation is
working, but the method has not demonstrated aggregate superiority. Exact local
replay and 23 tests pass. The milestone is complete as a prototype/experimental
draft; the thesis itself still needs review and stronger confirmation.
