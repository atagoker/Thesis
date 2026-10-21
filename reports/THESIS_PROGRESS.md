# Thesis implementation progress

Updated: 22 October 2026.

The project has reached a working-prototype and first-results milestone. A precise percentage of the whole thesis is not established: the implementation, final experimental study, written thesis, and defense have different remaining workloads.

## Completed

- An isolated Python 3.12/PyTorch environment in VS Code, with GPU execution verified.
- A substantial literature and methodology guide from earlier work.
- Download and validation of 40 original CWRU files with checksums and provenance.
- A documented four-class source/target pilot protocol and explicit metadata assumptions.
- Time-blocked preparation, independent regional filtering, nonoverlapping windows, and input/evaluator separation.
- A trained source-only custom 1D CNN, extra-source control, EMA teacher–student prototype, and extra source-only EMA control.
- Three training seeds and three target-load tasks with fixed saved checkpoints and per-run tables.
- Retrospective metrics, confusion matrices, real-signal plots, and learning/coverage curves.
- Six implementation/data checks and an exact label-access-blocked adaptation replay.
- A command-line prediction demonstration.
- A compatibility run of the official CAT CNN on real prepared CWRU data.
- Method, result, and learning documents explaining the implementation.
- Paderborn import: all 32 archives, 2,560 recordings, originals retained.
- Full Paderborn numerical audit: 2,559 valid/readable recordings and one explicitly excluded unreadable file outside the selected model partitions.
- A predeclared binary, bearing-disjoint Paderborn split with separate unlabeled adaptation and evaluator archives.
- Three GPU-trained Paderborn source-only baselines, frozen evaluation on four conditions, and exact seed-7 replay.
- A Paderborn prediction command, beginner run guide, per-bearing diagnostics and measured result report.
- Fifteen combined implementation/data tests passing; original CWRU demonstration still works.
- A separately declared Paderborn teacher-student commissioning implementation with
  matched continued-source and source-EMA controls, three targets and three seeds.
- Twelve new training trajectories (3 controls + 9 adaptations), 730 matched
  source updates each, and 24 frozen new checkpoints including diagnostic students.
- All four methods evaluated on the same held-out bearings: 36 primary
  method/task/seed results, with additional source-retention and window metrics.
- An audit of 90 epochs of actual pseudo-label exposures, six new figures,
  class-specific coverage, accepted precision and per-bearing failure analysis.
- Exact seed-7 reduced-speed replay of model tensors, histories and pseudo traces.
- Twenty-three tests now pass, including the original fifteen; the earlier count
  above records the baseline milestone rather than a second independent test suite.
- Adapted-model CLI inference on real healthy and faulty reduced-speed recordings.
- A substantive [English thesis draft](../thesis/THESIS_DRAFT.md), including eight
  chapters, references and evidence/reproduction appendices.
- AtaScope, a native Windows/Qt desktop interface with a vivid dark theme, Paderborn/CWRU
  signal inspection and GPU inference, real batch/epoch commissioning updates,
  saved experiment comparisons, session logging and cancellation.
- Desktop end-to-end validation: real predictions, 730 training updates, exact
  teacher-state/source-sequence parity and cancellation. The full unit suite now
  passes 26 tests. See [the desktop guide](DESKTOP_UI.md).

## Current finding

Teacher–student mean macro F1 is 86.07%, compared with 84.65% source-only, 87.20% extra source training, and 86.11% source-only EMA. The pilot does not demonstrate a benefit beyond the simpler controls. Binary healthy/faulty separation is already saturated for the source-only model in this small subset.

The separate Paderborn binary baseline shows substantial condition sensitivity and
seed variability. The new commissioning study has mean target recording macro F1
of 56.34% initial source-only, 65.42% continued-source, 66.84% source EMA and 66.38%
teacher-student. Against source EMA, the teacher improves three task/seed pairs,
worsens three and ties three. The mean difference is -0.46 percentage points.
Under lower radial force its fault recall is only 21.33%; accepted pseudo-label
precision is 20.29–28.15% in the final epoch. See
[the commissioning report](PADERBORN_COMMISSIONING_RESULTS.md). These binary values
are not directly comparable with four-class CWRU results. Only one independent
healthy bearing is in the Paderborn evaluation split. This extension is exploratory
because baseline test results were inspected before its design.

## Remaining before a final thesis study

1. Corroborate sample-rate metadata and strengthen recording/specimen grouping.
2. Choose a final development/evaluation boundary after these pilot tasks have been inspected.
3. Run direct pseudo-label, threshold, and augmentation ablations.
4. Investigate target-data value through a justified method extension, such as calibration or alignment, without tuning on final test labels.
5. Reproduce the complete selected reference method, including exact data and selection differences.
6. Add a genuinely independent confirmation study where feasible; the declared
   Paderborn adaptation and control extension itself is now complete.
7. Review and revise the written thesis draft with the student/supervisor, then
   apply institutional formatting and incorporate any further validated experiments.
8. Prepare presentation slides when requested.

## Immediate student task

Read [the Turkish milestone guide](COMMISSIONING_MILESTONE_TR.md), then the
[English thesis draft](../thesis/THESIS_DRAFT.md). Be able to explain why the
initial-model improvement does not prove a benefit over source EMA, why low
false-positive rate can coexist with missed faults, and why high confidence can
produce incorrect pseudo-labels. The agreed four-item milestone is implemented;
final-thesis acceptance and an exact completion percentage remain supervisor decisions.
