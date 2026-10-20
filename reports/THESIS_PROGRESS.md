# Thesis implementation progress

Updated: 19 October 2026.

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

## Current finding

Teacher–student mean macro F1 is 86.07%, compared with 84.65% source-only, 87.20% extra source training, and 86.11% source-only EMA. The pilot does not demonstrate a benefit beyond the simpler controls. Binary healthy/faulty separation is already saturated for the source-only model in this small subset.

The separate Paderborn binary baseline shows substantial condition sensitivity and seed variability. Recording macro F1 means are 84.26% at the source condition, 48.19% with lower speed, 89.68% with lower torque and 31.14% with lower radial force. Under lower radial force mean fault recall is only 18.67%. Paderborn teacher-student adaptation has not yet been implemented. See `PADERBORN_BASELINE_RESULTS.md`; these binary values are not directly comparable with four-class CWRU results. Only one independent healthy bearing is in this Paderborn evaluation split.

## Remaining before a final thesis study

1. Corroborate sample-rate metadata and strengthen recording/specimen grouping.
2. Choose a final development/evaluation boundary after these pilot tasks have been inspected.
3. Run direct pseudo-label, threshold, and augmentation ablations.
4. Investigate target-data value through a justified method extension, such as calibration or alignment, without tuning on final test labels.
5. Reproduce the complete selected reference method, including exact data and selection differences.
6. Extend the completed Paderborn bearing-disjoint baseline with predeclared adaptation/controls and additional untouched confirmation splits where feasible.
7. Complete the thesis chapters using verified methods and measured results.
8. Prepare presentation slides when requested.

## Immediate student task

Read `PADERBORN_START_HERE.md` and run its first prediction command. Send the terminal output or prediction JSON path. Then inspect the source-only result table and discuss why software execution is not the same as reliable fault detection. The next coding stage is a separately declared Paderborn teacher-student experiment with matched controls, not target-test-driven baseline tuning.
