"""Generate the progress report directly from measured pilot artifacts."""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thesislib.common import ROOT, read_json


def main():
    destination = ROOT / "outputs/pilot_v1"
    frame = pd.read_csv(destination / "results_by_seed.csv")
    summary = pd.read_csv(destination / "results_summary.csv")
    evaluation = read_json(destination / "evaluation.json")
    prepared = read_json(ROOT / "data/manifests/pilot_v1_summary.json")
    catalog = read_json(ROOT / "data/manifests/cwru48_records.json")
    train = read_json(destination / "training_complete.json")
    names = {"source_only": "Source only", "continued_source": "Extra source training",
             "teacher_student": "Teacher–student", "source_ema_control": "Source-only EMA control*"}
    rows = ["# Measured CWRU pilot results", "", "Date: 19 October 2026. Status: preliminary pilot.", "",
            "This report comes from saved prediction and metric files. It does not claim final thesis completion or industrial validation.", "",
            "## Main finding", "",
            f"Across the nine task/seed pairs, teacher–student changed macro F1 by **{evaluation['mean_paired_teacher_minus_source_pp']:+.2f} percentage points** relative to source-only, "
            f"**{evaluation['mean_paired_teacher_minus_continued_pp']:+.2f} points** relative to extra source training, and "
            f"**{evaluation['mean_paired_teacher_minus_source_ema_pp']:+.2f} points** relative to the source-only EMA control.", "",
            "The pilot therefore does not demonstrate an additional benefit from unlabeled target learning. Control runs reproduce the apparent gains without target inputs. The source-only EMA control matches teacher–student test macro F1 in eight of nine task/seed pairs, with a small difference in the remaining pair.", "",
            f"Teacher–student deteriorated relative to its paired source-only model in {evaluation['negative_transfer_task_seed_pairs']} of {evaluation['task_seed_pairs']} pairs. These failures remain in the result tables.", "",
            "## Target test macro F1 (%)", "", "Mean ± sample standard deviation over three training seeds. This is optimization variability on fixed temporal blocks, not an interval for independent machines.", "",
            "| Transfer | Source only | Extra source training | Teacher–student | Source-only EMA control* |",
            "|---|---:|---:|---:|---:|"]
    for load in [1,2,3]:
        values = []
        for method in names:
            row = summary[(summary.target_load_hp==load) & (summary.method==method)].iloc[0]
            values.append(f"{row.macro_f1_mean*100:.2f} ± {row.macro_f1_std*100:.2f}")
        rows.append(f"| 0 → {load} hp | " + " | ".join(values) + " |")
    rows += ["", "*The source-only EMA control was added after inspecting the initial pilot. It uses the same source batches, optimizer settings, update count, and EMA coefficient, with no target inputs. No target-informed hyperparameter changes were made. Original three-method results remain in the `_initial` artifacts.", "",
             "## Average across the three transfers and three seeds", "",
             "Each of the nine task/seed scores has equal weight. This is not a pooled accuracy over every prediction.", "",
             "| Method | Accuracy (%) | Macro F1 (%) |", "|---|---:|---:|"]
    for method, name in names.items():
        group = frame[frame.method==method]
        rows.append(f"| {name} | {group.accuracy.mean()*100:.2f} | {group.macro_f1.mean()*100:.2f} |")
    rows += ["", "## Detection versus diagnosis", "",
             "The source-only and teacher–student models achieved 100% faulty-window recall and 0% healthy-window false positives in all evaluated target task/seed pairs, using the multiclass argmax collapsed to healthy versus faulty. Their remaining errors are among fault types. This small binary detection problem is already saturated; multiclass diagnosis is the harder task here.", "",
             "These figures do not establish field sensitivity or a real-time false-alarm rate. The test collection is small, derives from the same recording families, and has laboratory fault prevalence.", "",
             "## What was actually built", "",
             f"- {len(catalog['records'])} original MATLAB recordings with source URLs, checksums, selected channel keys, and metadata flags.",
             f"- {prepared['total_windows']:,} windows from disjoint raw time regions; nominal 48 kHz to 12 kHz conversion and 1,024-sample windows.",
             f"- A {train['parameter_count']:,}-parameter custom 1D CNN with GroupNorm.",
             "- Source training, matched extra-source optimization, EMA teacher–student commissioning, and an additional source-only EMA control.",
             "- Three source training seeds, three directed target transfers, saved checkpoints, and separate evaluation.",
             "- Label-free target archives and a replay with evaluator-label access blocked.",
             "- A command-line predictor for explicitly identified MATLAB channels.",
             "- Public CAT source code downloaded, entry-point import checked, and its CNN exercised on real CWRU data with a GPU optimizer step.", "",
             "## Window counts", "", "| Load | Fit | Validation | Test |", "|---|---:|---:|---:|"]
    counts = pd.DataFrame(prepared["counts"]).groupby(["load_hp","split"]).windows.sum()
    for load in range(4):
        rows.append(f"| {load} hp | {counts.loc[(load,'fit')]} | {counts.loc[(load,'validation')]} | {counts.loc[(load,'test')]} |")
    rows += ["", "Only load-0 validation labels select source checkpoints. Target validation partitions are unused. Target test labels are confined to retrospective evaluation.", "",
             "## Data quality findings", ""]
    for record in catalog["records"]:
        for flag in record["quality_flags"]:
            rows.append(f"- {record['filename']} ({record['name']}): {flag}.")
    shortest = prepared["shortest_record"]
    rows += [f"- Shortest selected record: {shortest['name']}, {shortest['samples']:,} samples, nominal {shortest['duration_s_nominal']:.3f} seconds.", "",
             "## What the evidence supports", "",
             "The complete custom software pipeline works on real public vibration recordings and can adapt without receiving true target labels. It produces reproducible checkpoints and reports. Under this specific pilot, confident target pseudo-labels do not outperform matched source-only controls.", "",
             "## Limits and unfinished research", "",
             "- The healthy files' nominal 48 kHz sampling rate is explicitly assumed because their official baseline table and MAT contents lack an explicit rate field. Corroborate it before final frequency-based conclusions.",
             "- Temporal blocks share recordings, and load transfers may share physical bearing specimens.",
             "- Only source load 0 and three target loads were tested, with three seeds and one fixed hyperparameter configuration.",
             "- Three defect sizes are merged into broad fault-location classes; healthy has fewer windows than the pooled fault classes.",
             "- Target commissioning data include mixed conditions. Healthy-only commissioning, unknown faults, and changing online streams are untested.",
             "- The model is a custom EMA baseline. Calibration, adversarial alignment, and full CAT reproduction remain future work.",
             "- Further method development must distinguish these inspected pilot tasks from fresh final evaluation.", "",
             "## Files to inspect", "",
             "- `outputs/pilot_v1/results_by_seed.csv`: every measured target result.",
             "- `outputs/pilot_v1/results_summary.csv`: task-level means and sample standard deviations.",
             "- `outputs/pilot_v1/paired_changes.csv`: paired gains and failures.",
             "- `outputs/pilot_v1/predictions/`: saved test probabilities with IDs and evaluator labels.",
             "- `outputs/pilot_v1/figures/`: real signals, comparison, confusion matrices, and learning/coverage plots.",
             "- `outputs/pilot_v1/reproducibility_check.json`: identical replay with evaluator access blocked.",
             "- `outputs/pilot_v1/cat_compatibility.json`: exact scope of the public-code check.",
             "- `reports/EXPERIMENT_PROTOCOL_PILOT_V1.md`: the implemented method and information rules.", "",
             "## A defensible progress statement", "",
             "> I have implemented the data pipeline and a source-access teacher–student prototype, completed a three-seed cross-load pilot, and added controls for extra training and EMA. The initial results show that target pseudo-label learning adds little beyond those controls in this setup. I am now improving the evaluation design and investigating the adaptation mechanism.", ""]
    diagnostic_path = destination / "pseudolabel_diagnostics_by_class.csv"
    if diagnostic_path.exists():
        diagnostic = pd.read_csv(diagnostic_path)
        teacher = diagnostic[diagnostic.model == "teacher_student"].groupby("true_class")[["coverage", "accepted_precision"]].mean()
        section = ["## Why pseudo-label learning may be limited", "",
                   "Retrospective diagnostics on clean commissioning inputs from the fixed final teachers show strongly unequal coverage across true classes. The class labels used here were unavailable to adaptation.", "",
                   "| True class | Mean accepted fraction (%) | Mean correctness among accepted labels (%) |",
                   "|---|---:|---:|"]
        for class_name in ["healthy", "inner_race", "rolling_element", "outer_race"]:
            item = teacher.loc[class_name]
            section.append(f"| {class_name} | {item.coverage*100:.2f} | {item.accepted_precision*100:.2f} |")
        section += ["", "Coverage averages the nine task/seed fractions. Correctness averages nonempty accepted subsets rather than pooling every prediction. These final-model clean-input diagnostics are not the augmented training trajectory.", "",
                    "Rolling-element windows receive little accepted supervision, and many accepted labels for that true class are incorrect. This is consistent with confidence filtering concentrating on easier examples and retaining some confident errors. Calibration or alternative reliability measures are hypotheses for future testing, not demonstrated solutions here.", ""]
        position = rows.index("## Files to inspect")
        rows[position:position] = section
    report = ROOT / "reports/PILOT_RESULTS.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text("\n".join(rows), encoding="utf-8")
    print(f"Generated {report}", flush=True)


if __name__ == "__main__":
    main()
