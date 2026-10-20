# Understand the working thesis prototype

Read this together with the measured result report. It explains the actual code and choices implemented in the pilot, rather than every possible method in the literature guide.

## 1. What we built

We built a classifier that receives a short vibration sequence and predicts one of four condition classes. We trained it on recordings at load 0 hp. We then attempted to adapt it using unlabeled recordings at loads 1, 2, and 3 hp.

The engineering pipeline works end to end. The first scientific result is that the custom teacher–student procedure did not add a clear benefit beyond simpler training controls. That is a useful finding about this configuration, and it identifies what to investigate next.

## 2. What a downloaded file contains

The original measurements are MATLAB files. A filename such as `138.mat` is a dataset record identifier. Inside it, a key such as `X138_DE_time` identifies the drive-end vibration array. The values are successive acceleration measurements, not condition labels or timestamps.

Our downloader obtains files from the official tables and records what each file represents. It verifies that the selected channel is finite and nonconstant, calculates a checksum, and stores the exact channel key. This matters because some real CWRU files contain additional or mismatched variable names.

We selected the 48 kHz drive-end fault family and healthy baseline recordings. The healthy files lack an explicit sample-rate field in their baseline table and MAT contents; the pilot records their nominal 48 kHz rate as an assumption that remains to be corroborated. Do not say that the file itself supplied this rate.

## 3. How one long recording becomes examples

First we assign disjoint time regions to fit, validation, and test. We leave gaps between them. Then we filter and downsample each region independently to nominal 12 kHz. Finally we extract complete 1,024-sample windows without overlap.

One window is approximately 0.0853 seconds at that rate. It contains enough local waveform information for this pilot, but its duration is a design choice to study later.

Each window is standardized to mean zero and standard deviation near one. This makes the network emphasize waveform structure over overall amplitude. It also discards amplitude information that might otherwise help diagnosis. The original-amplitude figures are for inspection; the network receives standardized inputs.

The prepared collection contains 3,926 windows. The source fit set has 319, source validation has 91, and source test has 101. The three target test sets contain 219, 228, and 227 windows. These are window counts, not counts of independent bearings.

## 4. What the source and target mean

The source is the load-0 collection. We know its labels and use them for optimization and source model selection. Each target is a different load, treated separately. During adaptation the target archive exposes signals and IDs only.

The target commissioning pool contains healthy and faulty examples mixed together. The software does not know their true classes. This differs from collecting only healthy signals from a newly installed machine.

The conditions can share physical bearing specimens. Our temporal partitions also share parent recordings. Therefore the experiment answers a limited cross-load question. It does not establish that the model will work on a physically independent machine.

## 5. What a tensor is

A tensor is a numerical array. One input batch has shape `[batch, channels, time]`. For a full batch here, that is `[64, 1, 1024]`: 64 windows, one accelerometer channel, and 1,024 measurements per window.

The network returns a tensor of shape `[64, 4]`. Each row has four unrestricted scores called logits. Softmax turns that row into probabilities summing to one. Output index 0 means healthy, 1 inner race, 2 rolling element, and 3 outer race. This shared class map is part of the model artifact.

## 6. What the CNN learns

The first convolution applies short filters that move along the vibration sequence. Training changes those filter coefficients. Later layers combine local patterns into broader features. The classifier turns 64 final features into four scores.

Our custom model has 15,940 trainable parameters. It uses GroupNorm to avoid hidden changes to running batch statistics when target data are introduced. This is a deliberate simplification for the pilot; it is not the exact CAT model.

For each labeled source window, cross entropy penalizes a low predicted probability on its correct class. Backpropagation calculates which parameter changes reduce that objective, and Adam updates the parameters.

## 7. How the source model is selected

We train for 30 source epochs and evaluate only source validation after each epoch. We save the earliest checkpoint with the highest source-validation macro F1. We repeat this for seeds 7, 17, and 27.

The best selected epochs were 30, 29, and 28, respectively. The model is therefore not chosen because it happened to perform well on the target test. The source checkpoint for each seed supplies the same starting weights to its compared methods.

Source fit classes are imbalanced because three sizes are pooled into each fault class while healthy has one record per load. The training sampler gives inverse-frequency weight to source examples. It uses source labels, which are allowed. It does not balance target batches using hidden labels.

## 8. How the teacher–student step works

We copy the source model into an independent student and teacher. The teacher has no gradients. It predicts probabilities for a target window with a small noise perturbation. If the highest probability is at least 0.90, its predicted class becomes an accepted pseudo-label.

The student sees another version of that window, with slightly stronger noise and a small random gain change. It learns to predict the accepted teacher class. It also continues learning from labeled source batches.

The objective is source cross entropy plus a weighted, masked target cross entropy. Rejected examples contribute zero to the target term. The denominator remains the full target batch, so low acceptance reduces the effective target contribution. The target weight rises linearly to 0.5 over 10 adaptation epochs.

After each optimizer update, the teacher follows the student:

\[
\theta_T \leftarrow 0.99\theta_T + 0.01\theta_S.
\]

This averaging can stabilize predictions. It can also preserve old mistakes. Confidence filtering does not guarantee correctness.

Each adaptation job has 50 optimizer updates because the source loader defines five updates per epoch. The selected model is always the final teacher. There is no target-accuracy-based stopping.

## 9. Why we needed controls

Comparing the source model with the adapted teacher changes several things: it adds training steps, introduces target learning, and averages weights. An improvement alone would not tell us which change mattered.

The extra-source control adds the same number of source optimizer updates without target learning. The additional source-only EMA control averages that control's weights with the same coefficient. The latter was introduced after examining the first pilot, and we disclose that timing.

The source-only EMA replay generated exactly the same student parameters as the original extra-source run. This verifies the matched optimization path. Its teacher predictions almost completely matched the target-adapted teachers on test classification.

## 10. What the measured result means

Mean macro F1 across the three tasks and three seeds was 84.65% for source-only, 87.20% for extra source training, 86.07% for teacher–student, and 86.11% for source-only EMA.

Teacher–student is 1.41 percentage points above the original baseline, but 1.14 points below extra source training. It is also 0.05 points below the averaging control. We therefore cannot attribute a useful additional gain to unlabeled target learning in this pilot.

Two of the nine task/seed pairs deteriorated after teacher–student adaptation relative to source-only. We retain those results. A method does not become successful merely because its mean score increases.

The three seeds measure sensitivity to optimization on the same fixed recordings. They do not measure variability across three independent machines.

A retrospective diagnostic helps explain the difficult class. At threshold 0.90, final teachers accepted only about 6.6% of rolling-element windows on average. Correctness among those accepted labels averaged about 29.4% over nonempty task/seed subsets. Other classes had much more reliable accepted labels. Thus the student receives little useful target supervision for a major source of error. This is a plausible mechanism to investigate; it does not establish that a particular new calibration method will solve it.

## 11. Detection and diagnosis are different

If the true class is inner race and the network predicts outer race, it has made a diagnosis error. It may still have correctly detected that a fault exists.

In this small pilot, source-only and teacher–student separate healthy from faulty target windows perfectly using the multiclass argmax. The unresolved errors are among fault types. The thesis must explain that binary detection is easy in this subset rather than claim that adaptation solved a difficult detection problem.

A tougher evaluation might hold out independent bearing instances, different damage origins, or another dataset. Those are next research questions, not results already established here.

## 12. How we checked that labels were not used

Target fit archives contain only inputs and IDs. The training function reads that archive, source labels, and its source checkpoint. Target labels are in a separate evaluator directory.

A replay test installed a Python audit hook that raises an error if the commissioning process tries to open an evaluator-label file. The adaptation completed, and every teacher parameter matched the original run exactly. This is stronger evidence than merely naming a function “unsupervised.”

The evaluator later joins fixed predictions to labels to calculate metrics. That label access is necessary to measure performance and is permitted after the method and checkpoint are fixed.

## 13. What was done with GitHub code

We downloaded the official CAT repository, recorded its commit, checked that its source-training command imports, and exercised its public CNN on a real prepared CWRU source batch. Forward calculation, backward gradients, and the GPU optimizer update passed.

That is a compatibility check. We have not reproduced full CAT training or its published Paderborn results. Our pilot has no teacher calibration, adversarial alignment, or adaptive threshold. Explain this distinction directly if asked.

## 14. What to open in VS Code

Start with `configs/pilot_v1.json`, which contains the settings. Read `thesislib/common.py` for the network. Read `scripts/prepare_cwru.py` for the partitions, then `scripts/run_pilot.py` for source training and commissioning. `scripts/evaluate_pilot.py` is the only experiment stage that joins target predictions to condition labels.

Read `reports/PILOT_RESULTS.md` alongside `outputs/pilot_v1/figures/method_comparison.png`. Use `confusion_matrices.png` to see which faults are confused. The learning/coverage plot shows source validation and the fraction of accepted target pseudo-labels, not target training accuracy.

## 15. What to do before explaining this work tomorrow

Practice explaining one window's full path: raw recording, time region, resampling, standardization, CNN, probabilities, pseudo-label acceptance, student loss, teacher update, and final separate evaluation.

Then explain the result using the controls. A defensible sentence is: “I implemented the full custom pipeline and ran a preliminary cross-load experiment; the controls show that unlabeled target learning has not yet provided an additional benefit in this setup.”

Be ready to identify the sample-rate assumption and the shared-recording limitation. Discussing them demonstrates that you understand what your evidence can and cannot establish.
