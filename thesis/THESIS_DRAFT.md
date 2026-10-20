# Implementation of Label-Free Teacher–Student Commissioning for Vibration-Signal-Based Fault Detection

**Author:** Ata Göker  
**Degree:** Bachelor of Science  
**Study type:** Analytical and experimental  
**Working draft:** 20 October 2026

This is a substantive thesis draft based on the implemented repository and
measured experiments. It requires the student's review, supervisor feedback,
institutional formatting and further validation before submission. It is not a
claim that the complete degree project has reached a mathematically measurable
percentage. The results describe the current fixed experiments, including
unsuccessful adaptation. Citations refer to the reference list at the end.

## Abstract

A vibration-based diagnostic model can perform differently when the operating
condition or bearing specimen changes. Commissioning such a model on a target
condition is difficult when measurements are available but their fault labels
are not. This work implements a source-access teacher–student procedure in which
a model is first trained with labeled source data, then adapted using source
supervision and unlabeled target vibration signals. Confident teacher predictions
provide pseudo-labels for a student receiving perturbed versions of the same
signals. An exponential moving average of student parameters updates the teacher.

The engineering contribution is a Python/PyTorch prototype with data import,
signal checks, split-aware preprocessing, training, separate evaluation,
prediction and reproducible result reporting. A CWRU pilot uses four classes and
temporal separation. A second, binary Paderborn study uses bearing-disjoint
source-fit, source-validation, target-adaptation and evaluation groups. Three
random seeds and three target conditions are evaluated with initial source-only,
continued-source, source-only EMA and teacher–student methods.

In Paderborn, mean recording-level macro F1 over the nine task/seed combinations
is 56.34% for the initial model, 65.42% after additional source training, 66.84%
for source EMA and 66.38% for teacher–student adaptation. The target-adapted
teacher therefore improves over the initial model but does not demonstrate an
aggregate benefit over matched source EMA. Under reduced radial force, accepted
pseudo-label precision in the final epoch is only 20.29–28.15%, despite a 0.9
confidence filter. This identifies unreliable confident predictions as a
practical failure mode. The findings are exploratory and limited by one fixed
bearing split, one held-out healthy bearing, and previously inspected baseline
test results. They support careful control comparisons and label-access
separation, rather than a claim of general adaptation superiority.

**Keywords:** bearing fault detection; vibration; domain adaptation;
teacher–student; pseudo-labels; consistency learning; commissioning; PyTorch.

## 1. Introduction

### 1.1 Engineering problem

The diagnostic problem considered here begins with a measurement: a sequence of
vibration values from a machine containing a rolling bearing. The intended output
is a decision about bearing condition. In the principal experiment the decision
is binary, healthy or faulty. The algorithm does not determine the remaining
useful life of the bearing, prescribe a maintenance action, or identify every
possible damage mechanism. Those are different tasks requiring different data
and evaluation criteria.

Supervised learning requires examples whose correct condition is known. In a
laboratory benchmark, bearing identities and experimental descriptions provide
reference labels. On a newly commissioned machine, collecting vibration is often
more straightforward than obtaining representative labeled examples of every
fault. A diagnostic system may therefore begin with a model learned elsewhere
and a collection of target measurements with no labels available to the learning
procedure. The thesis investigates this restricted information setting.

Even when the class names remain unchanged, measurements can differ between
source and target. A change in speed, torque or radial force can change the
relationship between an observed signal and the features used by a classifier.
Different physical bearings can also produce different patterns. These changes
motivate testing a model outside its source operating condition. They also make
it unsafe to assume that a numerical confidence score transfers unchanged to a
new condition. Domain-adaptive fault diagnosis explicitly addresses this
source/target distinction [R1].

In this thesis, commissioning is implemented as an offline adaptation stage. A
fixed source model is copied, an unlabeled target pool is used for a fixed number
of updates, and a final model is saved for subsequent prediction. This is a
computational approximation of commissioning. The experiments use published
laboratory recordings, not a newly installed industrial sensor, a continuous
stream, or a controlled field deployment. Maintaining this distinction is
necessary when interpreting the engineering achievement.

### 1.2 Objective and research questions

The objective is to develop, implement and experimentally examine a diagnostic
algorithm trained on labeled source vibration data and adapted without target
training labels. The central scientific question is whether unlabeled target
signals provide useful additional information through teacher–student learning.
Simply showing that a model changes after adaptation is insufficient. An
improvement could also arise from additional supervised updates or from averaging
model weights over time.

The study therefore asks four questions. First, how does the initial source model
perform on different operating conditions and unseen bearing specimens? Second,
does target adaptation improve diagnosis beyond matched continued-source and
source-only EMA controls? Third, what do confidence, accepted pseudo-labels and
teacher/student agreement reveal about successful or unsuccessful adaptation?
Fourth, can the complete workflow be implemented with traceable data, explicit
label boundaries and repeatable saved outputs?

The desired research outcome is a defensible answer to those questions. A result
in which the proposed procedure does not outperform a simpler control is still
informative, provided that the comparison is fair and the limitation is reported.
No assumption of teacher–student superiority is built into the acceptance
criteria for the software or the interpretation of its experiments.

### 1.3 Scope and contributions

The implemented contribution is a small, inspectable diagnostic system. It
imports real data, checks signal integrity, prepares declared partitions, trains
models on a GPU, preserves adaptation traces and evaluates frozen checkpoints.
The study additionally makes a distinction between software verification and
scientific evidence. Unit tests establish selected invariants of the
implementation; measured held-out performance addresses diagnostic usefulness.
Neither substitutes for the other.

The algorithm is a custom confidence-filtered EMA teacher–student implementation.
It is informed by Mean Teacher, confidence-based consistency methods and
domain-adaptive diagnosis, but it is not a reproduction of the complete
Calibrated Adaptive Teacher method [R1, R3, R4]. Source training data remain
available throughout adaptation. Consequently, the implementation should be
called source-access unsupervised domain adaptation, not source-free adaptation.
The source-free study in the thesis bibliography motivates an adjacent but
stricter information setting [R2].

CWRU and Paderborn are separate experiments. The code does not train on CWRU and
then claim direct transfer to Paderborn. CWRU supplies an initial four-class pilot;
Paderborn supplies the more demanding bearing-disjoint binary study. Their
metrics answer different questions and must not be combined into a single
dataset-independent accuracy claim.

## 2. Background and related work

### 2.1 Signals, labels and domains

A vibration recording is an ordered numerical sequence. For learning, the
implementation divides that sequence into windows, each containing a fixed
number of samples. A model maps each window to class scores. Windowing allows a
fixed-input network to process recordings of different lengths, but it does not
create new independent physical bearings. Adjacent windows and repeated
measurements of the same specimen share characteristics. Evaluation must account
for this structure rather than treating every window as an unrelated experiment.

A label is a reference statement about condition, such as healthy or faulty.
It is not contained in the vibration amplitude itself. In a benchmark, filenames
and metadata may encode the bearing identity from which a label can be inferred.
For this reason, removing an explicit label array is necessary but not sufficient
as an implementation discipline: identifiers must also be excluded from model
features and not decoded by the learning procedure.

Let the source dataset be a collection of pairs (x_s, y_s), and let the target
adaptation pool contain x_t without y_t. Source and target distributions may
differ even when both contain the same class set. Unsupervised domain adaptation
uses labeled source and unlabeled target examples to address this difference.
A domain-adversarial approach, for example, learns representations intended to
make the domains less distinguishable while retaining source task information
[R5]. The present prototype instead studies a teacher-generated target objective;
it has no domain discriminator or adversarial alignment loss.

### 2.2 Teacher–student learning

Mean Teacher forms a teacher by exponentially averaging student model weights
and uses teacher predictions as consistency targets [R3]. The teacher and student
are computational models, not separate datasets or human experts. Their
architectures may be identical while their weights diverge during training. The
teacher's slower update is intended to stabilize the target provided to the
student. Weight averaging alone does not make a teacher's prediction correct.

FixMatch combines a high-confidence pseudo-label from a weakly augmented input
with training on a more strongly augmented version of the same input [R4]. The
relevant idea for this project is the relationship between confidence filtering
and cross-view agreement. The thesis does not reproduce the image augmentations
or benchmark results of FixMatch. Its waveform perturbations are explicitly
defined engineering choices that still require application-specific validation.

A pseudo-label is a prediction used as if it were a learning target. It can be
correct or incorrect. If a false prediction is both confident and repeatedly
reinforced, the model can become more consistent without becoming more accurate.
Teacher–student disagreement is therefore a diagnostic of the training dynamics,
not an independent measurement of truth. Correctness can only be assessed
retrospectively where an appropriate reference label exists.

### 2.3 Confidence calibration and fault diagnosis

Calibration concerns whether model confidence corresponds to observed
correctness. Guo and colleagues examine neural-network calibration and evaluate
post-processing methods including temperature scaling [R6]. For this thesis,
the practical consequence is that a softmax value of 0.9 cannot automatically be
read as a verified 90% chance of a correct diagnosis on a new machine.

Forest and Fink's Calibrated Adaptive Teacher addresses confidence quality during
self-training for domain-adaptive intelligent fault diagnosis. The paper
investigates calibration of teacher predictions, and its official repository
provides implementations including a Paderborn configuration [R1, R7]. This is a
close reference because confidence-based pseudo-label selection is also central
to the present prototype. Nevertheless, the architecture, split, losses and
selection rules in this thesis are different. Comparing isolated headline scores
against that paper would not establish a reproduction or a methodological gain.

The local official-code check verified a CNN forward pass, backward pass and
optimizer step on prepared CWRU inputs. It did not reproduce complete CAT training
or its published study. The pinned reference commit and compatibility evidence
are retained in the repository. This distinction avoids confusing an engineering
compatibility check with experimental replication.

### 2.4 Source-free adaptation and the selected boundary

The source-free fault-diagnosis study by Yu and colleagues considers adaptation
when source data are not available [R2]. The bibliographic record and publisher
abstract support its relevance; a full implementation-level replication has not
been performed here. The present work retains source supervision, so it cannot
claim to solve that stricter problem. A future source-free variant would need a
different training interface and new controls, rather than merely a renamed
configuration.

This boundary is useful for practical planning. The current experiment asks
whether target information helps when source examples can still stabilize
learning. If that version struggles, removing source examples adds another
challenge. Conversely, a successful source-access result would not by itself
prove that a source-free deployment is viable.

### 2.5 Position of the present study

The contribution is an implementation and controlled experimental examination,
not a new claim of state-of-the-art accuracy. The literature motivates EMA,
pseudo-labels, consistency and calibration concerns; the repository establishes
what was actually executed. The central methodological choice is to compare
against both additional source training and source-only EMA. This makes the
question more specific: does the target objective contribute beyond two simpler
explanations for improvement?

The remainder of this draft describes the data boundary, the implemented
algorithm, the verification evidence and the measured answer. Where the evidence
does not support a conclusion—such as general industrial reliability or a
validated calibration correction—the limitation is retained explicitly.

## 3. Datasets and experimental protocol

### 3.1 CWRU pilot

The CWRU Bearing Data Center provides laboratory bearing vibration measurements
and descriptions of the operating and fault conditions [R8]. The local pilot
uses 40 original MATLAB files: 36 drive-end fault recordings covering three fault
families, three fault sizes and four loads, together with four normal recordings.
Only the outer-race 6-o'clock position is selected. The collection used in this
experiment is therefore not every file displayed on the 48 kHz download page.
Downloading more files and including them in a valid experimental protocol are
different decisions.

The pilot predicts four classes: healthy, inner-race fault, rolling-element fault
and outer-race fault. Source load is 0 hp; targets are 1, 2 and 3 hp. Raw recordings
are divided into 60/20/20 temporal regions with guard gaps of 4096 original
samples. Filtering and resampling are performed separately within regions so
that preprocessing cannot carry samples across a split boundary. Nominal
48 kHz data are reduced to 12 kHz, edge regions are trimmed, and nonoverlapping
1024-sample windows are standardized independently.

This pilot provides a functioning source/target workflow, but its temporal split
is weaker than a specimen-disjoint evaluation. Different segments of the same
recording share acquisition and bearing characteristics. The normal recording
sample rate is also an explicitly recorded assumption because the selected file
metadata do not fully resolve it. These limitations constrain the interpretation
of the pilot and motivate the separate Paderborn study.

### 3.2 Paderborn collection and quality audit

The Paderborn Bearing DataCenter and its benchmark paper describe measurements
from healthy and damaged bearings under different operating conditions [R9].
The imported collection contains 32 bearing archives and 2560 MATLAB recordings,
with four conditions and 20 repetitions per bearing/condition. The project
retains the original archives and checks their integrity and extracted file
inventory. Dataset use is attributed to Christian Lessmeier and colleagues,
KAt-DataCenter, Chair of Design and Drive Technology, Paderborn University,
under the publisher's stated noncommercial attribution terms.

The model uses the vibration channel identified as `vibration_1`, not motor
current or a blindly selected first numerical array. The loader checks the
associated time-axis index, finite values, matching lengths, an increasing time
axis and a nonconstant signal. Preparation attempts the full 2560-file audit.
Of these, 2559 recordings are readable and pass the implemented checks. The file
`KA08/N15_M01_F10_KA08_2.mat` cannot be read by the installed SciPy reader and
raises `TypeError: Expecting matrix here`. The file belongs to a bearing/condition
combination outside all selected learning/evaluation partitions. Its exclusion
therefore does not remove a selected example. The passing archive CRC does not
justify calling the entire archive corrupt, and no silent repair is applied.

The nominal vibration sample rate is 64 kHz. Preparation also reveals that
embedded HostService timestamps and sample counts are not always consistent with
an exact 64 kHz grid over four seconds. The experiment retains the documented
nominal rate and records the discrepancy rather than silently substituting a
new physical sample rate. Frequency-specific physical interpretations need
further acquisition-metadata investigation. A successful numerical loader check
does not resolve that measurement issue.

### 3.3 Binary labels and independent groups

Paderborn is treated as a binary task. K001–K006 are healthy and the remaining
26 bearings are faulty, including bearings with combined inner/outer damage.
No rolling-element-only class is fabricated to force agreement with the CWRU
label set. The class mapping follows the benchmark metadata and is checked in
unit tests. Binary fault detection is the principal task matching the thesis
objective; fault-family classification in CWRU is an additional pilot task.

Every physical bearing belongs to exactly one group. All its recordings and
windows remain in that group. The fixed assignment is:

| Group | Bearing identities | Conditions used |
|---|---|---|
| Source fit | K001, K002, K003, KA01, KA03, KA04, KA05, KA06, KA07, KI01, KI03, KI04, KI05, KI07, KB23 | Source only |
| Source validation | K004, KA08, KA15, KI08, KI14 | Source only |
| Target adaptation | K005, KA09, KA16, KI16, KI17, KB24 | Each target separately |
| Evaluation | K006, KA22, KA30, KI18, KI21, KB27 | Source and all targets |

This assignment separates adaptation bearings from evaluation bearings. It tests
more than fitting unseen windows from the same target specimen. However, the
evaluation group has only one healthy bearing. Twenty recordings and hundreds
of its windows do not provide the diversity of twenty independent healthy
bearings. All reported false-positive behavior must be interpreted with that
restriction.

### 3.4 Operating conditions

The source condition is N15_M07_F10. Targets change one operating variable at a
time relative to that source:

| Code | Speed (rpm) | Torque (Nm) | Radial force (N) | Role |
|---|---:|---:|---:|---|
| N15_M07_F10 | 1500 | 0.7 | 1000 | Source |
| N09_M07_F10 | 900 | 0.7 | 1000 | Reduced-speed target |
| N15_M01_F10 | 1500 | 0.1 | 1000 | Reduced-torque target |
| N15_M07_F04 | 1500 | 0.7 | 400 | Reduced-force target |

The same six evaluation bearings appear under all four conditions. This supports
descriptive condition comparisons, but it also makes the conditions correlated.
They are not independent draws from different factories or machines. The study
examines operating-condition shift together with a fixed specimen separation
within one benchmark setup.

The selected partitions contain 300 source-fit recordings, 100 source-validation
recordings, 360 target-adaptation recordings and 480 evaluation recordings:
1240 recordings in total. The remaining collection members are retained and
audited but not inserted into training merely because they are available.
Changing that selection would define a new experiment.

### 3.5 Signal processing

Each selected Paderborn signal is antialias-filtered and resampled by a factor of
one quarter, from nominal 64 kHz to nominal 16 kHz. The first and last 128
resampled values are removed. The remainder is divided into 2048-sample windows
with stride 2048. This gives a nominal duration of 0.128 seconds per window and
typically 31 windows per recording. Signal-length variation is reflected in the
actual stored window counts rather than hidden by assuming a constant count.

For window x, normalization subtracts its own mean and divides by its own standard
deviation, with a small lower bound in the denominator. It requires no labels or
fitted statistics from other recordings. It also suppresses absolute-amplitude
information, which may contain useful condition evidence. This is an explicit
representation choice, not a universal statement that amplitude is irrelevant.
A future ablation could compare alternative normalization rules on a separately
declared development protocol.

The resulting selected Paderborn dataset contains 38,522 windows. Source fit has
9322 windows and source validation 3104. Each target adaptation set contains
approximately 3700 windows. The exact counts, original file hashes and partition
memberships are saved in the preparation manifest. No overlapping windows are
created to artificially enlarge the sample count.

### 3.6 Label access and model-selection boundary

Prepared source files contain inputs, labels and window IDs. Target-adaptation
and evaluation input files contain only inputs and IDs. Their reference labels
are stored in a separate evaluator directory. The trainer only receives the
target tensor, never its reference labels or filename-derived class features.
Checksums verify that the loaded arrays belong to the declared preparation.

A Python audit hook blocks access to evaluator labels, evaluation inputs, raw
signals, curator label manifests, published predictions and report reads during
commissioning. For source-control runs, all target inputs are additionally
blocked. This provides protection against accidental access in this program; it
is not represented as a general isolation mechanism against arbitrary malicious
code. The documented access rules and code interface remain essential.

The baseline selects a checkpoint using source-validation recording macro F1.
Commissioning uses a fixed final epoch. Its source-validation histories are
diagnostic and cannot select a different adapted checkpoint. The decision
threshold is fixed at 0.5 before adaptation evaluation. No target-label-based
threshold search is performed.

The Paderborn baseline test results were already inspected before the current
extension was designed. The commissioning study is consequently described as
exploratory. A local timestamped declaration freezes its settings before its own
training, but this does not erase prior knowledge of the benchmark. Strong final
claims would require a defensible additional confirmation boundary.

## 4. Implemented learning method

### 4.1 Network and source initialization

The prototype uses a compact one-dimensional convolutional network. It accepts a
single-channel window with shape (1, window_length). The first convolution has
16 output channels, kernel length 31 and stride 4, followed by GroupNorm, ReLU and
max pooling. The second and third convolutions use 32 and 64 channels, kernel
lengths 9 and 5, and stride 2. Each is followed by GroupNorm and ReLU. Adaptive
average pooling reduces the time dimension before a linear classifier. The
Paderborn two-class model has 15,810 trainable parameters. The default four-class
head remains available for the frozen CWRU experiment.

GroupNorm avoids running normalization statistics that would otherwise depend on
the sequence of source and target batches. The model still learns from domain
specific signals through its weights; this choice does not make it inherently
domain invariant. The small architecture was selected for an inspectable,
computationally inexpensive prototype rather than asserted to be optimal.

The Paderborn source model is trained for 20 epochs with Adam, learning rate
0.001, weight decay 0.00001 and batch size 128. A weighted sampler balances source
classes in expectation using source labels. For seeds 7, 17 and 27, the selected
source epochs are respectively 1, 1 and 2. Selection maximizes source-validation
recording macro F1, retaining the earliest epoch in a tie. Later low training
loss does not override poor held-out source validation.

### 4.2 Commissioning objective

Let the student's parameters be theta and the teacher's parameters be phi. For
source examples with reference labels, the supervised objective is the mean
cross entropy:

$$
L_s = -\frac{1}{B_s}\sum_{i=1}^{B_s}\log p_\theta(y_i^s\mid x_i^s).
$$

For an unlabeled target window, two perturbed views are constructed. The weak
view adds Gaussian noise with standard deviation 0.01. The stronger view
multiplies the standardized input by a gain sampled uniformly from 0.9 to 1.1
and adds Gaussian noise with standard deviation 0.05. These noise magnitudes are
relative to the independently standardized window, not physical acceleration
units. The gain and noise assumptions require further validation as
class-preserving perturbations.

The teacher distribution is q_j = softmax(f_phi(a_w(x_j))). A hard pseudo-label
is its largest-probability class, and confidence is that largest probability.
The acceptance mask m_j equals one when confidence is at least tau = 0.9 and zero
otherwise. The target objective is:

$$
\hat y_j = \arg\max_c q_{j,c}, \qquad
m_j = \mathbf{1}\{\max_c q_{j,c}\geq 0.9\},
$$

$$
L_t = -\frac{1}{B_t}\sum_{j=1}^{B_t}
m_j\log p_\theta(\hat y_j\mid a_s(x_j)).
$$

The denominator includes every target window in the batch, including rejected
windows. Consequently, low acceptance reduces the magnitude of the target
contribution. If nothing is accepted, the target loss is finite zero and supplies
no target gradient. Rejected examples are not assigned invented healthy labels.
This behavior is verified with explicit gradient tests.

The full objective at adaptation epoch e is:

$$
L(e) = L_s + \lambda(e)L_t, \qquad
\lambda(e)=0.5\frac{e}{10}, \quad e\in\{1,\ldots,10\}.
$$

This hard pseudo-label objective enforces agreement between teacher and student
across the two views. There is no additional MSE or KL consistency term in this
version. Describing the method as consistency learning therefore refers to this
specific masked cross-entropy mechanism, not to an unspecified extra loss.

### 4.3 Teacher update and adaptation schedule

Teacher and student begin as independent copies of the same selected source
checkpoint. The teacher does not receive gradients. After each student optimizer
update, the teacher is updated as:

$$
\phi_{k+1}=0.99\phi_k+0.01\theta_{k+1}.
$$

The update uses the newly optimized student, not its pre-update state. Teacher
buffers are copied from the student. Independent storage is essential: if the
two names referred to the same tensors, the intended averaging mechanism would
not exist. A test changes student parameters and verifies both independence and
the expected subsequent EMA value.

Every commissioning run resets Adam and uses learning rate 0.0002, weight decay
0.00001, batch size 128 and 10 epochs. An epoch is defined by one traversal of
the class-balanced source loader. It contains 73 updates, giving 730 updates per
run. Each source update is paired with one shuffled target batch. The target
iterator restarts when exhausted, so a target window may be used multiple times
within a source epoch. Last short batches are retained. Statistics counted over
these uses are described as exposures, not independent measurements.

Target sampling and augmentation use random generators independent of source
sampling. The exact sequence of sampled source indices is hashed. For each seed,
all three target adaptations and the source control must have the same source
sequence hash and update count. Thus, a target effect cannot be explained simply
by giving that method a different set of source minibatches.

The final epoch teacher is the reported adapted model. The final student is
preserved for diagnostics, but it is not substituted when its target score looks
better. All thresholds, epochs and loss weights are inherited from the earlier
CWRU pilot rather than searched using Paderborn target test labels. The source
validation score is logged but has no branch that changes adaptation selection.

### 4.4 Control methods

Four methods address distinct explanations for performance:

| Method | Additional source updates | Unlabeled target objective | Reported weights |
|---|---|---|---|
| Initial source-only | None | None | Previously selected source checkpoint |
| Continued source | 730 | None | Final supervised student |
| Source EMA | Same 730-update trajectory | None | EMA of supervised student |
| Teacher–student | 730, with matched source batches | Masked target consistency | Final target-adapted EMA teacher |

The source control is trained once per seed and supplies both the final student
and its EMA. These weights are evaluated on all targets. Repeating their scores
in different target columns does not create additional independent training
runs. Each target-adapted teacher starts from the original source checkpoint,
not from the model adapted to the previous condition.

The primary comparison is teacher–student against source EMA. It controls for
both additional source optimization and weight averaging. It does not isolate
confidence filtering, perturbations and target loss weighting individually.
Those component effects require separate ablations. The present study is an
initial controlled comparison of the complete target-learning contribution.

### 4.5 Prediction and evaluation units

At inference, a recording passes through the same preprocessing as training.
The model produces two softmax scores for every window. Recording probabilities
are arithmetic means over its windows. A mean faulty score at least 0.5 produces
the decision faulty; otherwise the decision is healthy. The predictor reports
the model role, adaptation target, input hash, checkpoint hash, window count,
quality metadata and scores.

Recording-level metrics are primary because the practical input is a recording
and adjacent windows are dependent. Window-level results remain available for
inspection. Per-bearing results further expose cases where a method succeeds on
some specimens but systematically misses others. The prototype has no
unknown-fault rejection mechanism and its scores are not calibrated industrial
failure probabilities.

For binary classification, fault recall is TP/(TP+FN) and healthy false-positive
rate is FP/(TN+FP). Balanced accuracy is the mean of healthy and faulty recalls.
Macro F1 gives equal weight to each class's F1. This prevents a large faulty
class from dominating the headline as easily as ordinary accuracy.

Each target evaluation contains 20 healthy and 100 faulty recordings. Predicting
faulty for every recording would achieve 83.33% accuracy while giving only 50%
balanced accuracy, 45.45% macro F1 and 100% healthy false-positive rate. This
example explains why an apparently high accuracy alone is inadequate. ROC AUC
and average precision are also retained as ranking measures, but a favorable
ranking score cannot replace the actual decisions at the fixed threshold.

## 5. Prototype implementation and verification

### 5.1 Modules and evidence flow

The project separates data processing, learning, evaluation and reporting. Raw
archives and MATLAB files are retained outside Git. Preparation produces
checksummed numerical inputs and manifests. Training reads the permitted inputs,
creates checkpoints and freezes their hashes. Evaluation reads reference labels
only after that freeze. The analysis stage then relates recorded pseudo-label
decisions to reference truth and generates figures.

The original baseline modules remain byte-for-byte unchanged because their hashes
are part of the existing experimental record. Commissioning is implemented in
new modules: `thesislib/commissioning.py` contains the learning procedure and
access guard; `scripts/train_paderborn_commissioning.py` handles execution and
freeze records; separate evaluation and analysis scripts handle target labels.
The command-line predictor accepts both baseline and supported adapted models.
Diagnostic-only student checkpoints are not accepted as if they were the declared
deployed teacher.

The training declaration records the configuration, protocol text, implementation,
preparation summary, baseline completion evidence, software versions and GPU.
Completion adds model, history and pseudo-trace hashes. Evaluators reject
changed evidence. A completed or even started training directory is not silently
overwritten, which prevents a later rerun from replacing an unfavorable outcome
under the same experiment name.

### 5.2 Recorded training diagnostics

Each epoch records source loss, full-batch target loss, target weight, acceptance,
confidence, accepted labels by predicted class, cross-view disagreement and
source-validation scores for both networks. Each actual target exposure also
retains its input index, epoch, teacher pseudo-label, confidence, acceptance and
student prediction. The trace stores no raw waveform or target reference label.

After training, the analysis joins those indices to separately stored reference
labels. It checks that trace counts agree with online history counts before
computing accepted pseudo-label precision and coverage by true class. This
provides a direct audit of the decisions that supplied gradients, instead of
substituting predictions from a different post-training model. Repeated exposure
counts are retained and labeled explicitly.

### 5.3 Tests and repeatability

The complete regression suite contains 23 passing tests at this milestone. The
new tests exercise restricted target access, rejection of target archives that
contain labels, source-order independence from augmentation randomness, teacher
independence, masked gradients and empty acceptance. A synthetic zero-target-weight
experiment reproduces both networks of the source control exactly, demonstrating
that merely constructing target views does not alter the source sampling path.
Additional tests cover the misleading all-faulty classifier and retrospective
trace accounting. Existing tests retain coverage of split separation, binary
label mapping, real-signal preprocessing and recording aggregation.

The seed-7 reduced-speed experiment and its source control were then retrained
with the same restricted label access. All four saved checkpoint state dictionaries
matched exactly, with maximum absolute tensor difference zero. Training histories
and pseudo-label traces also matched. This is evidence of repeatability on the
same installed software, data and GPU; it does not promise identical results on
every PyTorch release or hardware platform.

### 5.4 Executable demonstration

The adapted seed-7 reduced-speed teacher was used on two real Paderborn-format
recordings. The K006 example was classified healthy with a mean faulty score
approximately 0.0049; the KA22 example was classified faulty with score
approximately 0.8903. Both produced 31 windows and executed on CUDA. These
examples verify the inference path from a MATLAB recording to a saved report.
They belong to the existing evaluation collection and are therefore not new
independent validation cases.

The experiment is reproducible through documented commands and small published
artifacts. Large raw datasets, prepared signal arrays and the Python environment
are excluded from GitHub. A restoration command verifies the published manifest
and restores small runtime artifacts without overwriting different local files.
This keeps the repository reviewable while preserving dataset provenance and
the distinction between code access and dataset licensing.

## 6. Experimental results

### 6.1 CWRU pilot outcome

Across three target loads and three seeds, the existing four-class CWRU pilot
obtains mean macro F1 of 84.65% for initial source-only, 87.20% for continued
source, 86.11% for source EMA and 86.07% for teacher–student. The teacher improves
over the initial model but not the matched controls. In eight of nine comparisons
with source EMA, the final class predictions are identical. The small remaining
difference gives approximately -0.045 percentage points in overall macro F1.

This pilot does not establish added value from unlabeled target learning. Its
binary healthy/faulty separation is already saturated for the initial source
and teacher models, making that subset a limited test of improved fault detection.
The four-class difficulty largely concerns distinguishing fault families. The
source EMA control was introduced after inspection of the first pilot results;
its timing is documented rather than presented as part of an untouched initial
design. Paderborn includes all controls in the adaptation declaration from the
beginning.

### 6.2 Paderborn comparison across conditions

Table 6.1 reports mean recording macro F1 and sample standard deviation over
three seeds. The same evaluation group is used for every method. These are
binary results and should not be numerically ranked against four-class CWRU
values as though the tasks were identical.

| Target condition | Initial source | Continued source | Source EMA | Teacher–student |
|---|---:|---:|---:|---:|
| Reduced speed | 48.19 ± 30.00 | 61.19 ± 12.35 | 64.58 ± 8.66 | 64.99 ± 12.54 |
| Reduced torque | 89.68 ± 16.62 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Reduced radial force | 31.14 ± 15.57 | 35.07 ± 10.35 | 35.94 ± 12.49 | 34.15 ± 8.21 |

Table 6.1. Recording macro F1 (%), mean ± sample SD across seeds 7, 17 and 27.

![Paderborn comparison with every seed shown](../results/paderborn_commissioning_v1/figures/comparison_macro_f1.png)

Figure 6.1. Four methods evaluated on the same held-out recordings. Bars are
means and dots show all seeds; they are not independent-bearing confidence
intervals.

The mean across the nine target/seed combinations is 56.34% for the initial source
model, 65.42% for continued source, 66.84% for source EMA and 66.38% for the
teacher–student method. Thus, comparison only with the initial model would show
a 10.04 percentage-point gain and suggest useful adaptation. The matched EMA
comparison changes the interpretation: target learning is 0.46 percentage points
lower on average. Much of the improvement can be obtained without target inputs.

The condition-specific pattern matters. Reduced torque is solved perfectly on
this finite held-out set by all three methods with additional source updates.
There is therefore no visible target-learning advantage at the fixed classification
metric in that condition. Reduced speed shows a small positive mean difference
against EMA, but the direction varies by seed. Reduced radial force remains
poor for all methods and is worse on average for the target-adapted teacher than
for source EMA.

### 6.3 Paired effects and missed faults

Against source EMA, teacher–student improves three task/seed pairs, worsens three
and ties three. Reduced-speed differences are +3.25, +2.18 and -4.17 percentage
points for seeds 7, 17 and 27. Reduced-force differences are +6.57, -0.74 and
-11.22 percentage points. All reduced-torque pairs tie. These descriptive
differences show why selecting the most favorable seed would misrepresent the
study.

![Paired teacher versus source EMA effects](../results/paderborn_commissioning_v1/figures/paired_deltas.png)

Figure 6.2. Every paired comparison. Values left of zero indicate harm relative
to source EMA under the primary metric. The pairs share models and bearings
across conditions and should not be treated as nine independent population draws.

For reduced speed, teacher fault recall is 63.00 ± 17.35%. For reduced torque it
is 100%. For reduced radial force it is only 21.33 ± 10.21%, compared with
24.00 ± 14.80% for source EMA. Across three seeds the teacher misses 236 of the
300 repeated faulty recording predictions at reduced force. This count refers
to 100 unique faulty recordings evaluated three times, not 300 independent
faulty recordings.

Healthy false-positive rate is zero on all three targets for all four methods.
That favorable observation cannot offset the low fault recall, nor establish
robust specificity on a wider healthy population: only K006 represents healthy
test bearings. A model can avoid false alarms by labeling many recordings
healthy while still missing dangerous conditions.

Per-bearing results localize the reduced-force failure. KA22, KB27 and KI18
have zero mean fault recall for every compared method. Some detection remains
for KA30 and KI21. The difficulty is therefore not adequately described by a
single average score; several physical specimens are systematically missed.

![Fault recall for each held-out damaged bearing](../results/paderborn_commissioning_v1/figures/per_bearing_fault_recall.png)

Figure 6.3. Mean fault recall across seeds for each damaged test bearing.
Each bearing contributes 20 recordings per condition and seed.

### 6.4 Ranking scores and decision scores

The target-adapted teachers attain recording ROC AUC of 1.0 in all nine target
evaluations. This coexists with poor fixed-threshold recall under reduced radial
force. The distinction is mathematical: AUC measures the ranking of positive
and negative scores, whereas the reported class decision compares the mean
faulty score with 0.5. Good ranking does not ensure that this particular threshold
separates the classes correctly.

This observation motivates investigation of score calibration or a justified
label-free decision policy. It does not authorize fitting a threshold to the
already inspected target labels and presenting the resulting score as label-free
commissioning performance. Nor does it establish that a particular calibration
technique would solve the problem. For example, positive temperature scaling
preserves each window's logit ordering, so its effect on pseudo-label acceptance
and aggregated recording decisions must be distinguished carefully.

### 6.5 What the pseudo-label traces reveal

The final-epoch trace provides evidence about the learning targets actually used
by the student. In reduced speed, acceptance is 62.22–78.76%, with accepted
pseudo-label precision from 70.95% to 92.84%. The weaker precision for seed 27 is
consistent with its less favorable target result, but this association alone
does not prove a complete causal explanation.

Reduced force is much more problematic. The teacher accepts 75.83–83.73% of
exposures, yet only 20.29–28.15% of those accepted labels are correct. Mean
confidence within the accepted subset is approximately 98.50–99.16%. The system
is therefore highly confident on a largely incorrect training subset. Between
92.54% and 99.14% of accepted pseudo-labels are healthy even though the adaptation
pool contains mostly faulty recordings. Reference labels are used here only
for retrospective diagnosis of the failure.

![Confidence filtering and pseudo-label correctness](../results/paderborn_commissioning_v1/figures/pseudo_label_diagnostics.png)

Figure 6.4. Training exposures counted by epoch. The middle column requires
reference labels and is computed after all training checkpoints are frozen.

This is evidence of a harmful concentration toward healthy pseudo-labels under
reduced force, with many true faults accepted as healthy. It is more informative
than merely observing that confidence or acceptance increases. The student is
being encouraged to reproduce errors, and the EMA teacher inherits the student's
trajectory. The implemented consistency objective has no external information
that guarantees recovery from this condition.

Reduced torque illustrates another complication. Test classification is perfect
for the adapted teachers, but accepted pseudo-label precision on the separate
adaptation bearings is only 66.10–85.40% in the final epoch. In seed 17 all
accepted pseudo-labels are faulty. This does not contradict the held-out result:
adaptation and evaluation contain different bearing identities. It demonstrates
that one aggregate score does not describe all specimens and partitions. It
also shows why assumptions about the target class distribution cannot be inferred
from confident model predictions alone.

### 6.6 Consistency, source validation and retention

Teacher/student disagreement is low by the end of training, including in the
reduced-force runs with incorrect pseudo-labels. Agreement has therefore been
achieved without reliable target diagnosis. The source-validation teacher macro
F1 remains near 44.44% for most trajectories. These histories are recorded as
diagnostics and are not used to select an alternative adapted checkpoint after
seeing target results.

On held-out bearings at the source operating condition, continued source and
source EMA both achieve about 99.51% mean recording macro F1. The adapted
teachers achieve approximately 99.51–100%, depending on their adaptation target.
No reduction in this source-condition metric is observed relative to the matched
control. However, the gap between source validation and source-condition test
performance reveals sensitivity to the chosen bearing groups. It is not a
reason to replace the validation rule with a favorable test score.

## 7. Discussion and limitations

### 7.1 Answer to the research questions

The first research question is answered by the initial Paderborn baseline:
generalization varies substantially by condition and seed. Reduced force is
particularly difficult, and unseen-bearing performance cannot be inferred from
source training loss. The second question receives a qualified negative answer
for the implemented fixed protocol. Target adaptation improves over the initial
model but does not outperform matched source EMA on the mean primary metric.
There are individual gains and losses, rather than consistent superiority.

The third question is answered by the recorded training traces. Confidence-based
acceptance can admit many incorrect labels; consistency can increase while those
errors are reinforced. Retrospective class-specific analysis is necessary to
understand this failure. The fourth question is answered by the working software,
label-access controls, 23 tests, frozen evidence and exact local replay. The
engineering pipeline is operational even though the tested learning procedure
is not yet a reliable general solution for condition shift.

The CWRU and Paderborn experiments both emphasize matched controls. A comparison
against only the initial source model would overstate the evidence for target
learning. In Paderborn, extra supervised source training and averaging already
account for most of the observed aggregate improvement. This finding is a
contribution to the experimental assessment, not an implementation defect that
should be hidden by choosing a different headline.

### 7.2 Limits of the evidence

The principal limitation is the small number of independent bearing specimens.
All Paderborn conditions reuse the same evaluation group, and only one healthy
bearing is held out. Sample standard deviations across three seeds describe
optimization variability on that fixed group. They are not confidence intervals
over machines, factories, fault severities or healthy specimens. Additional
windows do not remedy this limitation.

The experiment is also exploratory because baseline test results were known
before the extension. Its parameters were fixed before adaptation and no target
selection was performed, but the existing knowledge must still be disclosed.
A new split involving previously unused data is not automatically untouched if
its design is chosen to produce favorable results. A meaningful confirmation
study needs a documented decision boundary and appropriate supervision.

The method includes several interacting choices: normalization, waveform
perturbations, confidence threshold, EMA coefficient and target weight. Only the
combined mechanism is compared with the declared controls. The present data
cannot identify the individual causal contribution of each component. In
particular, the observed pseudo-label errors motivate calibration research but
do not demonstrate that calibration alone resolves them.

The measurements are laboratory benchmarks. The prototype does not validate
sensor installation, transmission losses, continuous monitoring, unknown fault
types, maintenance decisions or real-machine commissioning. Nominal sample-rate
uncertainty remains documented, and the full CAT reference algorithm has not
been reproduced. These limits should remain visible when the system is shown
in a presentation.

### 7.3 Justified next experiments

The next experimental stage should begin with a documented development and
confirmation plan. Candidate ablations include removing the target loss,
changing weak/strong perturbations, varying a confidence policy, and comparing
explicit source-derived calibration. The current source controls and the
synthetic zero-weight test provide useful foundations but do not replace real
data ablations of every component.

Any threshold or calibration parameter that needs labels must use an allowed
source/development partition. Target reference labels from the current report
must not silently become tuning data while the same dataset is called an
untouched test. A source-free extension would additionally require removing
source minibatches and designing controls for that information setting.

More independent healthy bearings and a larger range of damaged specimens would
improve the evaluation more directly than simply increasing the sampling rate
or generating overlapping windows. Full CAT replication would provide a stronger
method reference if its data split and model-selection rules are carefully
reconciled with this thesis's stricter bearing separation. Final choices should
follow the supervisor's expected scope for a BSc project.

## 8. Conclusions

This work implements label-free target commissioning as a reproducible
source-access teacher–student procedure for bearing vibration signals. The
prototype connects dataset preparation, supervised learning, unlabeled target
adaptation, frozen evaluation and recording-level inference. It preserves
original evidence and supports inspection of both model outcomes and the
pseudo-label decisions that produced them.

The present experiments do not demonstrate that the custom target-learning
procedure consistently improves on matched source-only EMA. In Paderborn,
the difference is -0.46 percentage points in mean recording macro F1, while
individual task/seed pairs show both benefit and harm. The reduced-force
condition exposes confident but mostly incorrect pseudo-labels and substantial
missed-fault rates. These findings answer the thesis question more precisely
than reporting improvement over an initial source baseline alone.

The implemented system and measured study form a substantial intermediate
engineering and research milestone. Final submission still requires student
ownership of the explanation, supervisor review, stronger confirmation where
feasible, completion of selected further experiments and institutional document
formatting. No numerical percentage of overall thesis completion is asserted.

## References

References R1–R7 support methodological background; numerical results in Chapters
5–7 come from this repository's frozen experiments, not from those papers.
Publisher/author sources were checked during preparation of this draft. R2 is
used at the bibliographic/abstract level; its full method was not independently
reproduced. No unverifiable performance figures from that paper are quoted.

- **[R1]** F. Forest and O. Fink, “Calibrated Adaptive Teacher for Domain-Adaptive
  Intelligent Fault Diagnosis,” *Sensors*, 24(23), 7539, 2024.
  [DOI](https://doi.org/10.3390/s24237539);
  [author preprint, revised 2024](https://arxiv.org/abs/2312.02826).
- **[R2]** A. Yu, B. Cai, Q. Wu, M. M. García, J. Li and X. Chen,
  “Source-free domain adaptation method for fault diagnosis of rotation machinery
  under partial information,” *Reliability Engineering & System Safety*, 248,
  110181, 2024. [Publisher record](https://www.sciencedirect.com/science/article/pii/S0951832024002552);
  [DOI](https://doi.org/10.1016/j.ress.2024.110181).
- **[R3]** A. Tarvainen and H. Valpola, “Mean teachers are better role models:
  Weight-averaged consistency targets improve semi-supervised deep learning
  results,” *Advances in Neural Information Processing Systems*, 2017.
  [Author paper](https://arxiv.org/abs/1703.01780).
- **[R4]** K. Sohn, D. Berthelot, C.-L. Li, Z. Zhang, N. Carlini, E. D. Cubuk,
  A. Kurakin, H. Zhang and C. Raffel, “FixMatch: Simplifying Semi-Supervised
  Learning with Consistency and Confidence,” *NeurIPS*, 2020.
  [Author paper](https://arxiv.org/abs/2001.07685).
- **[R5]** Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle,
  F. Laviolette, M. Marchand and V. Lempitsky, “Domain-Adversarial Training of
  Neural Networks,” *Journal of Machine Learning Research*, 17(59), 1–35, 2016.
  [Journal paper](https://jmlr.org/papers/v17/15-239.html).
- **[R6]** C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On Calibration of
  Modern Neural Networks,” *Proceedings of the 34th International Conference on
  Machine Learning*, PMLR 70, 1321–1330, 2017.
  [Publisher paper](https://proceedings.mlr.press/v70/guo17a.html).
- **[R7]** EPFL Intelligent Maintenance and Operations Systems, *CAT official
  code repository*, pinned locally to commit
  `354cf94aef4f0afd1f81d7ef645bd40ffc68d8d5`.
  [Repository](https://github.com/EPFL-IMOS/CAT).
- **[R8]** Case Western Reserve University, *Bearing Data Center*.
  [Dataset and experimental description](https://engineering.case.edu/bearingdatacenter/welcome);
  [download page](https://engineering.case.edu/bearingdatacenter/download-data-file).
- **[R9]** C. Lessmeier, J. K. Kimotho, D. Zimmer and W. Sextro,
  “Condition Monitoring of Bearing Damage in Electromechanical Drive Systems
  Using Motor Current Signals: A Benchmark Data Set for Data-Driven
  Classification,” *European Conference of the Prognostics and Health Management
  Society*, 2016.
  [Paderborn DataCenter and required attribution](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter);
  [official benchmark paper](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf).

## Appendix A. Reproduction and evidence map

Run commands from the repository root using the installed environment. Existing
completed experiments should be inspected or restored, not overwritten.

```powershell
.\.venv\Scripts\python.exe scripts\restore_published_results.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\evaluate_paderborn_commissioning.py
.\.venv\Scripts\python.exe scripts\analyze_paderborn_commissioning.py
```

Tests and full evaluation need the original and prepared data. Restoring published
artifacts alone does not create those datasets. On a fresh training workspace
with the frozen source baseline and prepared inputs, run
`scripts/train_paderborn_commissioning.py` before evaluation. The replay command
uses `--replay-seed 7 --replay-condition N09_M07_F10` and also refuses to overwrite
an existing replay.

| Evidence | Repository path |
|---|---|
| Fixed adaptation settings | `configs/paderborn_commissioning_v1.json` |
| Declared protocol | `reports/PADERBORN_COMMISSIONING_PROTOCOL_V1.md` |
| Automatically generated measured report | `reports/PADERBORN_COMMISSIONING_RESULTS.md` |
| All task/seed metrics | `results/paderborn_commissioning_v1/results_by_seed.csv` |
| Paired gains and losses | `results/paderborn_commissioning_v1/paired_comparisons.csv` |
| Bearing-specific outcomes | `results/paderborn_commissioning_v1/per_bearing_results.csv` |
| Offline pseudo-label audit | `results/paderborn_commissioning_v1/pseudo_labels_by_epoch.csv` |
| Training freeze and hashes | `results/paderborn_commissioning_v1/training_complete.json` |
| Exact replay evidence | `results/paderborn_commissioning_v1/reproducibility_seed7_N09_M07_F10.json` |
| Models and traces | `results/paderborn_commissioning_v1/seed_*/` |

## Appendix B. Plain-language explanation

Imagine learning to recognize a damaged bearing from examples recorded under
one operating condition. The examples say which bearing is healthy or faulty.
Now the condition changes and new recordings arrive without answers. The first
model makes provisional guesses about those recordings. A student model tries
to give the same answer after a small signal perturbation, while still learning
from the original labeled examples. A slowly updated teacher supplies the guesses.

The essential catch is that a confident guess can be wrong. If both models
repeat the same mistake, agreement improves but diagnosis does not. The thesis
therefore compares the method against simply training longer on the original
examples and averaging that model's weights. Only a gain beyond those controls
would support added value from the unlabeled target recordings. In the current
experiments, that gain is not consistently demonstrated.
