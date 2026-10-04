# CSBP711 A1: Occlusion Detection in CASIA-B

## Objective
This assignment investigates classification of artificially generated
occlusion types in gait silhouette sequences.

It is a partial replication of the occlusion-detection component of:
Hasan et al. (2024), "Improving Gait Recognition Through Occlusion
Detection and Silhouette Sequence Reconstruction."
DOI: 10.1109/ACCESS.2024.3482430

Silhouette reconstruction and person recognition are outside the
current experiment's scope.

## Research connection
This experiment is a first step toward studying gait recognition
under occlusion as a possible PhD research direction.

## Dataset
- Dataset: CASIA-B
- Downloaded version: 64 × 64 silhouette sequences
- Local location: data/casia-b/
- Download source:
  https://drive.google.com/file/d/1Vcnwy2yVtZ5KUBp7WMTEg1EN0H2PUfJ6/view
- Download date: 27 September 2026
- Original dataset source and research-use terms: TODO
- Actual subject, sequence, and frame counts: pending inspection

Dataset files are excluded from this repository.

## Planned experimental protocol
- Use the 90-degree camera view.
- Use subjects 001–074 for development.
- Reserve a subject-disjoint subset of development data for validation.
- Use subjects 075–124 only for final testing.
- Keep all frames and occlusion variants of a recording in the same split.
- Apply artificial masks after silhouette normalization.

Occlusion labels:
- L-0: bottom 25% masked
- L-1: bottom 50% masked
- L-2: left 50% masked
- L-3: right 50% masked
- L-4: no occlusion

## Planned algorithms
1. Dummy classifier: simple baseline
2. Logistic regression
3. RBF support vector machine
4. 3D CNN occlusion detector

## Evaluation
Report accuracy, macro-F1, training time, and saved model size.
Use validation data for model selection.
Document common preprocessing, random seeds, and deviations from the paper.

## Ablation
Planned: test sensitivity to changes in occlusion-mask boundaries.
Record the hypothesis and expected ranking changes before running it.

## Reproduction
Instructions will be added and verified as the scripts are completed.

## Results
No experiments have been completed yet.


## AI assistance
An OpenAI assistant helped interpret the assignment and paper and plan
the experiment.

## Data audit

Run from the repository root:

    ./venv/bin/python3 inspect_data.py

The script checks the 90-degree silhouette sequences and exports:
- data_manifest.csv: sequence metadata and proposed evaluation roles.
- data_summary.json: dataset counts and quality checks.
- data_errors.csv: unreadable or invalid sequences.
- missing_sequences.csv: missing expected recordings.
- duplicate_sequences.csv: sequences with identical content.

The original dataset is not modified.
Exact-content checks do not rule out every form of near-duplicate data.

### Initial audit findings

The 90-degree subset contains:
- 124 subjects and 1,240 readable sequences.
- 73,282 frames, with sequence lengths ranging from 1 to 86.
- 744 normal, 248 bag, and 248 clothing sequences.
- 740 development sequences, 200 gallery sequences, and 300 probes
  before filtering.

No missing expected recordings, invalid arrays, empty frames, or
exact duplicate sequences were detected.

Six sequences contain fewer than 16 frames. Their treatment and the
resulting evaluation counts will be documented before training.
Exact duplicate checks do not exclude all possible near-duplicates.

### Filtering and validation

Six sequences shorter than 16 frames were excluded:
subject 005's bg-01, bg-02, nm-02, nm-03, nm-04, and nm-06.
The original dataset files were preserved.

After filtering, 734 development sequences, 200 test-gallery
sequences, and 300 test probes remain.

For model selection, the 74 development subjects are subdivided
into 60 training and 14 validation identities using seed 42.
Validation uses clean nm-01–nm-04 as gallery references and the
remaining recordings as probes.

This validation subdivision is our addition to the paper's protocol.
All models will use the same saved split manifest.
Final training scope will be reported explicitly.

## Simple recognition baseline

The baseline averages each silhouette sequence into a gait energy
image (GEI), using fixed pixel scaling by 255. Each probe is matched
to the nearest individual gallery GEI by Euclidean distance.
The predicted identity is the identity of that gallery sequence.

Gallery recordings remain clean. The five fixed occlusion conditions
are applied to probes without recropping.

This is our simple baseline, not a verified reproduction of the
paper's DM implementation. It uses all available sequence frames.

During development, evaluation uses validation identities only.
The baseline requires no model training; feature extraction time,
matching time, and stored gallery size are recorded separately.

### Validation baseline results

GEI nearest-gallery matching achieved the following rank-1 accuracy:

| Occlusion | Normal | Bag | Clothing |
|---|---:|---:|---:|
| Clean | 100.00% | 67.86% | 28.57% |
| Bottom 25% | 50.00% | 42.86% | 14.29% |
| Bottom 50% | 21.43% | 25.00% | 7.14% |
| Left 50% | 7.14% | 7.14% | 7.14% |
| Right 50% | 7.14% | 7.14% | 7.14% |

These are development results on 14 validation identities, with
28 probes per walking condition. They are not final test results
and are not directly comparable to the paper's reported scores.

Distances were computed using SciPy's squared Euclidean distance
routine, with checks for non-finite features and distances.