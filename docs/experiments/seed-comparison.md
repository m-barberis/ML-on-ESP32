# Training-seed comparison: seven versus ten epochs

## Question and method

Do the seven- and ten-epoch training recipes differ in average accuracy and
variability across random seeds? The durations were fixed by the earlier
[seven-fold experiments](seven-fold-epochs.md); they were not retuned here.

Ten seeds, **42–51**, each train one fresh model for ten epochs on all 2,520
training images. Weights are captured at epochs 7 and 10. Both checkpoints are
evaluated only after that trajectory finishes, giving 20 paired evaluations on
the same 372-image development test set. There is no early stopping.

Architecture, enhanced augmentation, 25% dropout, Adam at 0.001, and batch size
32 remain fixed. Every INT8 export uses the same 200 calibration images, selected
from training data with calibration seed 42. This isolates training randomness
from calibration-subset randomness. Sample standard deviations use `ddof=1`.

## Results

| Epochs | Float mean accuracy | Float sample SD | INT8 mean accuracy | INT8 sample SD | INT8 range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 96.37% | 3.09 pp | **96.51%** | **2.88 pp** | 90.32–100.00% |
| 10 | 96.56% | 1.38 pp | **96.83%** | **1.47 pp** | 94.35–98.92% |

| Seed | INT8 at epoch 7 | INT8 at epoch 10 |
| --- | ---: | ---: |
| 42 | 100.00% | 97.31% |
| 43 | 98.66% | 95.97% |
| 44 | 96.24% | 96.51% |
| 45 | 90.32% | 94.35% |
| 46 | 95.16% | 97.04% |
| 47 | 99.19% | 95.43% |
| 48 | 96.51% | 98.92% |
| 49 | 95.43% | 95.97% |
| 50 | 98.92% | 98.12% |
| 51 | 94.62% | 98.66% |

Ten epochs improved INT8 accuracy in six pairs and worsened it in four. The mean
paired difference (10 minus 7) was **+0.32 percentage points**, with sample SD
**2.80 points**. The mean advantage is small relative to variation between pairs;
these results do not establish a reliable accuracy advantage. Ten epochs had a
lower observed spread and a higher minimum accuracy in these ten runs. These are
supporting observations; the selected duration follows from preferring validation
loss for the seven-fold epoch-selection procedure, rather than this test comparison.

These statistics describe training randomness on this fixed development set.
They are not confidence intervals for real-camera accuracy. No individual seed
was selected for its test score. The selected deployment model retains the original
seed 42 and the loss-selected ten-epoch duration; see the [model card](../model-card.md).
The previously proposed unused seed 52 was not used.

## Reproduction and verification

```sh
python src/training/seed_comparison.py
```

The command refuses existing run outputs; use `--run NAME` for a fresh report
directory. No per-seed model files are retained. Exports and checkpoint weights
are evaluated in memory. All INT8 exports are 36,184 bytes, use the same five
operations, and contain no float tensors.

Exactly three files are saved in `reports/metrics/seed_comparison/`:

- [results.json](../../reports/metrics/seed_comparison/results.json): configuration,
  dataset hashes, calibration indices, per-checkpoint metrics, and paired summaries.
- [history.csv](../../reports/metrics/seed_comparison/history.csv): 100 training epochs.
- [predictions.csv](../../reports/metrics/seed_comparison/predictions.csv): 7,440
  predictions, including float probabilities for independent inspection.

Verification reproduced all reported accuracies, confusion matrices, means,
sample standard deviations, and paired differences from the prediction records.
Seed 42 reproduced both earlier final-fit histories and scores. Hash checks
confirmed that all 75 pre-existing report and model artifacts were preserved.
