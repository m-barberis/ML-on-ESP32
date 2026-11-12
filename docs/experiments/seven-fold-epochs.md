# Seven-fold epoch selection

## Question and method

The main procedure chooses the final training duration using validation loss,
holding out each of the seven sequences per class in turn. Loss retains
information about predicted probabilities beyond classification accuracy.
The accuracy-based stopping experiment is a secondary comparison.

Each fold holds out one entire sequence per class: 2,160 training images and
360 validation images. Every sequence is held out once. The architecture uses
enhanced augmentation and 25% dropout; batch size is 32. Fold seeds are 42–48.
Early stopping uses patience 7 and a 30-epoch cap, restoring the first optimum.
Only the monitored quantity changes between the two experiments. All 104 shared
training epochs, including shared final-fit epochs, matched between runs.

Each experiment takes the median of its seven selected epochs, then trains a
fresh seed-42 model on all 2,520 images for that fixed duration. The development
test set is evaluated afterward, with no final-fit early stopping.

## Results

| Fold | Accuracy-selected epoch | Loss-selected epoch |
| --- | ---: | ---: |
| 1 | 8 | 8 |
| 2 | 6 | 13 |
| 3 | 6 | 12 |
| 4 | 7 | 7 |
| 5 | 6 | 10 |
| 6 | 7 | 19 |
| 7 | 9 | 8 |
| Median | **7** | **10** |

No fold reached the cap. In several folds, loss continued improving after
accuracy first peaked. The actual loss-based rerun selected epoch 19 in fold 6;
its minimum within the earlier, accuracy-truncated history had been epoch 13.

| Final-fit criterion | Float / INT8 development-test accuracy | Float test loss |
| --- | ---: | ---: |
| Validation accuracy: 7 epochs | 100.00% / 100.00% (372/372) | 0.05525 |
| Validation loss: 10 epochs | 97.31% / 97.31% (362/372) | 0.06691 |

Both INT8 models are 36,184 bytes with the same five operations and no float
tensors. Loss-based stopping selected longer training, but did not improve this
run's development-test accuracy or loss. Both final scores use one seed and do
not establish stability across training seeds. Fold scores inform epoch selection,
so they are tuning results rather than independent performance estimates.
The accuracy-based pooled float validation score was 95.95%.

Reloading the saved models reproduced their scores. An audit found no shared
resolved paths, identical file contents, or identical preprocessed images across
official training and test splits. This does not rule out shared synthetic
characteristics. A perfect score alone does not establish overfitting; neither
model has been evaluated on a fresh camera dataset. Sequence prefixes do not
verify hand identities.

## Commands and artifacts

```sh
python src/training/seven_fold_epochs.py --monitor val_loss
python src/training/seven_fold_epochs.py --monitor val_accuracy
```

Existing run outputs are refused. Use a fresh `--run NAME` to repeat an experiment.
Validation loss is the default monitor. Default report directories are
`reports/metrics/seven_fold_loss/` for loss and `reports/metrics/seven_fold_epochs/`
for accuracy. Each contains exactly three files:

- `results.json`: configuration, folds, selected epochs, and final results.
- `history.csv`: combined training curves with stage, fold, and one-based epoch.
- `predictions.csv`: combined validation and development-test predictions.

Models are in `models/{float,int8}/<run>/final/`. The earlier accuracy-run reports
were consolidated while preserving metadata, histories, and predictions.
The ten-epoch, seed-42 loss-based exports are the selected deployment weights.
Exact copies are at `models/{float,int8}/final/`; the original eight-epoch candidate
is archived with the historical three-fold experiment. See the [model card](../model-card.md).

The follow-up [ten-seed comparison](seed-comparison.md) measures the stability
of both fixed durations, rather than choosing between these two seed-42 scores.

See [accuracy-run results](../../reports/metrics/seven_fold_epochs/results.json)
and [loss-run results](../../reports/metrics/seven_fold_loss/results.json).
