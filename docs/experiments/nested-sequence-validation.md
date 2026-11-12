# Three-fold nested sequence validation

## Question and method

Evaluate the fixed enhanced-augmentation/25%-dropout architecture on unseen
sequences while keeping evaluation separate from epoch selection.

Three outer folds hold out complete filename-prefix sequences. Within each
outer training portion, one sequence per class is reserved for inner validation.
Validation accuracy selects the duration with patience 7 and a 30-epoch cap.
A fresh model then trains on all outer training images for that duration,
calibrates using only outer training images, and is scored on the outer fold.
Every one of the 2,520 images is scored once out of fold.

Inner validation selected epochs **8, 16, and 7**. Their median, **8 epochs**,
was fixed before training a fresh seed-42 model on all 2,520 images. The official
test set was excluded from folds, epoch selection, training, and calibration.

## Results

| Evaluation | INT8 accuracy |
| --- | ---: |
| Outer fold 1: 1,080 images | 88.15% |
| Outer fold 2: 720 images | 98.06% |
| Outer fold 3: 720 images | 100.00% |
| Fold mean / sample standard deviation | 95.40% / 6.36 percentage points |
| Pooled out-of-fold: 2,520 images | 94.37% |
| Final refit: 372 development-test images | 93.82% |

Seven sequences per class do not divide evenly into three folds. The pooled
score weights images equally; the mean weights folds equally. Three folds with
this much variation do not establish a precise performance estimate. Sequences
are not verified hand identities.

The final refit scored below the earlier random-split dropout model (95.70%).
Its duration was chosen using inner validation rather than official test scores.
It is now a historical experiment. The selected deployment weights come from
the ten-epoch validation-loss procedure described in the [model card](../model-card.md).

## Command and artifacts

```sh
python src/training/cross_validate.py
```

The command refuses existing outputs. Fold reports are in
`reports/metrics/grouped_cv/`; its archived final reports are in
`reports/metrics/grouped_cv/final/`. Its final exports are in
`models/float/grouped_cv/final/` and `models/int8/grouped_cv/final/`.
The script writes to these experiment-specific paths, preserving the selected
deployment model at the canonical `final` paths.

See [fold results](../../reports/metrics/grouped_cv/results.json),
[archived final results](../../reports/metrics/grouped_cv/final/results.json), and
[the model card](../model-card.md) for the deployment contract and limitations.
