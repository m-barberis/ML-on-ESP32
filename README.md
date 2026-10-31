# ML-on-ESP32

Rock, paper, scissors image classification for deployment on an ESP32 with
TensorFlow Lite for Microcontrollers.

The current deployment candidate is
[`models/int8/final/rps_cnn.tflite`](models/int8/final/rps_cnn.tflite).
It is 36,184 bytes and scored 93.82% on the official development test set.
Three-fold sequence-held-out evaluation scored 94.37% pooled INT8 accuracy.
See [the model card](docs/model-card.md) for the evaluation details and limits.

## Python environment

Use Python 3.12. With [uv](https://docs.astral.sh/uv/) installed, create the
environment and install TensorFlow from the repository root:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"
```

If `.venv` already exists, just activate it. The environment and locally downloaded
Python runtime are excluded from Git. TensorFlow is pinned in `requirements.txt`.

## Dataset

Extract the official archives into these directories:

```text
data/raw/rps/{paper,rock,scissors}/           # 2,520 training PNGs
data/raw/rps-test-set/{paper,rock,scissors}/  # 372 test PNGs
```

The script reserves 20% of each training class for validation using seed 42:
2,016 training images and 504 validation images. The official test set is evaluated
after model selection and is not used for training, early stopping, or calibration.

## Train and export

From the repository root, with the environment activated:

```sh
python src/training/train.py
```

This runs one baseline, with at most 30 epochs and early stopping. It overwrites
the previous model and report files. Settings live in the script; `--epochs`,
`--batch-size`, and `--seed` are optional overrides.

The CNN uses 64x64 RGB input, three 3x3 convolution/ReLU/max-pooling blocks
(8, 16, 24 channels), then flatten, a 16-unit dense layer, and a 3-class softmax.
There are **29,515 trainable parameters**. Training uses small rotations,
translations, and horizontal flips. Augmentation is not part of the exported model.

Outputs:

- `models/float/rps_cnn.keras`: trained Keras model.
- `models/float/rps_cnn.tflite`: float TFLite model.
- `models/int8/rps_cnn.tflite`: fully integer-quantized deployment candidate.
- `reports/metrics/results.json`: accuracy, confusion matrices, sizes, quantization
  metadata, and deployment limitations.
- `reports/metrics/history.csv`: learning curves as epoch-by-epoch measurements.
- `reports/metrics/split.json`: exact training/validation membership.
- `reports/metrics/predictions.csv`: individual test predictions.

Datasets and intermediate models are ignored by Git. The final INT8 model,
scripts, and reports are tracked.

### Augmentation and dropout comparisons

The original baseline is preserved at the paths above. Named runs save their
models under `models/float/<run>/` and `models/int8/<run>/`, and their reports
under `reports/metrics/<run>/`. Existing named runs cannot be overwritten.

```sh
python src/training/train.py --run augmented --augmentation enhanced
python src/training/train.py --run augmented_dropout --augmentation enhanced --dropout 0.25
```

Use a fresh run name when repeating an experiment. Enhanced augmentation adds
10% zoom, 10% brightness variation, and 15% contrast variation to the original
rotations, translations, and flips. The optional dropout is placed before the
16-unit dense layer and is inactive during inference. Seed, split, model capacity,
optimizer, epoch budget, and early stopping remain the same in these comparisons.
Each run records its options in `config.json`.

The official test set now informs experiment design; treat its scores as a
development benchmark, rather than an independent final estimate. A fresh camera
test set will be needed for that final estimate.

The two approved comparisons kept seed 42 and identical split membership. Each
checkpoint was selected by validation accuracy before evaluating the official
test set; there was no further tuning after seeing these two test results.

| Run | INT8 validation | INT8 test | Paper recall | INT8 bytes |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 99.01% | 86.02% | 69.35% | 36,184 |
| Enhanced augmentation | 98.02% | 89.78% | 72.58% | 36,184 |
| Enhanced augmentation + 25% dropout | 99.40% | 95.70% | 87.10% | 36,184 |

The dropout run completed 22 epochs and restored epoch 15. Float and INT8 test
accuracy were both 95.70%. Its model is
`models/int8/augmented_dropout/rps_cnn.tflite`; its reports are in
`reports/metrics/augmented_dropout/`. The original baseline remains unchanged.
`reports/metrics/comparison.csv` contains the full comparison. All three exports
use the same five operations and have no float tensors; augmentation and dropout
added no inference operations or model bytes.

The filename audit in `reports/metrics/sequence_audit.json` found seven source
sequence prefixes per class, with all 21 prefixes shared between random training
and validation. This motivates holding out entire sequences for the next validation
experiment. Prefixes are sequence identifiers, not verified hand identities.
The grouped cross-validation below follows up on this audit. One seed per
configuration is not enough to establish the stability of these improvements.

### Grouped cross-validation and final model

```sh
python src/training/cross_validate.py
```

This command preserves existing outputs and refuses to overwrite a completed run.
It uses the fixed enhanced-augmentation/25%-dropout architecture, with three outer
folds holding out complete source sequences. In each fold, one sequence per class
from the remaining data is used for inner validation and early stopping (at most
30 epochs). A fresh model is then fit on all outer training images for that selected
duration, calibrated only on outer training images, and scored on the outer fold.

The inner best epochs were 8, 16, and 7. Their median, **8 epochs**, was fixed before
training a fresh final model on all **2,520 training images**. The official test set
was excluded from fold generation, epoch selection, training, and calibration.

| Evaluation | INT8 accuracy |
| --- | ---: |
| Outer fold 1 (1,080 images) | 88.15% |
| Outer fold 2 (720 images) | 98.06% |
| Outer fold 3 (720 images) | 100.00% |
| Fold mean / sample standard deviation | 95.40% / 6.36 percentage points |
| Pooled out-of-fold (2,520 images) | 94.37% |
| Final refit on official development test (372 images) | 93.82% |

Fold sizes differ because seven sequences per class cannot divide equally into
three folds. The pooled score weights images equally; the mean weights folds
equally. Fold variation is substantial, and three folds do not establish a tight
confidence interval. Group prefixes are not verified hand identities.

The previous random-split dropout model scored 95.70% on the same official test
set. The final refit is **not an accuracy improvement over that checkpoint**;
its epoch budget was chosen by grouped inner validation, not by selecting the best
official test score. Both artifacts are preserved. More trustworthy validation
does not guarantee a higher test score.

Artifacts:

- `models/int8/final/rps_cnn.tflite`: committed deployment candidate.
- `models/float/final/rps_cnn.keras`: local final Keras weights.
- `reports/metrics/final/`: final config, training history, results, and predictions.
- `reports/metrics/grouped_cv/`: methodology, fold memberships, histories,
  per-fold results, and predictions for every out-of-fold image.

The fixed model is ready for an ESP32 integration test, not yet validated on the
camera or microcontroller. Avoid further tuning against the official test scores.

## First baseline results

Training stopped after 13 epochs and restored epoch 6, the first epoch with the
highest validation accuracy. No architecture search or test-driven retraining was
performed.

| Model | Validation accuracy | Test accuracy | TFLite size |
| --- | ---: | ---: | ---: |
| Float | 99.01% | 85.75% | 122,340 bytes |
| INT8 | 99.01% | 86.02% | 36,184 bytes |

INT8 test recall is 69.35% for paper, 100% for rock, and 88.71% for scissors.
Paper is the main weakness. The validation/test gap suggests the random validation
split is easier; synthetic hand styles may be shared across training and validation.
These results do not establish accuracy on real camera images. A camera test set
is still needed. No exact file duplicates were found across the official splits.

## ESP32 input/output contract

- Input: fixed shape `[1, 64, 64, 3]`, RGB, `int8`.
- Preprocessing: bilinear resize the full image, normalize pixels to `[0, 1]`, then
  quantize with `round(value / input_scale + input_zero_point)`, clipped to
  `[-128, 127]`. Read the exact scale and zero point from `results.json`.
- Output order: **paper, rock, scissors**. Choose the highest output value.
- Required operations: `CONV_2D`, `MAX_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`,
  and `SOFTMAX`; ReLU is fused into the corresponding layers.
- INT8 inputs/outputs and absence of float tensors were verified with the desktop
  TFLite interpreter. This does not replace an ESP32 TFLM test.

The 36 KB model is a promising deployment candidate, but tensor arena allocation,
camera buffers, operation support, and latency must be measured on the actual board.
There is no background class: an empty scene still receives one of the three labels.
