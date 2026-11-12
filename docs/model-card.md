# Rock Paper Scissors selected deployment model

The selected weights come from the ten-epoch, seed-42 final fit following
seven-fold validation-loss epoch selection. The earlier eight-epoch model is
preserved with the [historical three-fold experiment](experiments/nested-sequence-validation.md).

## Intended use

Recognize one rock, paper, or scissors gesture in a controlled camera view.
There is no background or unknown class. Empty scenes also receive a gesture label.

## Artifact and input contract

- Model: `models/int8/final/rps_cnn.tflite` (36,184 bytes).
- Source export: `models/int8/seven_fold_loss/final/rps_cnn.tflite`.
  These files are byte-identical; no retraining or re-export was used to select it.
- INT8 SHA-256: `9326de6d69ec2d424980715403433e955977674733d86c6a4c66497d9f3b4ca7`.
- Architecture: three convolution/ReLU/max-pooling blocks (8/16/24 channels),
  flatten, 25% training-only dropout, dense 16, dense 3 with softmax.
- Parameters: 29,515. Dropout adds no inference operations.
- Input: `[1, 64, 64, 3]`, RGB, INT8. Resize the full frame with bilinear
  interpolation, normalize to `[0, 1]`, then quantize with
  `round(value / input_scale + input_zero_point)`, clipped to `[-128, 127]`.
  Input scale is approximately `1/255`, with zero point `-128`.
- Output class order: **paper, rock, scissors**. Choose the largest output value.
- Output quantization: scale `1/256`, zero point `-128`.
- Required operations: `CONV_2D`, `MAX_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`,
  `SOFTMAX`. ReLU is fused. No float tensors were found.
- Exact tensor metadata and selection provenance:
  [final results](../reports/metrics/final/results.json).

## Training and model selection

TensorFlow 2.21.0, seed 42, Adam at 0.001, batch size 32. Training uses rotations,
translations, flips, zoom, brightness, and contrast variation. The shared CNN and
augmentation code is in `src/training/train.py`; the main procedure is in
`src/training/seven_fold_epochs.py`, which defaults to validation loss.

Seven folds each hold out one complete sequence per class. Each fresh model
trains on 2,160 images and uses 360 for validation. Every sequence is held out
once. Early stopping minimizes validation loss with patience 7, `min_delta=0`,
and a 30-epoch cap. Fold seeds are 42–48.

Validation loss is preferred because it retains information about predicted
probabilities beyond classification accuracy. Selected durations were
**8, 13, 12, 7, 10, 19, and 8 epochs**; no fold reached the cap. Their median,
**10 epochs**, sets the final duration. The original seed **42** was retained;
it was not selected by ranking individual seeds on test accuracy.

A fresh final model trains on all 2,520 images for ten epochs without early
stopping. INT8 calibration uses 200 training images selected with seed 42.
The official 372-image test set is excluded from folds, early stopping, final
training, and calibration. It is evaluated after the final fit.

## Results

| Evaluation of the selected seed-42 model | Float | INT8 |
| --- | ---: | ---: |
| Test accuracy | 97.31% (362/372) | 97.31% (362/372) |
| Paper recall | 97.58% | 98.39% |
| Rock recall | 100.00% | 100.00% |
| Scissors recall | 94.35% | 93.55% |

Float test cross-entropy is 0.06691. Quantization changes some individual
predictions but preserves the overall test accuracy.

The supporting [ten-seed study](experiments/seed-comparison.md) evaluated the
same ten-epoch recipe with seeds 42–51: INT8 mean accuracy **96.83%**, sample
standard deviation **1.47 percentage points**, and observed range
**94.35%–98.92%**. This is variation across training seeds on the same test
images, not a confidence interval for accuracy on new images.

## Provenance and secondary experiments

The source run's complete folds, histories, and predictions remain in
`reports/metrics/seven_fold_loss/`. The canonical `reports/metrics/final/`
contains three files: selected-model results, its ten-epoch training history,
and its 372 test predictions. Canonical Keras and float TFLite files are in
`models/float/final/`.

The original eight-epoch exports and reports are archived under
`models/{float,int8}/grouped_cv/final/` and
`reports/metrics/grouped_cv/final/`. The accuracy-based seven-fold run,
augmentation comparisons, and paired seed study remain supporting experiments.
See the [experiment index](../README.md#secondary-experiments).

## Hardware validation still required

Sequence identifiers are not verified hand identities. Validation folds guide
epoch selection and do not provide an independent evaluation of the selected
checkpoint. This is a synthetic dataset; real-camera accuracy is unmeasured.

Desktop verification covers model loading, integer-only tensors, operation
types, and predictions. ESP32 TFLM operator support, tensor arena size, combined
camera/inference memory, preprocessing parity, and latency still need device tests.
