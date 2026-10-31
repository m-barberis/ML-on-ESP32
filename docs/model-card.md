# Rock Paper Scissors deployment candidate

## Intended use

Recognize one rock, paper, or scissors gesture in a controlled camera view.
There is no background or unknown class. Empty scenes also receive a gesture label.

## Artifact and input contract

- Model: `models/int8/final/rps_cnn.tflite` (36,184 bytes).
- Architecture: three convolution/ReLU/max-pooling blocks (8/16/24 channels),
  flatten, 25% training-only dropout, dense 16, dense 3 with softmax.
- Parameters: 29,515. Dropout adds no inference operations.
- Input: `[1, 64, 64, 3]`, RGB, INT8. Resize the full frame with bilinear
  interpolation, normalize to `[0, 1]`, then quantize using the input tensor scale
  and zero point. The saved values are approximately `1/255` and `-128`.
- Output class order: **paper, rock, scissors**. Choose the largest output value.
- Output quantization: scale `1/256`, zero point `-128`.
- Required operations: `CONV_2D`, `MAX_POOL_2D`, `RESHAPE`, `FULLY_CONNECTED`,
  `SOFTMAX`. ReLU is fused. No float tensors were found.
- Model hash and exact tensor metadata: `reports/metrics/final/results.json`.

## Training and model selection

TensorFlow 2.21.0, seed 42, Adam at 0.001, batch size 32. Training uses small
rotations, translations, flips, zoom, brightness, and contrast variation. Source
code is in `src/training/train.py`; the cross-validation/refit procedure is in
`src/training/cross_validate.py`.

Architecture and augmentation were fixed before three-fold grouped cross-validation.
Entire filename-prefix sequences were held out in each outer fold. A separate
inner sequence per class selected the training duration, with no outer-fold access
for early stopping. Each outer model was refit from scratch on its full training
portion. All 2,520 images were scored exactly once out of fold.

Inner-selected durations were 8, 16, and 7 epochs. A fresh final model used the
median duration (8 epochs) and all 2,520 official training images. Calibration
used 200 training images. The official 372-image test set was read only after the
final model and duration were fixed in this procedure.

## Results

| Evaluation | Float | INT8 |
| --- | ---: | ---: |
| Pooled grouped out-of-fold accuracy | 94.17% | 94.37% |
| Unweighted fold mean accuracy | 95.25% | 95.40% |
| Sample standard deviation across folds | 6.62 pp | 6.36 pp |
| Final model on official development test | 94.09% | 93.82% |

Final INT8 test recall: paper 86.29%, rock 100%, scissors 95.16%.
The INT8 conversion reduced final test accuracy by 0.27 percentage points.

The final model scored below the earlier random-split dropout checkpoint (95.70%
INT8 test accuracy). It was retained as the final all-data refit under the stated
selection procedure, without changing its duration in response to test results.

## Limits and next acceptance checks

The official test set informed earlier experiment design, so it is a development
benchmark rather than a fresh final test. Cross-validation estimates performance
of the training procedure on smaller training sets; it is not a direct measurement
of the single final model. Three folds show substantial variation.

Sequence identifiers are not verified hand identities; grouping by prefix prevents
within-sequence overlap but does not establish person-independent evaluation.
This is a synthetic dataset, and real camera accuracy remains unmeasured.

Desktop verification covers model loading, integer-only tensors, standard operation
types, and predictions. ESP32 TFLM operator support, tensor arena size, combined
camera/inference memory, preprocessing parity, and latency still need device tests.
