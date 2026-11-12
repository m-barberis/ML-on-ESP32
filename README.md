# ML-on-ESP32

Rock, paper, scissors image classification for an ESP32 using TensorFlow Lite
for Microcontrollers. The CNN takes 64×64 RGB images and has 29,515 parameters.

## Final model

The training duration is selected by **seven-fold cross-validation using
validation loss**. Each fold holds out one complete image sequence per class
and trains on the remaining six. Validation loss guides early stopping because
it accounts for predicted probabilities beyond whether a classification is
correct. Sequence prefixes identify image sequences, not verified hand identities.

The seven best epochs were **8, 13, 12, 7, 10, 19, and 8**. Their median,
**10 epochs**, was used to train a fresh model on all **2,520 training images**
with the original **seed 42**. This duration follows from the validation-loss
criterion; the test score was not the criterion used to choose the duration.

The selected model achieved **97.31% test accuracy (362/372)** for both float
and INT8 exports. Its weights are selected for deployment on the ESP32:

- [INT8 model](models/int8/final/rps_cnn.tflite): **36,184 bytes**.
- [Training and test results](reports/metrics/final/results.json).
- [Model card](docs/model-card.md): training procedure and ESP32 input/output contract.

Across ten training seeds, the **ten-epoch INT8 model** achieved a mean test
accuracy of **96.83%**, with a sample standard deviation of **1.47 percentage
points**. The observed range across seeds was **94.35%–98.92%**. This range is
not a confidence interval. See the [seed study](docs/experiments/seed-comparison.md).

## Setup and reproduce training

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
source .venv/bin/activate
```

TensorFlow is pinned in `requirements.txt`. If `.venv` exists, just activate it.
Extract the official dataset into:

```text
data/raw/rps/{paper,rock,scissors}/           # 2,520 training PNGs
data/raw/rps-test-set/{paper,rock,scissors}/  # 372 test PNGs
```

Reproduce the seven-fold selection and final fit using a fresh run name:

```sh
python src/training/seven_fold_epochs.py --monitor val_loss --run my_loss_run
```

Validation loss is the default monitor. Existing outputs are refused. Each run
saves three reports in `reports/metrics/<run>/` and models in
`models/{float,int8}/<run>/final/`. The selected seed-42 run is preserved in
`seven_fold_loss`; its exact exports are also stored at the canonical `final`
paths above. A new run does not automatically replace the selected model.
Dataset downloads and intermediate models are ignored by Git.

## Secondary experiments

Supporting experiments are documented separately:

- [Baseline, augmentation, and dropout](docs/experiments/baseline-augmentation.md).
- [Historical three-fold nested sequence validation](docs/experiments/nested-sequence-validation.md).
- [Seven-fold procedure and accuracy-based comparison](docs/experiments/seven-fold-epochs.md).
- [Multi-seed study](docs/experiments/seed-comparison.md).

## ESP32 integration

The selected weights are ready for integration. Camera accuracy, tensor arena
allocation, combined camera/inference memory, and latency still need hardware
validation. See the [model card](docs/model-card.md) for preprocessing,
quantization, class order, and required operations. The model has no background
or unknown class.
