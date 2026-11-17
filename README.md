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

The Freenove ESP32-Wrover benchmark runs the selected INT8 model on all 372
test images. The computer resizes and quantizes each image using the training
pipeline and sends one tensor at a time over USB. The ESP32 runs TensorFlow Lite
Micro and returns the three raw class scores and measurements.

Close any open serial monitor. From the repository root, activate the Python
environment, install the requirements, and get the board's port with
`pio device list`. Use the port listed on your system in both commands:

```sh
uv pip install -r requirements.txt
PORT=/dev/cu.usbserial-10
pio run -d firmware/ESP32_ML -t upload --upload-port "$PORT" &&
python -m scripts.benchmark_esp32 --port "$PORT"
```

At 115200 baud, the transfer alone takes about 6.6 minutes, plus time for
inference. Each run saves `predictions.csv` and `summary.json` in
a new directory under `runs/esp32/`. The CSV is updated as results arrive. The
summary is written after all 372 images finish.

- **Accuracy** includes the confusion matrix (rows are true classes, columns
  are predictions), per-class recall, and agreement with desktop TensorFlow Lite.
- **Inference latency** measures only the ESP32's `Invoke()` call. Serial
  round-trip latency includes sending the image, inference, and receiving the
  result. Both include the first image and report the mean and maximum.
- **Memory** reports the reserved tensor arena and the mean and maximum of
  TensorFlow Lite Micro's reported arena footprint. It also reports mean
  internal heap occupancy immediately after inference and a peak estimate from
  ESP-IDF's per-region low-water marks during inference. The heap measurement
  covers the system's internal byte-addressable heap and excludes the static
  arena and task stack usage within allocations.

See the [model card](docs/model-card.md) for the input contract and class order.
The model has no background or unknown class. Camera performance requires a
separate hardware check.
