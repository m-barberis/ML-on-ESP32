# ML-on-ESP32

Rock, paper, scissors image classification for an ESP32 using TensorFlow Lite
for Microcontrollers. The CNN takes 64×64 RGB images and has 29,515 parameters.
Fully INT8 exports occupy 36,184 bytes.

The current deployment candidate remains
[`models/int8/final/rps_cnn.tflite`](models/int8/final/rps_cnn.tflite).
It scored 93.82% on the official development test set. Later experiments are
preserved separately; a replacement has not been selected. Camera accuracy,
ESP32 memory use, and inference latency still need hardware validation.

## Setup

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
data/raw/rps-test-set/{paper,rock,scissors}/  # 372 development-test PNGs
```

The official test set has informed experiment design, so it is a development
benchmark. A fresh camera dataset is needed for a final performance estimate.

## Train and export

Run a named baseline from the repository root:

```sh
python src/training/train.py --run my_baseline
```

Choose a fresh name for each run. Models go to `models/float/<run>/` and
`models/int8/<run>/`; reports go to `reports/metrics/<run>/`. Named runs refuse
existing report directories. Running `train.py` without `--run` overwrites the
original baseline outputs. Dataset downloads and intermediate models are ignored
by Git; the original final INT8 candidate is retained.

## Experiments

| Experiment | Main finding | Details |
| --- | --- | --- |
| Baseline, augmentation, dropout | INT8 development-test accuracy: 86.02%, 89.78%, 95.70% | [Summary](docs/experiments/baseline-augmentation.md) |
| Three-fold nested sequence validation | Selected 8 epochs; final INT8 accuracy 93.82% | [Summary](docs/experiments/nested-sequence-validation.md) |
| Seven-fold epoch selection | Accuracy selected 7 epochs; loss selected 10. Seed-42 INT8 accuracies: 100% and 97.31% | [Summary](docs/experiments/seven-fold-epochs.md) |
| Ten paired training seeds | INT8 mean ± sample SD: 7 epochs, 96.51% ± 2.88 pp; 10 epochs, 96.83% ± 1.47 pp | [Summary](docs/experiments/seed-comparison.md) |

Ten epochs had a lower observed spread across seeds; its mean accuracy advantage
was small. The seven-fold and seed-comparison runs each keep three report files:
results, combined histories, and combined predictions.

## ESP32 integration

See the [model card](docs/model-card.md) for preprocessing, quantization,
class order, supported operations, and deployment checks. The model always
predicts paper, rock, or scissors; it has no background or unknown class.
