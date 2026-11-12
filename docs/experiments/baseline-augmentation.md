# Baseline, augmentation, and dropout

## Question and method

Can stronger augmentation and dropout improve generalization without increasing
ESP32 inference cost? All three runs used seed 42 and the same random stratified
split: 2,016 training and 504 validation images. Checkpoints were selected by
validation accuracy, with patience 7 and at most 30 epochs, before evaluating
the 372-image official development test set.

The CNN uses three convolution/ReLU/max-pooling blocks (8, 16, 24 channels),
flatten, a 16-unit dense layer, and a three-class softmax. Input is 64×64 RGB;
there are 29,515 trainable parameters. Training uses Adam at 0.001 and batch size
32. Basic augmentation includes rotation, translation, and horizontal flips.
Enhanced augmentation adds 10% zoom, 10% brightness variation, and 15% contrast
variation. Optional 25% dropout is placed before the 16-unit dense layer.

## Results

| Run | Selected epoch | INT8 validation | INT8 development test | Paper recall |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 6 | 99.01% | 86.02% | 69.35% |
| Enhanced augmentation | 4 | 98.02% | 89.78% | 72.58% |
| Enhanced augmentation + dropout | 15 | 99.40% | 95.70% | 87.10% |

The baseline completed 13 epochs; the dropout run completed 22. Baseline float
test accuracy was 85.75%; dropout float accuracy was 95.70%. All INT8 exports
are 36,184 bytes and use the same five operations with no float tensors.
Augmentation and dropout add no inference operations.

The filename audit found seven sequences per class, with all 21 shared between
random training and validation. Related images may make this validation split
easier than unseen-sequence evaluation. Prefixes identify sequences, not verified
hand identities. Each configuration used one seed; the improvements need repeated
runs to establish stability. No exact file duplicates were found across the
official training and test splits.

## Commands and artifacts

```sh
python src/training/train.py
python src/training/train.py --run augmented --augmentation enhanced
python src/training/train.py --run augmented_dropout --augmentation enhanced --dropout 0.25
```

The first command overwrites the original baseline. Existing named report
directories are refused; use a fresh `--run` name for repeats.

Baseline models are in `models/float/` and `models/int8/`; named models are in
`models/{float,int8}/<run>/`. Keras and float TFLite exports accompany each INT8
model. Baseline reports are in `reports/metrics/`; named reports are in their
respective subdirectories. Reports retain configuration, exact split membership,
histories, results, and predictions.

See [the comparison CSV](../../reports/metrics/comparison.csv) and
[the sequence audit](../../reports/metrics/sequence_audit.json).
The official test set is a development benchmark, not an untouched final test.
