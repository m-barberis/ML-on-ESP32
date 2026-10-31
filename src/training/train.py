"""Train one small CNN, export full INT8 TFLite, and evaluate both models."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["paper", "rock", "scissors"]
IMAGE_SIZE = 64


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def collect_images(directory):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        images = sorted((directory / name).glob("*.png"))
        if not images:
            raise ValueError(f"No PNG images found in {directory / name}")
        paths.extend(images)
        labels.extend([label] * len(images))
    return paths, np.array(labels, dtype=np.int32)


def load_images(paths):
    images = []
    for path in paths:
        image = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
        image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
        images.append(image.numpy() / 255.0)
    return np.asarray(images, dtype=np.float32)


def split_training(labels, seed):
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in range(len(CLASSES)):
        indices = rng.permutation(np.flatnonzero(labels == label))
        count = round(0.2 * len(indices))
        validation.extend(indices[:count])
        training.extend(indices[count:])
    return rng.permutation(training), np.asarray(validation)


def build_model(dropout=0.0, seed=42):
    layers = [tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))]
    for filters in (8, 16, 24):
        layers.extend([
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
        ])
    layers.append(tf.keras.layers.Flatten())
    if dropout:
        layers.append(tf.keras.layers.Dropout(dropout, seed=seed + 3))
    layers.extend([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model = tf.keras.Sequential(layers, name="rps_tiny_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def training_dataset(images, labels, batch_size, seed, enhanced=False):
    layers = [
        tf.keras.layers.RandomRotation(0.05, fill_mode="nearest", seed=seed),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed + 1),
        tf.keras.layers.RandomFlip("horizontal", seed=seed + 2),
    ]
    if enhanced:
        layers.extend([
            tf.keras.layers.RandomZoom(0.1, 0.1, fill_mode="nearest", seed=seed + 4),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0.0, 1.0), seed=seed + 5),
            tf.keras.layers.RandomContrast(0.15, value_range=(0.0, 1.0), seed=seed + 6),
        ])
    augmentation = tf.keras.Sequential(layers)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images), seed=seed).batch(batch_size)
    dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))
    return dataset.prefetch(1)


def export_models(model, calibration_images, seed):
    indices = np.random.default_rng(seed).choice(
        len(calibration_images), size=min(200, len(calibration_images)), replace=False,
    )

    def representative_dataset():
        for i in indices:
            yield [calibration_images[i:i + 1]]

    # Fix batch size to one to avoid dynamic shape operations on the MCU.
    export_model = tf.keras.models.clone_model(
        model, input_tensors=tf.keras.Input(batch_shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3)),
    )
    export_model.set_weights(model.get_weights())
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    float_bytes = converter.convert()
    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return float_bytes, converter.convert()


def metrics(labels, predictions):
    matrix = tf.math.confusion_matrix(labels, predictions, num_classes=3).numpy()
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "confusion_matrix": matrix.tolist(),
        "per_class_recall": {
            name: float(matrix[i, i] / matrix[i].sum())
            for i, name in enumerate(CLASSES)
        },
    }


def evaluate_tflite(model_bytes, images):
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    if input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8:
        raise ValueError("Expected INT8 input and output")
    scale, zero_point = input_info["quantization"]
    predictions = []
    for image in images:
        quantized = np.clip(np.rint(image / scale + zero_point), -128, 127)
        interpreter.set_tensor(input_info["index"], quantized.astype(np.int8)[None])
        interpreter.invoke()
        predictions.append(int(interpreter.get_tensor(output_info["index"])[0].argmax()))
    # These tensor sizes are inspectable, but are NOT TFLM's arena requirement.
    tensors = interpreter.get_tensor_details()
    operations = sorted({op["op_name"] for op in interpreter._get_ops_details()
                         if op["op_name"] != "DELEGATE"})
    details = {
        "input_shape": input_info["shape"].tolist(),
        "input_dtype": "int8",
        "input_scale": float(scale),
        "input_zero_point": int(zero_point),
        "output_dtype": "int8",
        "output_scale": float(output_info["quantization"][0]),
        "output_zero_point": int(output_info["quantization"][1]),
        "operators": operations,
        "has_float_tensors": any(t["dtype"] == np.float32 for t in tensors),
    }
    if details["has_float_tensors"]:
        raise ValueError("Export contains unexpected float tensors")
    return np.asarray(predictions), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default=None,
                        help="Save a named run separately from the original baseline")
    parser.add_argument("--augmentation", choices=("basic", "enhanced"), default="basic")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--export-only", action="store_true",
                        help="Re-export and evaluate existing weights without training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be between 0 (inclusive) and 1 (exclusive)")
    if args.run and (Path(args.run).name != args.run or args.run in {".", ".."}):
        parser.error("run must be a single folder name")

    reports = ROOT / "reports/metrics"
    float_dir = ROOT / "models/float"
    int8_dir = ROOT / "models/int8"
    if args.run:
        reports /= args.run
        float_dir /= args.run
        int8_dir /= args.run
    config = {
        "run": args.run or "baseline", "maximum_epochs": args.epochs,
        "batch_size": args.batch_size, "seed": args.seed,
        "augmentation": args.augmentation, "dropout": args.dropout,
    }
    if args.export_only:
        saved_config = json.loads((reports / "config.json").read_text())
        if saved_config != config:
            parser.error("export-only options must match the saved run configuration")
    elif args.run and reports.exists():
        parser.error("run already exists; choose a new name to preserve its results")
    reports.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        save_json(reports / "config.json", config)

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    paths, labels = collect_images(ROOT / "data/raw/rps")
    train_indices, val_indices = split_training(labels, args.seed)
    images = load_images(paths)
    x_train, y_train = images[train_indices], labels[train_indices]
    x_val, y_val = images[val_indices], labels[val_indices]
    save_json(reports / "split.json", {
        "seed": args.seed,
        "validation_fraction": 0.2,
        "classes": CLASSES,
        "training": [str(paths[i].relative_to(ROOT)) for i in train_indices],
        "validation": [str(paths[i].relative_to(ROOT)) for i in val_indices],
    })

    training = training_dataset(
        x_train, y_train, args.batch_size, args.seed, args.augmentation == "enhanced",
    )
    validation = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

    model = (tf.keras.models.load_model(float_dir / "rps_cnn.keras")
             if args.export_only else build_model(args.dropout, args.seed))
    model.summary()
    print(f"Training: {len(y_train)}; validation: {len(y_val)}", flush=True)
    if not args.export_only:
        model.fit(
            training,
            validation_data=validation,
            epochs=args.epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=7, restore_best_weights=True,
                ),
                tf.keras.callbacks.CSVLogger(str(reports / "history.csv")),
            ],
            verbose=2,
        )

    float_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)
    if not args.export_only:
        model.save(float_dir / "rps_cnn.keras")

    # Calibrate on training images only. Augmentation stays outside the export.
    float_bytes, int8_bytes = export_models(model, x_train, args.seed)
    (float_dir / "rps_cnn.tflite").write_bytes(float_bytes)
    (int8_dir / "rps_cnn.tflite").write_bytes(int8_bytes)

    # Model selection is finished before reading the official test images.
    test_paths, y_test = collect_images(ROOT / "data/raw/rps-test-set")
    x_test = load_images(test_paths)
    float_validation = model.predict(x_val, verbose=0).argmax(axis=1)
    float_test = model.predict(x_test, verbose=0).argmax(axis=1)
    int8_validation, deployment = evaluate_tflite(int8_bytes, x_val)
    int8_test, _ = evaluate_tflite(int8_bytes, x_test)
    with (reports / "history.csv").open() as file:
        history = list(csv.DictReader(file))
    results = {
        "config": config,
        "tensorflow_version": tf.__version__,
        "seed": args.seed,
        "training": {
            "maximum_epochs": args.epochs,
            "completed_epochs": len(history),
            "selected_epoch": 1 + int(max(history, key=lambda row: float(row["val_accuracy"]))["epoch"]),
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "early_stopping": {"monitor": "val_accuracy", "patience": 7},
        },
        "classes": CLASSES,
        "confusion_matrix_axes": "rows: true class; columns: predicted class",
        "parameters": model.count_params(),
        "samples": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "chance_accuracy": 1 / 3,
        "float": {"validation": metrics(y_val, float_validation),
                  "test": metrics(y_test, float_test), "tflite_bytes": len(float_bytes)},
        "int8": {"validation": metrics(y_val, int8_validation),
                 "test": metrics(y_test, int8_test), "tflite_bytes": len(int8_bytes)},
        "deployment": deployment,
        "model_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        "limitations": [
            "Random stratified validation split can share synthetic hand styles with training.",
            "Official test results now inform experiment design; this is a development benchmark.",
            "Public dataset results do not establish accuracy on real camera images.",
            "ESP32 tensor arena, camera RAM, operator compatibility, and latency require device testing.",
        ],
    }
    results["test_accuracy_drop_percentage_points"] = 100 * (
        results["float"]["test"]["accuracy"] - results["int8"]["test"]["accuracy"]
    )
    results["targets"] = {
        "float_validation_at_least_90_percent": results["float"]["validation"]["accuracy"] >= 0.9,
        "int8_test_drop_at_most_2_percentage_points": results["test_accuracy_drop_percentage_points"] <= 2,
        "int8_file_below_100_kb": len(int8_bytes) < 100_000,
    }
    save_json(reports / "results.json", results)
    with (reports / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "true_class", "float_prediction", "int8_prediction"])
        for path, truth, pred_float, pred_int8 in zip(test_paths, y_test, float_test, int8_test):
            writer.writerow([str(path.relative_to(ROOT)), CLASSES[truth],
                             CLASSES[pred_float], CLASSES[pred_int8]])
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
