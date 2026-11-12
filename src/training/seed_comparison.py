"""Compare fixed seven- and ten-epoch checkpoints across ten paired seeds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from train import (
    CLASSES, ROOT, build_model, collect_images, evaluate_tflite, export_models,
    load_images, metrics, np, save_json, tf, training_dataset,
)


SEEDS = list(range(42, 52))
EPOCHS = (7, 10)
CALIBRATION_SEED = 42


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def aggregate(runs):
    expected = {(seed, epochs) for seed in SEEDS for epochs in EPOCHS}
    if len(runs) != len(expected) or {(r["seed"], r["epochs"]) for r in runs} != expected:
        raise ValueError("Aggregation requires exactly one result per planned seed and duration")
    by_epochs = {}
    for epochs in EPOCHS:
        selected = [r for r in runs if r["epochs"] == epochs]
        by_epochs[str(epochs)] = {
            "n": len(selected),
            **{kind: describe([r[kind]["accuracy"] for r in selected])
               for kind in ("float", "int8")},
        }
    paired = []
    for seed in SEEDS:
        pair = {r["epochs"]: r for r in runs if r["seed"] == seed}
        paired.append({"seed": seed,
                       **{kind: pair[10][kind]["accuracy"] - pair[7][kind]["accuracy"]
                          for kind in ("float", "int8")}})
    return {
        "accuracy_units": "fraction; multiply by 100 for percent or percentage-point differences",
        "by_epochs": by_epochs,
        "paired_10_minus_7": {
            "per_seed": paired,
            **{kind: {**describe([r[kind] for r in paired]),
                      "ten_better": sum(r[kind] > 0 for r in paired),
                      "seven_better": sum(r[kind] < 0 for r in paired),
                      "ties": sum(r[kind] == 0 for r in paired)}
               for kind in ("float", "int8")},
        },
    }


class CaptureCheckpoints(tf.keras.callbacks.Callback):
    def __init__(self, seed, history_path):
        super().__init__()
        self.seed, self.history_path = seed, history_path
        self.snapshots = {}

    def on_epoch_end(self, epoch, logs=None):
        completed = epoch + 1
        if completed in EPOCHS:
            self.snapshots[completed] = self.model.get_weights()
        exists = self.history_path.exists()
        with self.history_path.open("a", newline="") as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["seed", "epoch", "accuracy", "loss"])
            writer.writerow([self.seed, completed, logs["accuracy"], logs["loss"]])


def manifest(paths, labels):
    return [{"path": str(p.relative_to(ROOT)), "class": CLASSES[label],
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p, label in zip(paths, labels)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="seed_comparison")
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in {".", ".."}:
        parser.error("run must be a single folder name")
    reports = ROOT / "reports/metrics" / args.run
    if reports.exists():
        parser.error("Run outputs already exist; choose a fresh run name")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    train_paths, labels = collect_images(ROOT / "data/raw/rps")
    test_paths, test_labels = collect_images(ROOT / "data/raw/rps-test-set")
    if len(train_paths) != 2520 or len(test_paths) != 372:
        raise ValueError("Expected the unchanged 2,520 training and 372 development-test images")
    training_manifest = manifest(train_paths, labels)
    test_manifest = manifest(test_paths, test_labels)
    assert not {p.resolve() for p in train_paths} & {p.resolve() for p in test_paths}
    assert not {r["sha256"] for r in training_manifest} & {r["sha256"] for r in test_manifest}
    calibration_indices = np.random.default_rng(CALIBRATION_SEED).choice(
        len(labels), size=200, replace=False,
    )
    summary = {
        "status": "running",
        "config": {
            "run": args.run, "seeds": SEEDS, "epochs": list(EPOCHS),
            "method": "One ten-epoch trajectory per seed; capture weights at seven and ten",
            "early_stopping": False, "tensorflow_version": tf.__version__,
            "batch_size": 32, "augmentation": "enhanced", "dropout": 0.25,
            "optimizer": "Adam", "learning_rate": 0.001,
            "calibration_seed": CALIBRATION_SEED,
            "calibration_training_indices": calibration_indices.tolist(),
            "primary_metric": "INT8 development-test accuracy; sample standard deviation (ddof=1)",
            "model_retention": "No per-seed model files; weights and exports used in memory only",
            "deployment_candidate": "Unchanged; no seed selected by its individual test score",
            "limitations": "Training variability on a fixed development benchmark, not new-data uncertainty",
        },
        "classes": CLASSES, "training_manifest": training_manifest,
        "development_test_manifest": test_manifest,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), ROOT / "src/training/train.py")},
        "runs": [],
    }
    reports.mkdir(parents=True)
    save_json(reports / "results.json", summary)
    images, test_images = load_images(train_paths), load_images(test_paths)
    for seed in SEEDS:
        print(f"SEED {seed}: training ten epochs; capture checkpoints at {EPOCHS}", flush=True)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = build_model(dropout=0.25, seed=seed)
        dataset = training_dataset(images, labels, 32, seed, enhanced=True)
        capture = CaptureCheckpoints(seed, reports / "history.csv")
        model.fit(dataset, epochs=10, callbacks=[capture], verbose=2)
        assert set(capture.snapshots) == set(EPOCHS)
        for epochs in EPOCHS:
            model.set_weights(capture.snapshots[epochs])
            probabilities = model.predict(test_images, verbose=0)
            pred_float = probabilities.argmax(axis=1)
            # The fixed seed selects the same 200 training images for all exports.
            float_bytes, int8_bytes = export_models(model, images, CALIBRATION_SEED)
            pred_int8, deployment = evaluate_tflite(int8_bytes, test_images)
            result = {
                "seed": seed, "epochs": epochs,
                "samples": {"training": len(labels), "development_test": len(test_labels)},
                "float": metrics(test_labels, pred_float),
                "float_test_loss": float(tf.keras.losses.sparse_categorical_crossentropy(
                    test_labels, probabilities).numpy().mean()),
                "int8": metrics(test_labels, pred_int8),
                "float_bytes": len(float_bytes), "int8_bytes": len(int8_bytes),
                "int8_sha256": hashlib.sha256(int8_bytes).hexdigest(), "deployment": deployment,
            }
            prediction_path = reports / "predictions.csv"
            exists = prediction_path.exists()
            with prediction_path.open("a", newline="") as file:
                writer = csv.writer(file)
                if not exists:
                    writer.writerow(["seed", "epochs", "path", "true_class", "float_prediction",
                                     "int8_prediction", *[f"float_probability_{c}" for c in CLASSES]])
                for i, p in enumerate(test_paths):
                    writer.writerow([seed, epochs, str(p.relative_to(ROOT)), CLASSES[test_labels[i]],
                                     CLASSES[pred_float[i]], CLASSES[pred_int8[i]],
                                     *[float(v) for v in probabilities[i]]])
            summary["runs"].append(result)
            save_json(reports / "results.json", summary)
            print(f"CHECKPOINT: seed={seed} epochs={epochs} float={result['float']['accuracy']:.6f} "
                  f"int8={result['int8']['accuracy']:.6f}", flush=True)
    summary.update({"status": "complete", "summary": aggregate(summary["runs"])})
    save_json(reports / "results.json", summary)
    print(f"SUMMARY: {json.dumps(summary['summary'])}", flush=True)


if __name__ == "__main__":
    main()
