"""Select epochs with seven sequence-held-out folds (validation loss by default)."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from cross_validate import save_models, sequence
from train import (
    CLASSES, ROOT, build_model, collect_images, evaluate_tflite, load_images,
    metrics, np, save_json, tf, training_dataset,
)


def make_folds(paths, labels, seed):
    groups = np.array([sequence(path) for path in paths])
    rng = np.random.default_rng(seed)
    held_out = [[] for _ in range(7)]
    for label in range(len(CLASSES)):
        class_groups = np.unique(groups[labels == label])
        if len(class_groups) != 7:
            raise ValueError("This experiment requires exactly seven sequences per class")
        for fold, group in enumerate(rng.permutation(class_groups)):
            held_out[fold].append(str(group))
    folds = []
    for validation_groups in held_out:
        mask = np.isin(groups, validation_groups)
        training, validation = np.flatnonzero(~mask), np.flatnonzero(mask)
        assert not set(groups[training]) & set(groups[validation])
        assert set(labels[training]) == set(labels[validation]) == {0, 1, 2}
        for label in range(len(CLASSES)):
            assert len(np.unique(groups[validation][labels[validation] == label])) == 1
        folds.append((training, validation))
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(len(paths)))
    return folds


def best_epoch(history, monitor):
    select = np.argmin if monitor == "val_loss" else np.argmax
    return int(select(history[monitor])) + 1


class CombinedHistory(tf.keras.callbacks.Callback):
    """Stream all folds and the final fit into one CSV, including partial runs."""

    def __init__(self, path, stage, fold):
        super().__init__()
        self.path, self.stage, self.fold = path, stage, fold

    def on_epoch_end(self, epoch, logs=None):
        columns = ["stage", "fold", "epoch", "accuracy", "loss", "val_accuracy", "val_loss"]
        exists = self.path.exists()
        with self.path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            if not exists:
                writer.writeheader()
            writer.writerow({"stage": self.stage, "fold": self.fold, "epoch": epoch + 1,
                             **{k: logs[k] for k in columns[3:] if k in logs}})


def fit(images, labels, training, seed, epochs, history_path, stage, fold,
        monitor, validation_indices=None):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(dropout=0.25, seed=seed)
    dataset = training_dataset(images[training], labels[training], 32, seed, enhanced=True)
    callbacks = [CombinedHistory(history_path, stage, fold)]
    validation = None
    if validation_indices is not None:
        validation = tf.data.Dataset.from_tensor_slices(
            (images[validation_indices], labels[validation_indices]),
        ).batch(32)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="min" if monitor == "val_loss" else "max",
            patience=7, min_delta=0, restore_best_weights=True,
        ))
    history = model.fit(dataset, validation_data=validation, epochs=epochs,
                        callbacks=callbacks, verbose=2).history
    selected = best_epoch(history, monitor) if validation is not None else epochs
    return model, selected, history


def append_predictions(path, paths, labels, float_predictions, stage, fold, int8_predictions=None):
    columns = ["stage", "fold", "path", "true_class", "float_prediction", "int8_prediction"]
    exists = path.exists()
    with path.open("a", newline="") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(columns)
        for i, p in enumerate(paths):
            writer.writerow([stage, fold, str(p.relative_to(ROOT)), CLASSES[labels[i]],
                             CLASSES[float_predictions[i]],
                             "" if int8_predictions is None else CLASSES[int8_predictions[i]]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--monitor", choices=("val_accuracy", "val_loss"), default="val_loss")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    run = args.run or ("seven_fold_loss" if args.monitor == "val_loss" else "seven_fold_epochs")
    if Path(run).name != run or run in {".", ".."}:
        parser.error("run must be a single folder name")
    reports = ROOT / "reports/metrics" / run
    outputs = [reports, ROOT / "models/float" / run, ROOT / "models/int8" / run]
    if any(path.exists() for path in outputs):
        parser.error("Run outputs already exist; choose a fresh run name")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    paths, labels = collect_images(ROOT / "data/raw/rps")
    folds = make_folds(paths, labels, args.seed)
    config = {
        "run": run, "seed": args.seed, "folds": 7,
        "tensorflow_version": tf.__version__, "maximum_epochs": 30,
        "early_stopping": {"monitor": args.monitor, "patience": 7, "min_delta": 0,
                           "mode": "min" if args.monitor == "val_loss" else "max",
                           "restore_best_weights": True, "ties": "first optimum"},
        "validation": "One complete sequence per class per fold; each sequence held out once",
        "fold_seeds": [args.seed + i for i in range(7)],
        "group_definition": "Filename prefix before final hyphen; not verified hand identity",
        "batch_size": 32, "augmentation": "enhanced", "dropout": 0.25,
        "optimizer": "Adam", "learning_rate": 0.001,
        "final_epochs": f"Median of seven best epochs selected using {args.monitor}",
        "score_interpretation": "Fold scores inform epoch selection; not independent evaluation",
        "official_test_role": "Development benchmark only, evaluated after final training",
    }
    summary = {
        "status": "running", "config": config, "classes": CLASSES,
        "training_manifest": [{"path": str(p.relative_to(ROOT)),
                               "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in paths],
        "splits": [{"fold": i + 1, "seed": args.seed + i,
                    "training_indices": training.tolist(),
                    "validation_indices": validation.tolist(),
                    "validation_sequences": sorted({sequence(paths[j]) for j in validation})}
                   for i, (training, validation) in enumerate(folds)],
        "folds": [],
    }
    reports.mkdir(parents=True)
    save_json(reports / "results.json", summary)
    images = load_images(paths)
    predictions = np.full(len(labels), -1, dtype=np.int32)
    for fold, (training, validation) in enumerate(folds, start=1):
        seed = args.seed + fold - 1
        print(f"FOLD {fold}/7: {len(training)} training, {len(validation)} validation", flush=True)
        model, epochs, history = fit(
            images, labels, training, seed, 30, reports / "history.csv",
            "validation", fold, args.monitor, validation,
        )
        pred = model.predict(images[validation], verbose=0).argmax(axis=1)
        predictions[validation] = pred
        result = {
            "fold": fold, "seed": seed,
            "samples": {"training": len(training), "validation": len(validation)},
            "completed_epochs": len(history["loss"]), "selected_epoch": epochs,
            "selected_validation_accuracy": history["val_accuracy"][epochs - 1],
            "selected_validation_loss": history["val_loss"][epochs - 1],
            "minimum_validation_loss_epoch": best_epoch(history, "val_loss"),
            "maximum_validation_accuracy_epoch": best_epoch(history, "val_accuracy"),
            "reached_epoch_cap": len(history["loss"]) == 30,
            "float_validation": metrics(labels[validation], pred),
        }
        append_predictions(reports / "predictions.csv", [paths[i] for i in validation],
                           labels[validation], pred, "validation", fold)
        summary["folds"].append(result)
        save_json(reports / "results.json", summary)
        print(f"FOLD RESULT: {json.dumps(result)}", flush=True)

    assert np.all(predictions >= 0)
    selected = [r["selected_epoch"] for r in summary["folds"]]
    final_epochs = int(np.median(selected))
    summary.update({"selected_epochs": selected, "final_epochs": final_epochs,
                    "pooled_float_validation": metrics(labels, predictions)})
    save_json(reports / "results.json", summary)
    print(f"FINAL: selected epochs {selected}; training all {len(labels)} images for {final_epochs} epochs", flush=True)
    model, _, _ = fit(images, labels, np.arange(len(labels)), args.seed, final_epochs,
                      reports / "history.csv", "final", "", args.monitor)
    float_bytes, int8_bytes = save_models(model, images, f"{run}/final", args.seed)
    test_paths, test_labels = collect_images(ROOT / "data/raw/rps-test-set")
    test_images = load_images(test_paths)
    probabilities = model.predict(test_images, verbose=0)
    pred_float = probabilities.argmax(axis=1)
    pred_int8, deployment = evaluate_tflite(int8_bytes, test_images)
    final_result = {
        "epochs": final_epochs, "selected_epochs": selected,
        "samples": {"train": len(labels), "development_test": len(test_labels)},
        "float": {"test": metrics(test_labels, pred_float), "bytes": len(float_bytes),
                  "test_loss": float(tf.keras.losses.sparse_categorical_crossentropy(
                      test_labels, probabilities).numpy().mean())},
        "int8": {"test": metrics(test_labels, pred_int8), "bytes": len(int8_bytes)},
        "deployment": deployment, "model_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        "limitations": [config["score_interpretation"], config["official_test_role"],
                        "Sequence prefixes are not verified hand identities",
                        "One seed per fold; fold variation also includes training randomness"],
    }
    append_predictions(reports / "predictions.csv", test_paths, test_labels,
                       pred_float, "development_test", "", pred_int8)
    summary.update({"status": "complete", "final": final_result})
    save_json(reports / "results.json", summary)
    print(f"FINAL RESULT: {json.dumps(final_result)}", flush=True)


if __name__ == "__main__":
    main()
