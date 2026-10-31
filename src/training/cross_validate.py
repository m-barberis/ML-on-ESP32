"""Three-fold sequence-held-out evaluation, then refit one final deployment model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from train import (
    CLASSES, ROOT, build_model, collect_images, evaluate_tflite, export_models,
    load_images, metrics, np, save_json, tf, training_dataset,
)


def sequence(path):
    """Filename prefix identifies a rendering sequence, not verified hand identity."""
    return path.stem.rsplit("-", 1)[0]


def make_folds(paths, labels, seed):
    groups = np.array([sequence(path) for path in paths])
    rng = np.random.default_rng(seed)
    held_out_groups = [[] for _ in range(3)]
    for label in range(len(CLASSES)):
        class_groups = np.unique(groups[labels == label])
        if len(class_groups) < 4:
            raise ValueError("Need at least four sequences per class for outer and inner splits")
        for fold, chunk in enumerate(np.array_split(rng.permutation(class_groups), 3)):
            held_out_groups[fold].extend(chunk.tolist())

    folds = []
    for fold, held_out in enumerate(held_out_groups):
        outer_mask = np.isin(groups, held_out)
        outer_train = np.flatnonzero(~outer_mask)
        outer_test = np.flatnonzero(outer_mask)
        inner_groups = []
        for label in range(len(CLASSES)):
            available = np.unique(groups[(labels == label) & ~outer_mask])
            inner_groups.append(str(rng.choice(available)))
        inner_mask = np.isin(groups, inner_groups)
        inner_validation = np.flatnonzero(inner_mask & ~outer_mask)
        inner_train = np.flatnonzero(~inner_mask & ~outer_mask)
        for a, b in [(inner_train, inner_validation), (outer_train, outer_test)]:
            assert not set(groups[a]) & set(groups[b])
            assert set(labels[a]) == set(labels[b]) == {0, 1, 2}
        folds.append((inner_train, inner_validation, outer_train, outer_test))
    outer_indices = np.concatenate([fold[3] for fold in folds])
    assert sorted(outer_indices.tolist()) == list(range(len(paths)))
    return folds


def fit(images, labels, indices, seed, epochs, history_path, validation_indices=None):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(dropout=0.25, seed=seed)
    dataset = training_dataset(images[indices], labels[indices], 32, seed, enhanced=True)
    callbacks = [tf.keras.callbacks.CSVLogger(str(history_path))]
    validation = None
    if validation_indices is not None:
        validation = tf.data.Dataset.from_tensor_slices(
            (images[validation_indices], labels[validation_indices]),
        ).batch(32)
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True,
        ))
    history = model.fit(dataset, validation_data=validation, epochs=epochs,
                        callbacks=callbacks, verbose=2)
    selected_epoch = (int(np.argmax(history.history["val_accuracy"])) + 1
                      if validation is not None else epochs)
    return model, selected_epoch


def save_models(model, calibration_images, run, seed):
    float_dir, int8_dir = ROOT / "models/float" / run, ROOT / "models/int8" / run
    float_dir.mkdir(parents=True, exist_ok=True)
    int8_dir.mkdir(parents=True, exist_ok=True)
    model.save(float_dir / "rps_cnn.keras")
    float_bytes, int8_bytes = export_models(model, calibration_images, seed)
    (float_dir / "rps_cnn.tflite").write_bytes(float_bytes)
    (int8_dir / "rps_cnn.tflite").write_bytes(int8_bytes)
    return float_bytes, int8_bytes


def save_predictions(path, paths, labels, float_predictions, int8_predictions):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "true_class", "float_prediction", "int8_prediction"])
        for p, truth, f, q in zip(paths, labels, float_predictions, int8_predictions):
            writer.writerow([str(p.relative_to(ROOT)), CLASSES[truth], CLASSES[f], CLASSES[q]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reports = ROOT / "reports/metrics/grouped_cv"
    final_reports = ROOT / "reports/metrics/final"
    if any(path.exists() for path in [reports, final_reports, ROOT / "models/float/final",
                                      ROOT / "models/int8/final"]):
        parser.error("Grouped CV/final outputs already exist; preserve them before a new run")
    reports.mkdir(parents=True)
    final_reports.mkdir(parents=True)
    config = {
        "seed": args.seed, "folds": 3, "maximum_inner_epochs": 30,
        "early_stopping": {"monitor": "val_accuracy", "patience": 7},
        "group_definition": "Filename prefix before final hyphen; not verified hand identity",
        "inner_validation": "One complete sequence per class from outer training only",
        "outer_refit": "Fresh model on all outer training data for inner-selected epochs",
        "final_epochs": "Median of the three inner-selected epoch counts",
        "batch_size": 32, "augmentation": "enhanced", "dropout": 0.25,
        "selection_rule": "Fixed architecture; no choice based on official test or outer scores",
    }
    save_json(reports / "config.json", config)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    paths, labels = collect_images(ROOT / "data/raw/rps")
    images = load_images(paths)
    folds = make_folds(paths, labels, args.seed)
    out_of_fold_float = np.full(len(paths), -1, dtype=np.int32)
    out_of_fold_int8 = np.full(len(paths), -1, dtype=np.int32)
    fold_results = []
    for fold, (inner_train, inner_validation, outer_train, outer_test) in enumerate(folds):
        fold_dir = reports / f"fold_{fold + 1}"
        fold_dir.mkdir()
        seed = args.seed + fold
        save_json(fold_dir / "split.json", {
            name: [str(paths[i].relative_to(ROOT)) for i in indices]
            for name, indices in [("inner_training", inner_train),
                                  ("inner_validation", inner_validation),
                                  ("outer_training", outer_train), ("outer_test", outer_test)]
        })
        print(f"FOLD {fold + 1}: selecting epochs with inner validation", flush=True)
        _, epochs = fit(images, labels, inner_train, seed, 30,
                        fold_dir / "inner_history.csv", inner_validation)
        print(f"FOLD {fold + 1}: refitting outer training for {epochs} epochs", flush=True)
        model, _ = fit(images, labels, outer_train, seed, epochs, fold_dir / "history.csv")
        float_bytes, int8_bytes = save_models(model, images[outer_train],
                                             f"grouped_cv/fold_{fold + 1}", seed)
        pred_float = model.predict(images[outer_test], verbose=0).argmax(axis=1)
        pred_int8, deployment = evaluate_tflite(int8_bytes, images[outer_test])
        out_of_fold_float[outer_test], out_of_fold_int8[outer_test] = pred_float, pred_int8
        result = {
            "fold": fold + 1, "selected_epochs": epochs,
            "samples": {"inner_train": len(inner_train), "inner_validation": len(inner_validation),
                        "outer_train": len(outer_train), "outer_test": len(outer_test)},
            "float": metrics(labels[outer_test], pred_float),
            "int8": metrics(labels[outer_test], pred_int8),
            "float_bytes": len(float_bytes), "int8_bytes": len(int8_bytes),
            "deployment": deployment, "model_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        }
        save_json(fold_dir / "results.json", result)
        save_predictions(fold_dir / "predictions.csv", [paths[i] for i in outer_test],
                         labels[outer_test], pred_float, pred_int8)
        fold_results.append(result)
        print(f"FOLD RESULT: {json.dumps(result)}", flush=True)

    assert np.all(out_of_fold_float >= 0) and np.all(out_of_fold_int8 >= 0)
    final_epochs = int(np.median([r["selected_epochs"] for r in fold_results]))
    summary = {"config": config, "classes": CLASSES, "folds": fold_results,
               "final_epochs": final_epochs}
    for kind, predictions in [("float", out_of_fold_float), ("int8", out_of_fold_int8)]:
        accuracies = [r[kind]["accuracy"] for r in fold_results]
        summary[kind] = {
            "fold_mean_accuracy": float(np.mean(accuracies)),
            "fold_sample_std_accuracy": float(np.std(accuracies, ddof=1)),
            "pooled_out_of_fold": metrics(labels, predictions),
        }
    save_json(reports / "results.json", summary)
    save_predictions(reports / "predictions.csv", paths, labels,
                     out_of_fold_float, out_of_fold_int8)

    # Duration is fixed before reading the official test set. No early stopping
    # on the final fit: every official training image contributes to its weights.
    print(f"FINAL: refitting all {len(labels)} training images for {final_epochs} epochs", flush=True)
    save_json(final_reports / "config.json", {**config, "epochs": final_epochs,
                                             "training_samples": len(labels)})
    model, _ = fit(images, labels, np.arange(len(labels)), args.seed, final_epochs,
                   final_reports / "history.csv")
    float_bytes, int8_bytes = save_models(model, images, "final", args.seed)
    test_paths, test_labels = collect_images(ROOT / "data/raw/rps-test-set")
    test_images = load_images(test_paths)
    pred_float = model.predict(test_images, verbose=0).argmax(axis=1)
    pred_int8, deployment = evaluate_tflite(int8_bytes, test_images)
    final_result = {
        "tensorflow_version": tf.__version__, "classes": CLASSES,
        "parameters": model.count_params(), "epochs": final_epochs,
        "epoch_selection": "Median inner-validation best epoch across three grouped folds",
        "samples": {"train": len(labels), "test": len(test_labels)},
        "float": {"test": metrics(test_labels, pred_float), "tflite_bytes": len(float_bytes)},
        "int8": {"test": metrics(test_labels, pred_int8), "tflite_bytes": len(int8_bytes)},
        "deployment": deployment, "model_sha256": hashlib.sha256(int8_bytes).hexdigest(),
        "limitations": [
            "Sequence prefixes are not verified hand identities; grouped CV does not prove identity separation.",
            "Outer folds evaluate smaller training sets; their mean is not the final model's measured accuracy.",
            "Official test is a development benchmark after earlier experiments, not an untouched final test.",
            "Real-camera accuracy, ESP32 tensor arena, camera RAM and latency are unverified.",
        ],
    }
    save_json(final_reports / "results.json", final_result)
    save_predictions(final_reports / "predictions.csv", test_paths, test_labels,
                     pred_float, pred_int8)
    print(f"CV SUMMARY: {json.dumps(summary)}", flush=True)
    print(f"FINAL RESULT: {json.dumps(final_result)}", flush=True)


if __name__ == "__main__":
    main()
