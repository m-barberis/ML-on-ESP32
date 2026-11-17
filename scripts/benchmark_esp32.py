"""Stream the official RPS test set to the ESP32 and save benchmark results."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import monotonic
import zlib

import numpy as np
import serial
import tensorflow as tf

from src.training.train import CLASSES, IMAGE_SIZE, ROOT, collect_images, load_images, metrics

MODEL = ROOT / "models/int8/final/rps_cnn.tflite"
TEST_SET = ROOT / "data/raw/rps-test-set"
BAUD = 115200
SAMPLES_PER_CLASS = 124


def read_response(connection, expected, timeout):
    deadline = monotonic() + timeout
    pending = bytearray()
    while monotonic() < deadline:
        chunk = connection.read_until(size=512)
        pending.extend(chunk)
        if not chunk.endswith(b"\n"):
            if len(pending) >= 512:
                pending.clear()
            continue
        fields = pending.decode("ascii", errors="replace").split()
        pending.clear()
        if fields[:2] == ["RPS", "ERROR"]:
            raise RuntimeError(f"ESP32 reported {' '.join(fields[2:])}")
        if fields[:2] == ["RPS", expected]:
            return fields[2:]
    raise TimeoutError(f"No RPS {expected} response on {connection.port}")


def connect_benchmark(connection, model_bytes):
    # USB serial may reset the board on open. Poll until its startup completes.
    for _ in range(10):
        connection.write(b"?")
        try:
            response = read_response(connection, "READY", timeout=1)
            break
        except TimeoutError:
            continue
    else:
        raise RuntimeError("ESP32 benchmark did not respond. Confirm the firmware and reset the board.")

    version, model_crc, arena_capacity, heap_capacity, cpu_mhz = response
    if version != "1":
        raise RuntimeError(f"Unsupported firmware protocol version {version}")
    if int(model_crc, 16) != zlib.crc32(model_bytes):
        raise RuntimeError("The flashed model differs from the local model. Rebuild and upload.")
    return {
        "protocol_version": int(version),
        "model_crc32": model_crc,
        "arena_reserved_bytes": int(arena_capacity),
        "internal_heap_capacity_bytes": int(heap_capacity),
        "configured_cpu_mhz": int(cpu_mhz),
    }


def infer_on_board(connection, pixels):
    started = monotonic()
    connection.write(b"I" + pixels)
    response = read_response(connection, "RESULT", timeout=8)
    round_trip_ms = 1000 * (monotonic() - started)
    image_crc, *values = response
    if int(image_crc, 16) != zlib.crc32(pixels):
        raise RuntimeError("The image received by the ESP32 differs from the sent image.")
    paper, rock, scissors, invoke_us, arena_used, heap_after, heap_peak = map(int, values)
    scores = [paper, rock, scissors]
    if not all(-128 <= score <= 127 for score in scores):
        raise RuntimeError("The ESP32 sent scores outside the INT8 range.")
    return {
        **{f"{label}_int8": score for label, score in zip(CLASSES, scores)},
        "prediction": CLASSES[int(np.argmax(scores))],
        "invoke_us": invoke_us,
        "serial_round_trip_ms": round_trip_ms,
        "arena_used_bytes": arena_used,
        "internal_heap_used_after_bytes": heap_after,
        "internal_heap_peak_estimate_bytes": heap_peak,
    }


def average_and_peak(values):
    values = list(values)
    return {"average": mean(values), "peak": max(values)}


def benchmark(port, output):
    paths, labels = collect_images(TEST_SET)
    if np.bincount(labels).tolist() != [SAMPLES_PER_CLASS] * len(CLASSES):
        raise ValueError(f"Expected {SAMPLES_PER_CLASS} PNGs per class in {TEST_SET}")

    model_bytes = MODEL.read_bytes()
    reference = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    reference.allocate_tensors()
    input_info = reference.get_input_details()[0]
    output_info = reference.get_output_details()[0]
    if (tuple(input_info["shape"]) != (1, IMAGE_SIZE, IMAGE_SIZE, 3)
            or input_info["dtype"] != np.int8 or output_info["dtype"] != np.int8):
        raise ValueError("Unexpected model input or output")
    scale, zero_point = input_info["quantization"]

    rows = []
    desktop_predictions = []
    with serial.Serial(port, BAUD, timeout=0.25, write_timeout=5, exclusive=True) as connection:
        board = connect_benchmark(connection, model_bytes)
        output.mkdir(parents=True, exist_ok=False)
        print(f"Saving per-image results to {output / 'predictions.csv'}", flush=True)
        started = monotonic()
        with (output / "predictions.csv").open("w", newline="") as stream:
            writer = None
            for index, (path, label) in enumerate(zip(paths, labels), start=1):
                image = load_images([path])
                tensor = np.clip(np.rint(image / scale + zero_point), -128, 127).astype(np.int8)
                reference.set_tensor(input_info["index"], tensor)
                reference.invoke()
                desktop = int(reference.get_tensor(output_info["index"])[0].argmax())
                try:
                    result = infer_on_board(connection, tensor.tobytes())
                except (OSError, RuntimeError, ValueError) as error:
                    raise RuntimeError(f"Image {index}/{len(paths)} ({path.name}): {error}") from error
                row = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "true_class": CLASSES[label],
                    "desktop_prediction": CLASSES[desktop],
                    **result,
                }
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                desktop_predictions.append(desktop)
                if index % 25 == 0 or index == len(paths):
                    print(f"{index}/{len(paths)} images", flush=True)
        wall_seconds = monotonic() - started

    predictions = np.asarray([CLASSES.index(row["prediction"]) for row in rows])
    report = {
        "test_set": TEST_SET.relative_to(ROOT).as_posix(),
        "images": len(rows),
        "class_order": CLASSES,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "serial_port": port,
        "baud": BAUD,
        "board": board,
        "classification": {
            "correct": int(np.count_nonzero(labels == predictions)),
            **metrics(labels, predictions),
            "desktop_accuracy": float(np.mean(labels == desktop_predictions)),
            "desktop_prediction_agreement": float(np.mean(predictions == desktop_predictions)),
        },
        "latency": {
            "invoke_ms": average_and_peak(row["invoke_us"] / 1000 for row in rows),
            "serial_round_trip_ms": average_and_peak(row["serial_round_trip_ms"] for row in rows),
            "pass_wall_seconds": wall_seconds,
        },
        "memory": {
            "arena_used_bytes": average_and_peak(row["arena_used_bytes"] for row in rows),
            "internal_heap_used_bytes": {
                "average_after_invoke": mean(row["internal_heap_used_after_bytes"] for row in rows),
                "peak_estimate_during_invoke": max(row["internal_heap_peak_estimate_bytes"] for row in rows),
            },
        },
    }
    summary = json.dumps(report, indent=2) + "\n"
    (output / "summary.json").write_text(summary, encoding="utf-8")
    print(summary, end="")


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="ESP32 serial port from pio device list")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/esp32" / stamp,
                        help="New directory for predictions.csv and summary.json")
    args = parser.parse_args()
    try:
        benchmark(args.port, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Benchmark stopped: {error}\n")


if __name__ == "__main__":
    main()
