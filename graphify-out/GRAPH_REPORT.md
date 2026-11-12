# Graph Report - ESP32_ML  (2026-11-12)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 50 nodes · 163 edges · 11 communities (4 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `498646e9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cross_validate.py
- seven_fold_epochs.py
- main
- main
- collect_images
- seed_comparison.py
- CombinedHistory
- sequence
- CaptureCheckpoints
- export_models

## God Nodes (most connected - your core abstractions)
1. `main()` - 16 edges
2. `main()` - 15 edges
3. `main()` - 14 edges
4. `main()` - 13 edges
5. `load_images()` - 9 edges
6. `metrics()` - 9 edges
7. `save_json()` - 9 edges
8. `collect_images()` - 9 edges
9. `evaluate_tflite()` - 9 edges
10. `build_model()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `build_model()`  [EXTRACTED]
  src/training/seed_comparison.py → src/training/train.py
- `fit()` --calls--> `build_model()`  [EXTRACTED]
  src/training/seven_fold_epochs.py → src/training/train.py
- `main()` --calls--> `training_dataset()`  [EXTRACTED]
  src/training/seed_comparison.py → src/training/train.py
- `fit()` --calls--> `training_dataset()`  [EXTRACTED]
  src/training/seven_fold_epochs.py → src/training/train.py
- `save_models()` --calls--> `export_models()`  [EXTRACTED]
  src/training/cross_validate.py → src/training/train.py

## Import Cycles
- None detected.

## Communities (11 total, 7 thin omitted)

### Community 0 - "cross_validate.py"
Cohesion: 0.36
Nodes (4): fit(), save_predictions(), build_model(), training_dataset()

### Community 1 - "seven_fold_epochs.py"
Cohesion: 0.46
Nodes (6): save_models(), append_predictions(), best_epoch(), fit(), main(), make_folds()

### Community 2 - "main"
Cohesion: 0.60
Nodes (5): main(), load_images(), main(), metrics(), save_json()

### Community 4 - "collect_images"
Cohesion: 0.50
Nodes (3): collect_images(), evaluate_tflite(), split_training()

## Knowledge Gaps
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CombinedHistory` connect `CombinedHistory` to `seven_fold_epochs.py`?**
  _High betweenness centrality (0.120) - this node is a cross-community bridge._
- **Why does `CaptureCheckpoints` connect `CaptureCheckpoints` to `main`, `seed_comparison.py`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `cross_validate.py`, `main`, `collect_images`, `seed_comparison.py`, `CaptureCheckpoints`, `export_models`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._