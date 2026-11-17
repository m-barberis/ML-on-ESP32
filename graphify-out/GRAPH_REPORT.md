# Graph Report - ESP32_ML  (2026-11-17)

## Corpus Check
- 41 files · ~109,453 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 61 file(s) not represented in the graph (top: (none) 28, .csv 27, .tflite 2)

## Summary
- 117 nodes · 257 edges · 11 communities (7 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `8b98b7b3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train.py
- seven_fold_epochs.py
- main.cpp
- Rock Paper Scissors selected deployment model
- README.md
- benchmark_esp32.py
- seed_comparison.py
- cross_validate.py
- CaptureCheckpoints
- dependencies
- AGENTS.md

## God Nodes (most connected - your core abstractions)
1. `main()` - 16 edges
2. `main()` - 15 edges
3. `main()` - 14 edges
4. `benchmark()` - 13 edges
5. `main()` - 13 edges
6. `collect_images()` - 11 edges
7. `load_images()` - 11 edges
8. `metrics()` - 11 edges
9. `save_json()` - 9 edges
10. `evaluate_tflite()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `benchmark()` --calls--> `collect_images()`  [EXTRACTED]
  scripts/benchmark_esp32.py → src/training/train.py
- `benchmark()` --calls--> `load_images()`  [EXTRACTED]
  scripts/benchmark_esp32.py → src/training/train.py
- `benchmark()` --calls--> `metrics()`  [EXTRACTED]
  scripts/benchmark_esp32.py → src/training/train.py
- `save_models()` --calls--> `export_models()`  [EXTRACTED]
  src/training/cross_validate.py → src/training/train.py
- `main()` --calls--> `collect_images()`  [EXTRACTED]
  src/training/cross_validate.py → src/training/train.py

## Import Cycles
- None detected.

## Communities (11 total, 4 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.42
Nodes (6): collect_images(), evaluate_tflite(), load_images(), main(), metrics(), split_training()

### Community 1 - "seven_fold_epochs.py"
Cohesion: 0.24
Nodes (8): save_models(), sequence(), append_predictions(), best_epoch(), CombinedHistory, fit(), main(), make_folds()

### Community 3 - "Rock Paper Scissors selected deployment model"
Cohesion: 0.29
Nodes (7): Artifact and input contract, Hardware validation still required, Intended use, Provenance and secondary experiments, Results, Rock Paper Scissors selected deployment model, Training and model selection

### Community 4 - "README.md"
Cohesion: 0.09
Nodes (21): Baseline, augmentation, and dropout, Commands and artifacts, Question and method, Results, Command and artifacts, Question and method, Results, Three-fold nested sequence validation (+13 more)

### Community 5 - "benchmark_esp32.py"
Cohesion: 0.22
Nodes (6): average_and_peak(), benchmark(), connect_benchmark(), infer_on_board(), main(), read_response()

### Community 6 - "seed_comparison.py"
Cohesion: 0.33
Nodes (5): aggregate(), describe(), main(), manifest(), export_models()

### Community 7 - "cross_validate.py"
Cohesion: 0.35
Nodes (7): fit(), main(), make_folds(), save_predictions(), build_model(), save_json(), training_dataset()

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Rock Paper Scissors selected deployment model` connect `Rock Paper Scissors selected deployment model` to `README.md`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Should `main.cpp` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._
- **Should `README.md` be split into smaller, more focused modules?**
  _Cohesion score 0.08547008547008547 - nodes in this community are weakly interconnected._