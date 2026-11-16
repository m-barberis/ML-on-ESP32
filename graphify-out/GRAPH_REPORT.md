# Graph Report - ESP32_ML  (2026-11-16)

## Corpus Check
- 40 files · ~108,459 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 62 file(s) not represented in the graph (top: (none) 28, .csv 27, .tflite 2)

## Summary
- 98 nodes · 211 edges · 13 communities (9 shown, 4 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS · INFERRED: 1 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d83afa40`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- cross_validate.py
- seven_fold_epochs.py
- Rock Paper Scissors selected deployment model
- Baseline, augmentation, and dropout
- seed_comparison.py
- ML-on-ESP32
- Three-fold nested sequence validation
- Training-seed comparison: seven versus ten epochs
- Seven-fold epoch selection
- dependencies
- AGENTS.md

## God Nodes (most connected - your core abstractions)
1. `main()` - 16 edges
2. `main()` - 15 edges
3. `main()` - 14 edges
4. `main()` - 13 edges
5. `save_json()` - 9 edges
6. `collect_images()` - 9 edges
7. `load_images()` - 9 edges
8. `metrics()` - 9 edges
9. `evaluate_tflite()` - 9 edges
10. `build_model()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `save_models()` --calls--> `export_models()`  [EXTRACTED]
  src/training/cross_validate.py → src/training/train.py
- `main()` --calls--> `save_json()`  [EXTRACTED]
  src/training/cross_validate.py → src/training/train.py
- `main()` --calls--> `build_model()`  [EXTRACTED]
  src/training/seed_comparison.py → src/training/train.py
- `main()` --calls--> `collect_images()`  [EXTRACTED]
  src/training/seed_comparison.py → src/training/train.py
- `main()` --calls--> `evaluate_tflite()`  [EXTRACTED]
  src/training/seed_comparison.py → src/training/train.py

## Import Cycles
- None detected.

## Communities (13 total, 4 thin omitted)

### Community 0 - "cross_validate.py"
Cohesion: 0.27
Nodes (11): fit(), main(), save_predictions(), build_model(), collect_images(), evaluate_tflite(), load_images(), main() (+3 more)

### Community 1 - "seven_fold_epochs.py"
Cohesion: 0.21
Nodes (10): make_folds(), save_models(), sequence(), append_predictions(), best_epoch(), CombinedHistory, fit(), main() (+2 more)

### Community 3 - "Rock Paper Scissors selected deployment model"
Cohesion: 0.29
Nodes (7): Artifact and input contract, Hardware validation still required, Intended use, Provenance and secondary experiments, Results, Rock Paper Scissors selected deployment model, Training and model selection

### Community 4 - "Baseline, augmentation, and dropout"
Cohesion: 0.40
Nodes (4): Baseline, augmentation, and dropout, Commands and artifacts, Question and method, Results

### Community 6 - "seed_comparison.py"
Cohesion: 0.24
Nodes (6): aggregate(), CaptureCheckpoints, describe(), main(), manifest(), export_models()

### Community 7 - "ML-on-ESP32"
Cohesion: 0.40
Nodes (5): ESP32 integration, Final model, ML-on-ESP32, Secondary experiments, Setup and reproduce training

### Community 8 - "Three-fold nested sequence validation"
Cohesion: 0.50
Nodes (4): Command and artifacts, Question and method, Results, Three-fold nested sequence validation

### Community 9 - "Training-seed comparison: seven versus ten epochs"
Cohesion: 0.50
Nodes (4): Question and method, Reproduction and verification, Results, Training-seed comparison: seven versus ten epochs

### Community 10 - "Seven-fold epoch selection"
Cohesion: 0.50
Nodes (4): Commands and artifacts, Question and method, Results, Seven-fold epoch selection

## Knowledge Gaps
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Rock Paper Scissors selected deployment model` connect `Rock Paper Scissors selected deployment model` to `README.md`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `ML-on-ESP32` connect `ML-on-ESP32` to `README.md`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._