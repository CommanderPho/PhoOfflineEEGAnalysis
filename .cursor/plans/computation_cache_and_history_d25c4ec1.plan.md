---
name: Computation cache and history
overview: Add a disk-backed cache and a computation history table so that per-file EEG results (from EEGComputations.run_all) can be reused when the same XDF file and parameters are processed again, avoiding redundant spectrogram/CWT/topo computation.
todos: []
isProject: false
---

# Caching system for main_analyze_run.py

## Current behavior

In [main_analy