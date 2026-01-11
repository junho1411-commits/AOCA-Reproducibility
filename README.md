# AOCA Reproducibility Package (IEEE Access)

This repository provides **lightweight, fully reproducible reference code**
for the experiments reported in the paper:

> *An Agent-Oriented Context Architecture for Cost-Efficient Enterprise Document Automation*

The implementation is intentionally **simulation-based and deterministic**
to enable independent verification without access to proprietary datasets,
large language models, or external APIs.

---

## Scope and Design Principles

- This code validates **relative efficiency trends**, not absolute performance
- All experiments use **normalized cost and latency models**
- No external dependencies are required
- Each script reproduces **one table or figure** in the paper

---

## How to Run

All experiments can be executed independently:

```bash
python3 experiment1_cost_efficiency.py
python3 experiment2_dataset_efficiency.py
python3 experiment3_context_scaling.py
python3 experiment4_ablation.py
python3 experiment5_scalability_latency.py
