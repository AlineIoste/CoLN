# CoLN — Collaborative Learning with Non-convex aggregation

This repository provides a reference implementation of **CoLN** (Collaborative Learning with Non-convex aggregation) as described in the accompanying paper.

## What’s in this repo

- `coln/aggregation.py`: CoLN aggregation (`combined_learning_coln`)
- `coln/trainer.py`: a runnable federated loop (`run_federated_coln`) using Keras models
- `coln/data.py`: utilities to build the `clients` dictionary from a pandas DataFrame
- `examples/toy_aggregation.py`: end-to-end synthetic example (no private data)

## Installation (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Quickstart (synthetic demo)

```bash
python examples/toy_aggregation.py
```

## Using your own dataset

Provide a pandas DataFrame `df` with:
- a binary target column (e.g., `icu_admission`)
- a hospital/site column used to split clients (e.g., `source_hospital`)

```python
from coln.data import prepare_clients_from_dataframe
from coln.trainer import run_federated_coln

clients, scaler, feature_cols = prepare_clients_from_dataframe(
    df,
    target_col="icu_admission",
    hospital_col="source_hospital",
)

global_model, history, final_metrics = run_federated_coln(clients, rounds=15)
```

## Citation

Add your BibTeX here (and keep `CITATION.cff` updated).

## License

Choose a license (MIT/Apache-2.0/etc.) and update `LICENSE`.
