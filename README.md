# Credit Score Prediction — Local Deployment

Individual project: an earlier iteration of the credit-tier classifier (Poor / Standard /
Good), with its own end-to-end pipeline and a local Streamlit app.

## Open this first
- [`eda_notebook.ipynb`](./eda_notebook.ipynb) — EDA, preprocessing, and model training.
- [`app.py`](./app.py) — the Streamlit app.

## Pipeline
- `train.py` / `pipeline.py` — training pipeline.
- `split.py` — train/test/validation split.
- `evaluation.py` — held-out evaluation.
- `data_ingestion.py` — data loading utilities.
- `artifacts/` — the trained pipeline.

No dataset is committed here — see the notebook for the source.
