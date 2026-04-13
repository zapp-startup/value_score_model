# Value Score Model

Supporting AI model package for the Zapp Django integration milestone.

This repository contains the reusable training, feature engineering, evaluation, and scoring code behind Zapp's value-score logic. The official submission-facing integration docs live in the Django backend repo:

- Primary Django repo: [backend](https://github.com/zapp-startup/backend)
- Supporting model repo: [value_score_model](https://github.com/zapp-startup/value_score_model)

## What This Package Does

The package implements a personalized value-scoring pipeline for subscriptions.

- generates or consumes training data
- engineers user, subscription, transaction, and feedback features
- routes samples across three scoring tiers
- saves model artifacts locally
- produces batch scoring outputs with evidence payloads

Core pieces already present in the repo:

- `train.py`
  entry point for local training and evaluation
- `models/`
  tiered scoring models, including cold-start, XGBoost, and neural paths
- `pipeline/`
  feature engineering, subscription signal preprocessing, and overutilisation logic
- `scoring/daily_batch.py`
  local batch scoring entry point
- `tests/test_pipeline.py`
  end-to-end and component-level validation

## Local Setup

From the parent `Zapp` directory:

```powershell
cd ..
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Train Locally

To run a local synthetic training pass:

```powershell
cd ..
python -m value_score_model.train --synthetic --n-users 50 --n-merchants 20
```

You can also point the trainer at prepared CSV exports:

```powershell
cd ..
python -m value_score_model.train --data-dir value_score_model/data --model-dir value_score_model/checkpoints
```

## Run Batch Scoring

After training and saving a checkpoint:

```powershell
cd ..
python -m value_score_model.scoring.daily_batch --model-dir value_score_model/checkpoints --data-dir value_score_model/data --output-dir value_score_model/outputs
```

## Data Preparation Helpers

The repo includes scripts for turning exported or generated data into the CSV layout expected by the model package.

- `scripts/build_training_data.py`
  builds normalized training CSVs from Django-style exports
- `scripts/fill_subscription_signals.py`
  fills subscription-level signal fields when needed

## Evaluation And Testing

The package already includes test coverage for:

- synthetic data generation
- feature engineering integrity
- overutilisation bonus logic
- cold-start, XGBoost, and neural scoring behavior
- end-to-end fit/predict round trips
- evaluation metrics such as MAE, RMSE, Spearman ranking, calibration, and NDCG

If you run training locally, `train.py` writes evaluation metrics to `eval_metrics.json` inside the chosen model directory.

## Artifact And Weight Policy

This repo is intentionally configured to avoid committing generated model artifacts.

- model checkpoints stay local
- downloaded weights stay local
- caches and outputs stay local

Ignored examples include:

- `*.pt`
- `*.bin`
- `*.safetensors`
- `*.pkl`
- checkpoint folders
- output folders

That keeps the repo lightweight and aligned with the assignment requirement to avoid committing large binary model files.

## Relationship To The Django Repo

This package is the model workbench and scoring foundation. The Django-side submission deliverables, API documentation, and Canvas submission file live in the backend repo, because that is the primary integrated application for the assignment.
