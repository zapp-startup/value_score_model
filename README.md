# Value Score Model

## Synthetic evaluation harness

Stress-test scoring on many synthetic users and export CSV / JSONL for review.

```bash
# From the directory that contains the `value_score_model` package folder:
python value_score_model/scripts/simulate_and_export_value_scores.py --n-users 2000 --seed 42 --months-history 6 --outdir outputs
```

Outputs:

- `outputs/value_score_eval_2000.csv` — user-level rollup (mean `value_score` across that user’s active subscriptions)
- `outputs/value_score_eval_2000_subscriptions.csv` — **one row per scored subscription** (this is what `ValueScoreModel.predict` outputs: `value_score`, `base_value_score`, `overutilisation_bonus`, tier, features, merchant, price, etc.)
- `outputs/value_score_eval_2000_transactions.csv` — **one row per generated transaction**, with the **subscription-level** scores joined on (`user_id`, `merchant_id`, `subscription_id`). Same score repeats for every txn on that sub so you can audit txn context; the model does **not** assign a distinct score per transaction unless you add a new scoring path.
- `outputs/value_score_eval_2000.jsonl` — one JSON object per user (includes `subscriptions_scored` with per-sub scores)
- `outputs/value_score_eval_summary.txt` — two blocks: user-mean **display** (`value_score`) and user-mean **learned base** (`base_value_score`); histograms use `scoring.display_max` and `scoring.learned_score_max`
- `outputs/value_score_eval_manifest.json` — machine-readable score contract (grains, scales, output paths)

### Score contract (subscription model)

- **`base_value_score`** — learned tiers, **0..`learned_score_max`** (default 100). Training target `build_target` lives on this scale.
- **`overutilisation_bonus`** — rule-based, gated (see `overutilisation.*`); optional **`min_base_score_for_any_bonus`** so bonus does not apply when the base is very low.
- **`value_score` / `display_value_score`** — `clip(base + bonus, 0, display_max)` (default display 150).
- **Recommendations** (`scoring/daily_batch.py`): `recommendation.score_column` is `combined` (thresholds on `value_score`) or `base` (uses `base_*_threshold` on 0–100).

### Transaction model (Phase B, separate path)

XGBoost on per-transaction features (not the subscription `ValueScoreModel`). Train/test split is **grouped by `user_id`**.

**Synthetic (handcrafted proxy target):**

```bash
python value_score_model/scripts/train_transaction_value.py --synthetic harness --n-users 200 --out checkpoints/transaction_value.pkl
```

**Real CSV folder:** if `transaction_valuations.csv` exists (from `build_training_data.py`) and enough rows join to `transactions.csv` (`transaction_training.min_supervised_rows` in YAML), training uses supervised **`value_score`** mapped from `[0, label_max]` to `[0, learned_score_max]`; otherwise it falls back to the proxy target.

```bash
PYTHONPATH=. python value_score_model/scripts/train_transaction_value.py \
  --data-dir /path/to/normalized_csvs --out checkpoints/txn_real.pkl --seed 42
```

Options: `--data-dir`, `--synthetic` (`sample` \| `harness`), `--n-users`, `--seed`, `--test-size`, `--out`, `--config`, `--force-proxy` (always proxy labels).

Train a checkpoint first (optional):

```bash
python value_score_model/train.py --synthetic --n-users 500 --seed 42 --model-dir checkpoints/
```

Then score with:

```bash
python value_score_model/scripts/simulate_and_export_value_scores.py --n-users 2000 --seed 42 --model-dir checkpoints/
```

Tests:

```bash
pytest value_score_model/tests/test_simulate_export.py -v
```
