# Freight Rate Prediction

Predicts the posted rate of truckload freight loads for November–December 2025 from January–October 2025 history.

**Result:** MAE $42 per load (1.8% MAPE) on a Sep–Oct time holdout, against $88 for a lane-median baseline. With 8 cities never seen in training, MAE is $46 (20 of 20 random draws beat the baseline).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Data files go in `data/` (`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`).

## Run

```bash
python -m src.train          # validation report, final model, validation_predictions.csv, December predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
python -m src.diagnostics    # quote_signal regimes and market_index by month
python -m src.stats_tests    # hypothesis tests behind every decision -> results/stats_tests.json
python -m pytest tests       # unit tests
```

`src.train` takes about 30 seconds on a laptop. Everything uses a fixed seed.

## Approach

- **Target:** log(rate per mile). It removes the distance scale and turns errors into percentages. Prediction = exp(output) × distance.
- **Model:** scikit-learn `HistGradientBoostingRegressor`, squared loss, 31 leaves, monotone constraints (heavier → pricier, longer → cheaper per mile).
- **Features:** distance, pickup/delivery coordinates and their deltas, straight-line distance and detour ratio, equipment, weight, day of week. City names are left out on purpose: 8 validation cities never appear in training.
- **Split:** by time, never at random. Train Jan–Aug, validate Sep–Oct, the same direction as the real task. Extra holdouts: unseen cities (20 random draws), and August alone (see below).

## Key findings

1. **`quote_signal` is a trap.** It switches meaning by month: equal to the true rate per mile (Jan, Feb, Mar, Jun, Sep), inversely related to it (Apr, May, Jul, Oct), or pure noise (Aug). A label-free per-day correlation test shows Nov–Dec are noise months, and its distribution in validation matches August (KS p = 0.92). Using it looks great on Sep–Oct and triples the error on August ($114 vs $40). It is excluded.
2. **`market_index` hurts.** It tracks prices within a month but not across months (January and September share the same index at different price levels). Adding it raises Sep–Oct MAE from $42 to $117. It is excluded.
3. **Data quality.** 1.4% of rates are entry errors (2.4–4.7× or 0.2–0.4× the lane norm; in direct months the quote for those loads is normal), dropped from training. Negative weights are sign errors (the sign has no price effect, p = 0.84) and are made positive. Coordinates are perturbed and clipped, so they are used as approximate positions.
4. **Seasonality is the main residual risk.** Prices swing ~7.7% across months, but nothing available predicts the Nov–Dec level; the model predicts the yearly average level.

## Layout

| Path | Contents |
| --- | --- |
| `src/pipeline.py` | loading, cleaning, features |
| `src/train.py` | holdouts, variant comparison, final fit, predictions |
| `src/diagnostics.py` | quote_signal regime and market_index tables |
| `src/stats_tests.py` | Diebold-Mariano, Wilcoxon, sign, KS, permutation, conformal, Holm |
| `tests/` | unit tests |
| `results/` | validation and test outputs (JSON) |
| `validation_predictions.csv` | final predictions, `load_id,predicted_rate` |
| `scorer_results/candidate_december.png` | December chart from `score.py` |
