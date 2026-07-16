# Quant & Statistics Best Practices

## Purpose
This document is a compact working guide for quantitative research in Cursor. It focuses on model estimation, backtesting, prediction, validation, and production hygiene.

## Core Principles
- Start with a clear target variable, prediction horizon, and decision rule.
- Separate research, validation, and production code.
- Treat every feature as suspect until it survives strict out-of-sample testing.
- Prefer simple, interpretable models unless complexity adds measurable value.
- Track assumptions explicitly and revisit them when regime changes appear.

## Data Handling
- Define the exact information set available at each timestamp.
- Align all features to the time they would have been known in reality.
- Remove or neutralize lookahead bias, survivorship bias, selection bias, and publication bias.
- Record timezone, trading calendar, corporate actions, and data vendor quirks.
- Use robust cleaning rules and keep raw data immutable.

## Feature Engineering
- Build features from economically plausible signals, not just statistical fit.
- Lag inputs appropriately and avoid using future aggregates.
- Normalize within the correct cross-section or rolling window.
- Check feature stability across time, assets, and regimes.
- Drop features that are redundant, unstable, or difficult to explain.

## Model Estimation
- Start with a benchmark model and beat it before increasing complexity.
- Use regularization when the feature set is large relative to observations.
- Estimate uncertainty, not only point forecasts.
- Inspect residuals for autocorrelation, heteroskedasticity, heavy tails, and structural breaks.
- Refit only when the gain from update frequency exceeds transaction and operational costs.

## Validation Design
- Use time-series splits, walk-forward validation, or purged cross-validation for dependent data.
- Keep the final test set untouched until the end.
- Tune hyperparameters only on training or validation folds.
- Evaluate robustness across multiple sample windows and subperiods.
- Prefer repeated evidence over a single strong backtest.

## Backtest Discipline
- Model all costs: fees, spreads, slippage, market impact, borrow, funding, and latency.
- Use realistic execution assumptions and state them clearly.
- Enforce position limits, cash constraints, and turnover constraints.
- Avoid rebalancing at impossible prices or using data that arrives after the trade decision.
- Report turnover, drawdown, exposure, hit rate, and capacity alongside returns.

## Prediction Practice
- Distinguish classification, regression, ranking, and probability forecasting.
- Calibrate probabilities when decisions depend on confidence.
- Evaluate using metrics that match the objective, such as log loss, Brier score, AUC, RMSE, MAE, or rank correlation.
- Check calibration curves, prediction intervals, and error concentration by regime.
- Use ensemble methods only when they improve stability or calibration.

## Overfitting Controls
- Limit the number of degrees of freedom.
- Penalize complexity through regularization, early stopping, or feature selection.
- Use nested validation when model selection is extensive.
- Adjust expectations for multiple testing and data snooping.
- Ask whether the edge survives after realistic costs and implementation frictions.

## Statistics Checks
- Test stationarity only when the assumption matters for the model.
- Examine autocorrelation, partial autocorrelation, volatility clustering, and cross-sectional dependence.
- Use robust standard errors when errors are not i.i.d.
- Compare results under alternative specifications, not just one preferred setup.
- Report confidence intervals and uncertainty bands whenever possible.

## Performance Metrics
- Use both statistical and economic metrics.
- For trading strategies, report CAGR, Sharpe, Sortino, max drawdown, Calmar, hit rate, profit factor, turnover, and capacity.
- For predictive models, report calibration, discrimination, and error distribution.
- For classification with imbalance, avoid relying on accuracy alone.
- Compare against a naive baseline and the current production model.

## Reproducibility
- Fix random seeds where appropriate.
- Version data, code, parameters, and evaluation windows.
- Log every experiment with inputs, outputs, and decision notes.
- Keep notebooks for exploration and scripts for repeatable runs.
- Make results reproducible from a clean environment.

## Production Readiness
- Add sanity checks for missing data, stale feeds, and extreme values.
- Monitor live performance against backtest expectations.
- Detect regime drift, feature drift, and label drift.
- Build rollback rules before deploying capital.
- Prefer alertable, observable systems over clever but opaque ones.

## Common Failure Modes
- Using future data by accident.
- Over-optimizing on a small sample.
- Ignoring transaction costs.
- Confusing in-sample fit with genuine predictive power.
- Changing the research question after seeing the results.

## Research Checklist
1. Define the hypothesis and trading or decision rule.
2. Verify the data timeline and feature availability.
3. Build a simple baseline.
4. Design a proper time-aware validation scheme.
5. Measure predictive and economic performance.
6. Stress test across regimes and assumptions.
7. Document limitations and failure cases.
8. Promote only what remains robust after costs.

## Suggested Workflow
- Explore the data.
- Form a hypothesis.
- Build a baseline.
- Validate out of sample.
- Stress test.
- Estimate capacity and costs.
- Deploy cautiously.
- Monitor and retrain only when justified.

## Short Template
### Model Card
- Objective:
- Horizon:
- Target:
- Universe:
- Features:
- Train window:
- Validation method:
- Test window:
- Costs assumed:
- Key metrics:
- Known failure modes:

## Notes for Cursor
- Prefer small, testable functions.
- Separate data prep, model training, backtesting, and reporting.
- Write assertions for time alignment and no-lookahead checks.
- Use typed inputs where possible.
- Keep experiment code readable enough to audit later.
