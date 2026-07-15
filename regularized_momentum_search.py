import importlib
from pprint import pformat

import numpy as np
import pandas as pd

import strategy_experiments
import teamName


LOOKBACKS = [5, 10, 20, 30, 40, 50, 60, 70]
TRAIN_DAYS = 400
RANDOM_SEED = 20260709
N_RANDOM_CANDIDATES = 3000
CONCENTRATION_PENALTIES = [0.00, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50]
MAX_WEIGHT_PENALTIES = [0.00, 0.005, 0.01, 0.02, 0.05, 0.10]
ASSET_SHARPE_FLOOR = 0.0


def annualised_sharpe(series):
    series = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return -np.inf
    volatility = series.std()
    if not np.isfinite(volatility) or volatility <= 0:
        return -np.inf
    return np.sqrt(250) * series.mean() / volatility


def make_z_scores(prices_df):
    returns = prices_df.pct_change()
    past_returns = returns.shift(1)
    z_scores = {}

    for lookback in LOOKBACKS:
        momentum = past_returns.rolling(lookback).sum()
        volatility = past_returns.rolling(lookback).std()
        vol_normalised = momentum / volatility
        cross_mean = vol_normalised.mean(axis=1)
        cross_std = vol_normalised.std(axis=1)
        z_scores[lookback] = vol_normalised.sub(cross_mean, axis=0).div(cross_std, axis=0)

    return returns, z_scores


def candidate_weights(rng):
    candidates = []
    uniform = np.full(len(LOOKBACKS), 1.0 / len(LOOKBACKS))
    candidates.append(uniform)

    for lookback_index in range(len(LOOKBACKS)):
        one_hot = np.zeros(len(LOOKBACKS))
        one_hot[lookback_index] = 1.0
        candidates.append(one_hot)

    for alpha in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        draws = rng.dirichlet(np.full(len(LOOKBACKS), alpha), size=N_RANDOM_CANDIDATES // 6)
        candidates.extend(draws)

    return np.array(candidates)


def optimise_for_penalty(prices_df, concentration_penalty, max_weight_penalty):
    rng = np.random.default_rng(RANDOM_SEED)
    returns, z_scores = make_z_scores(prices_df)
    train_slice = slice(0, TRAIN_DAYS)
    test_slice = slice(TRAIN_DAYS, None)
    candidates = candidate_weights(rng)
    concentrations = np.sum(candidates**2, axis=1)
    max_weights = np.max(candidates, axis=1)

    rows = []
    selected_weights = {}

    asset_universe = list(teamName.MOMENTUM_HORIZON_WEIGHTS)

    for asset in asset_universe:
        asset_returns = returns[asset].to_numpy()
        asset_z = np.column_stack([z_scores[lookback][asset].to_numpy() for lookback in LOOKBACKS])
        existing_weights = np.array(
            [teamName.MOMENTUM_HORIZON_WEIGHTS[asset].get(lookback, 0.0) for lookback in LOOKBACKS]
        )
        asset_candidates = np.vstack([candidates, existing_weights])
        asset_concentrations = np.sum(asset_candidates**2, axis=1)
        asset_max_weights = np.max(asset_candidates, axis=1)

        strategy_returns = (asset_z @ asset_candidates.T) * asset_returns[:, None]
        train_returns = strategy_returns[train_slice, :]
        test_returns = strategy_returns[test_slice, :]

        train_means = np.nanmean(train_returns, axis=0)
        train_stds = np.nanstd(train_returns, axis=0, ddof=1)
        train_sharpes = np.divide(
            np.sqrt(250) * train_means,
            train_stds,
            out=np.full_like(train_means, -np.inf),
            where=train_stds > 0,
        )

        test_means = np.nanmean(test_returns, axis=0)
        test_stds = np.nanstd(test_returns, axis=0, ddof=1)
        test_sharpes = np.divide(
            np.sqrt(250) * test_means,
            test_stds,
            out=np.full_like(test_means, -np.inf),
            where=test_stds > 0,
        )

        objectives = train_sharpes - concentration_penalty * asset_concentrations - max_weight_penalty * asset_max_weights
        best_index = int(np.nanargmax(objectives))
        weights = asset_candidates[best_index]
        best = {
            "asset": asset,
            "weights": weights,
            "objective": objectives[best_index],
            "train_sharpe": train_sharpes[best_index],
            "test_sharpe": test_sharpes[best_index],
            "concentration": asset_concentrations[best_index],
            "max_weight": asset_max_weights[best_index],
        }
        if best["train_sharpe"] > ASSET_SHARPE_FLOOR:
            selected_weights[asset] = {
                lookback: round(float(weight), 4)
                for lookback, weight in zip(LOOKBACKS, best["weights"])
            }
            rows.append(
                {
                    key: value
                    for key, value in best.items()
                    if key != "weights"
                }
                | selected_weights[asset]
            )

    return selected_weights, pd.DataFrame(rows).sort_values("train_sharpe", ascending=False)


def evaluate_weight_set(selected_weights):
    return strategy_experiments.evaluate_module(
        cap=10_000_000,
        momentum_weight=0.60,
        pairs_weight=0.20,
        tuple_weight=0.20,
        momentum_horizon_weights=selected_weights,
    )


def main():
    prices_df = pd.read_csv("prices.txt", sep=r"\s+")
    rows = []
    best = None

    for concentration_penalty in CONCENTRATION_PENALTIES:
        for max_weight_penalty in MAX_WEIGHT_PENALTIES:
            selected_weights, asset_table = optimise_for_penalty(
                prices_df,
                concentration_penalty=concentration_penalty,
                max_weight_penalty=max_weight_penalty,
            )
            result = evaluate_weight_set(selected_weights)
            row = {
                "concentration_penalty": concentration_penalty,
                "max_weight_penalty": max_weight_penalty,
                "n_assets": len(selected_weights),
                "avg_concentration": asset_table["concentration"].mean(),
                "avg_max_weight": asset_table["max_weight"].mean(),
                **result,
            }
            rows.append(row)
            if best is None or row["score"] > best["result"]["score"]:
                best = {
                    "result": row,
                    "weights": selected_weights,
                    "asset_table": asset_table,
                }

    summary = pd.DataFrame(rows).sort_values("score", ascending=False)
    print("\nTop regularised momentum settings")
    print(summary.head(10).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nBest eval-style result")
    print(best["result"])

    print("\nBest asset weights")
    print(pformat(best["weights"], sort_dicts=False, width=140))

    print("\nBest asset table")
    display_columns = ["asset", "train_sharpe", "test_sharpe", "concentration", "max_weight", *LOOKBACKS]
    print(best["asset_table"][display_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
