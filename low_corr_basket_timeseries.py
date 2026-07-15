import re
import os

import numpy as np
import pandas as pd

from stable_signal_research import (
    ASSETS,
    POSITION_LIMITS,
    candidate_targets,
    evaluate_dollars,
    lead_lag_targets,
    load_prices,
    portfolio_from_candidates,
    score,
)


SIGNAL_ROW = "low_corr_top30"
TOP_N_VALUES = [5, 10, 15, 20, 25, 30]


def max_drawdown(cumulative_pl):
    peaks = np.maximum.accumulate(cumulative_pl)
    drawdowns = cumulative_pl - peaks
    return float(np.min(drawdowns))


def metrics_from_daily_pl(daily_pl):
    daily_pl = np.asarray(daily_pl, dtype=float)
    mean_pl = float(np.mean(daily_pl))
    std_pl = float(np.std(daily_pl))
    sharpe = 0.0 if std_pl <= 0 else float(np.sqrt(250.0) * mean_pl / std_pl)
    cumulative = np.cumsum(daily_pl)
    return {
        "days": int(len(daily_pl)),
        "total_pl": float(np.sum(daily_pl)),
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "ann_sharpe": sharpe,
        "score": float(score(mean_pl, std_pl)),
        "worst_day": float(np.min(daily_pl)),
        "best_day": float(np.max(daily_pl)),
        "positive_day_rate": float(np.mean(daily_pl > 0)),
        "max_drawdown": max_drawdown(cumulative),
    }


def load_signal_names():
    portfolios = pd.read_csv("stable_signal_portfolios.csv")
    row = portfolios.loc[portfolios["portfolio"] == SIGNAL_ROW]
    if row.empty:
        raise ValueError(f"Could not find portfolio row {SIGNAL_ROW!r}")
    return str(row.iloc[0]["signals"]).split(";")


def build_targets_for_names(prices, signal_names):
    specs_by_name = {spec["name"]: spec for spec in build_specs_once()}
    targets_by_name = {}
    asset_to_idx = {asset: idx for idx, asset in enumerate(ASSETS)}
    lead_lag_pattern = re.compile(r"^(.+)_lead(\d+)_(.+)_corr([+-]1)$")

    for name in signal_names:
        if name in specs_by_name:
            targets_by_name[name] = candidate_targets(prices, specs_by_name[name])
            continue

        match = lead_lag_pattern.match(name)
        if match is None:
            raise ValueError(f"Could not parse signal name {name!r}")

        leader, lag, follower, corr_sign = match.groups()
        targets_by_name[name] = lead_lag_targets(
            prices,
            asset_to_idx[leader],
            asset_to_idx[follower],
            int(lag),
            float(corr_sign),
        )

    return targets_by_name


def build_specs_once():
    from stable_signal_research import build_specs

    return build_specs()


def evaluate_window(prices, targets, label, start_day, end_day):
    result = evaluate_dollars(prices, targets, start_day, end_day, return_daily=True)
    daily_pl = result.pop("daily_pl")
    return {
        "window": label,
        "start_day": start_day,
        "end_day": end_day,
        **metrics_from_daily_pl(daily_pl),
    }


def make_rolling_rows(prices, targets, window_size, step):
    rows = []
    n_days = prices.shape[1]
    for start in range(1, n_days - window_size + 1, step):
        end = start + window_size
        rows.append(
            evaluate_window(
                prices,
                targets,
                f"{start}-{end}",
                start,
                end,
            )
        )
    return rows


def plot_rolling_scores(blocks, rolling_100, cumulative_df):
    try:
        os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    for top_n, frame in rolling_100.groupby("top_n"):
        axes[0].plot(
            frame["end_day"],
            frame["score"],
            marker="o",
            linewidth=1.5,
            label=f"top {top_n}",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Low-correlation basket rolling 100-day score")
    axes[0].set_xlabel("End day")
    axes[0].set_ylabel("Score")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(cumulative_df["day"], cumulative_df["cumulative_pl"], color="#1f77b4")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Low-correlation top 30 cumulative PnL")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Cumulative PnL")

    fig.savefig("low_corr_basket_score_over_time.png", dpi=160)
    plt.close(fig)


def main():
    prices = load_prices()
    n_days = prices.shape[1]
    signal_names = load_signal_names()
    targets_by_name = build_targets_for_names(prices, signal_names)

    block_rows = []
    rolling_rows = []
    rolling_50_rows = []
    topn_summary_rows = []

    portfolios = {}
    quarter = n_days // 4
    block_defs = [
        (f"1-{quarter}", 1, quarter),
        (f"{quarter}-{2 * quarter}", quarter, 2 * quarter),
        (f"{2 * quarter}-{3 * quarter}", 2 * quarter, 3 * quarter),
        (f"{3 * quarter}-{n_days}", 3 * quarter, n_days),
        (f"1-{n_days // 2}", 1, n_days // 2),
        (f"{n_days // 2}-{n_days}", n_days // 2, n_days),
    ]

    for top_n in TOP_N_VALUES:
        names = signal_names[:top_n]
        targets = portfolio_from_candidates(targets_by_name, names)
        portfolios[top_n] = targets

        for label, start, end in block_defs:
            row = evaluate_window(prices, targets, label, start, end)
            block_rows.append({"top_n": top_n, **row})

        for row in make_rolling_rows(prices, targets, window_size=100, step=25):
            rolling_rows.append({"top_n": top_n, **row})

        for row in make_rolling_rows(prices, targets, window_size=50, step=25):
            rolling_50_rows.append({"top_n": top_n, **row})

    blocks = pd.DataFrame(block_rows)
    rolling_100 = pd.DataFrame(rolling_rows)
    rolling_50 = pd.DataFrame(rolling_50_rows)

    for top_n in TOP_N_VALUES:
        roll = rolling_100[rolling_100["top_n"] == top_n]
        topn_summary_rows.append({
            "top_n": top_n,
            "rolling100_mean_score": float(roll["score"].mean()),
            "rolling100_median_score": float(roll["score"].median()),
            "rolling100_min_score": float(roll["score"].min()),
            "rolling100_neg_windows": int((roll["score"] < 0).sum()),
            "rolling100_mean_std_pl": float(roll["std_pl"].mean()),
            "rolling100_max_std_pl": float(roll["std_pl"].max()),
        })

    summary = pd.DataFrame(topn_summary_rows)

    top30_result = evaluate_dollars(
        prices,
        portfolios[30],
        1,
        prices.shape[1],
        return_daily=True,
    )
    cumulative_df = pd.DataFrame({
        "day": np.arange(2, prices.shape[1] + 1),
        "daily_pl": top30_result["daily_pl"],
    })
    cumulative_df["cumulative_pl"] = cumulative_df["daily_pl"].cumsum()

    blocks.to_csv("low_corr_basket_block_scores.csv", index=False)
    rolling_100.to_csv("low_corr_basket_rolling100_scores.csv", index=False)
    rolling_50.to_csv("low_corr_basket_rolling50_scores.csv", index=False)
    summary.to_csv("low_corr_basket_topn_summary.csv", index=False)
    cumulative_df.to_csv("low_corr_basket_top30_daily_pl.csv", index=False)
    plot_rolling_scores(blocks, rolling_100, cumulative_df)

    print("\nBlock scores")
    print(
        blocks[
            [
                "top_n",
                "window",
                "score",
                "mean_pl",
                "std_pl",
                "ann_sharpe",
                "worst_day",
                "max_drawdown",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nRolling 100-day summary")
    print(summary.round(3).to_string(index=False))

    print("\nTop 30 worst 100-day windows")
    print(
        rolling_100[rolling_100["top_n"] == 30]
        .sort_values("score")
        .head(8)[
            [
                "window",
                "score",
                "mean_pl",
                "std_pl",
                "ann_sharpe",
                "worst_day",
                "max_drawdown",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print("\nTop 30 best 100-day windows")
    print(
        rolling_100[rolling_100["top_n"] == 30]
        .sort_values("score", ascending=False)
        .head(8)[
            [
                "window",
                "score",
                "mean_pl",
                "std_pl",
                "ann_sharpe",
                "worst_day",
                "max_drawdown",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
