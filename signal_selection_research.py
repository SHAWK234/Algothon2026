import itertools

import numpy as np
import pandas as pd

from stable_signal_research import (
    ASSETS,
    N_INST,
    candidate_targets,
    evaluate_dollars,
    lead_lag_targets,
    load_prices,
    portfolio_from_candidates,
)


TOP_N_VALUES = [5, 10, 15, 20, 30, 40]
LEAD_LAG_LAGS = [1, 2, 3, 5, 10]
MAX_LEAD_LAG_CANDIDATES = 250

PRIMARY_SPLIT = ("train_1_250_test_250_500", 1, 250, 250, 500)
WALK_FORWARD_SPLITS = [
    ("wf_train_1_200_test_200_300", 1, 200, 200, 300),
    ("wf_train_50_250_test_250_350", 50, 250, 250, 350),
    ("wf_train_100_300_test_300_400", 100, 300, 300, 400),
    ("wf_train_150_350_test_350_500", 150, 350, 350, 500),
]


def build_fixed_candidates():
    from stable_signal_research import build_specs

    candidates = []
    for spec in build_specs():
        candidates.append({**spec, "kind": "fixed"})
    return candidates


def build_lead_lag_candidates(prices, train_start, train_end):
    train_prices = prices[:, train_start - 1 : train_end]
    returns = train_prices[:, 1:] / train_prices[:, :-1] - 1.0
    raw = []

    for lag in LEAD_LAG_LAGS:
        for leader_idx, follower_idx in itertools.permutations(range(N_INST), 2):
            x = returns[leader_idx, :-lag]
            y = returns[follower_idx, lag:]
            if len(x) < 60 or np.std(x) <= 0 or np.std(y) <= 0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            raw.append((abs(corr), corr, leader_idx, follower_idx, lag))

    raw.sort(reverse=True)
    candidates = []
    for _, corr, leader_idx, follower_idx, lag in raw[:MAX_LEAD_LAG_CANDIDATES]:
        corr_sign = float(np.sign(corr))
        candidates.append({
            "name": f"{ASSETS[leader_idx]}_lead{lag}_{ASSETS[follower_idx]}_corr{corr_sign:+.0f}",
            "asset_idx": follower_idx,
            "asset": ASSETS[follower_idx],
            "family": "lead_lag",
            "mode": "momentum",
            "kind": "lead_lag",
            "leader_idx": leader_idx,
            "leader": ASSETS[leader_idx],
            "lag": lag,
            "corr_sign": corr_sign,
            "train_corr": corr,
        })
    return candidates


def normalized_candidate(candidate):
    candidate = dict(candidate)
    for key in ["asset_idx", "leader_idx", "lag", "lookback", "vol_window", "z_window"]:
        if key in candidate and pd.notna(candidate[key]):
            candidate[key] = int(candidate[key])
    if "corr_sign" in candidate and pd.notna(candidate["corr_sign"]):
        candidate["corr_sign"] = float(candidate["corr_sign"])
    if "threshold" in candidate and pd.notna(candidate["threshold"]):
        candidate["threshold"] = float(candidate["threshold"])
    if "lag_signal" in candidate and pd.notna(candidate["lag_signal"]):
        candidate["lag_signal"] = bool(candidate["lag_signal"])
    return candidate


def targets_for_candidate(prices, candidate):
    candidate = normalized_candidate(candidate)
    if candidate["kind"] == "lead_lag":
        return lead_lag_targets(
            prices,
            candidate["leader_idx"],
            candidate["asset_idx"],
            candidate["lag"],
            candidate["corr_sign"],
        )
    return candidate_targets(prices, candidate)


def train_subwindows(train_start, train_end):
    train_len = train_end - train_start
    window = 50 if train_len < 220 else 75
    step = 25
    windows = []
    for start in range(train_start, train_end - window + 1, step):
        windows.append((start, start + window))
    return windows


def metrics_prefix(result, prefix):
    return {
        f"{prefix}_score": result["score"],
        f"{prefix}_mean_pl": result["mean_pl"],
        f"{prefix}_std_pl": result["std_pl"],
        f"{prefix}_sharpe": result["sharpe"],
        f"{prefix}_worst_day": result["worst_day"],
    }


def candidate_rows_for_split(prices, split):
    split_name, train_start, train_end, test_start, test_end = split
    candidates = build_fixed_candidates()
    candidates.extend(build_lead_lag_candidates(prices, train_start, train_end))
    rows = []

    subwindows = train_subwindows(train_start, train_end)
    for candidate in candidates:
        targets = targets_for_candidate(prices, candidate)
        train_result = evaluate_dollars(prices, targets, train_start, train_end)
        test_result = evaluate_dollars(prices, targets, test_start, test_end)
        sub_scores = [
            evaluate_dollars(prices, targets, start, end)["score"]
            for start, end in subwindows
        ]

        rows.append({
            **candidate,
            "split": split_name,
            **metrics_prefix(train_result, "train"),
            **metrics_prefix(test_result, "test"),
            "train_roll_min_score": float(np.min(sub_scores)),
            "train_roll_mean_score": float(np.mean(sub_scores)),
            "train_roll_neg_count": int(np.sum(np.asarray(sub_scores) < 0)),
        })

    return pd.DataFrame(rows)


def sorted_candidates(candidates, selector_name):
    selected = candidates.copy()

    if selector_name == "train_score":
        return selected.sort_values(
            ["train_score", "train_roll_min_score"],
            ascending=[False, False],
        )

    if selector_name == "strict_train":
        selected = selected[
            (selected["train_score"] > 0)
            & (selected["train_sharpe"] > 1.0)
            & (selected["train_roll_min_score"] > 0)
            & (selected["train_roll_neg_count"] == 0)
        ]
        return selected.sort_values(
            ["train_roll_min_score", "train_score"],
            ascending=[False, False],
        )

    if selector_name == "strict_one_per_asset":
        selected = sorted_candidates(selected, "strict_train")
        return selected.drop_duplicates("asset_idx")

    raise ValueError(f"Unknown selector {selector_name!r}")


def train_daily_pl_for_names(prices, candidates, names, train_start, train_end):
    by_name = candidates.set_index("name").to_dict("index")
    daily_by_name = {}
    target_by_name = {}
    for name in names:
        candidate = by_name[name]
        targets = targets_for_candidate(prices, candidate)
        result = evaluate_dollars(prices, targets, train_start, train_end, return_daily=True)
        daily_by_name[name] = result["daily_pl"]
        target_by_name[name] = targets
    return daily_by_name, target_by_name


def greedy_low_corr_names(prices, candidates, train_start, train_end, max_n, corr_limit):
    pool = sorted_candidates(candidates, "strict_train")
    pool = pool.drop_duplicates("name")
    probe_names = pool.head(250)["name"].tolist()
    daily_by_name, _ = train_daily_pl_for_names(
        prices,
        candidates,
        probe_names,
        train_start,
        train_end,
    )

    selected_names = []
    selected_assets = set()
    for _, row in pool.iterrows():
        if len(selected_names) >= max_n:
            break
        if row["asset_idx"] in selected_assets:
            continue
        name = row["name"]
        if name not in daily_by_name:
            continue

        daily = daily_by_name[name]
        too_correlated = False
        for selected_name in selected_names:
            other = daily_by_name[selected_name]
            if np.std(daily) <= 0 or np.std(other) <= 0:
                continue
            corr = float(np.corrcoef(daily, other)[0, 1])
            if abs(corr) > corr_limit:
                too_correlated = True
                break
        if too_correlated:
            continue

        selected_names.append(name)
        selected_assets.add(row["asset_idx"])

    return selected_names


def evaluate_portfolio_names(prices, candidates, names, train_start, train_end, test_start, test_end):
    by_name = candidates.set_index("name").to_dict("index")
    targets_by_name = {}
    for name in names:
        targets_by_name[name] = targets_for_candidate(prices, by_name[name])
    targets = portfolio_from_candidates(targets_by_name, names)
    train_result = evaluate_dollars(prices, targets, train_start, train_end)
    test_result = evaluate_dollars(prices, targets, test_start, test_end)
    return train_result, test_result


def build_selection_rows(prices, split, candidates):
    split_name, train_start, train_end, test_start, test_end = split
    rows = []
    selectors = [
        "train_score",
        "strict_train",
        "strict_one_per_asset",
    ]

    for selector in selectors:
        ranked = sorted_candidates(candidates, selector)
        for top_n in TOP_N_VALUES:
            names = ranked.head(top_n)["name"].tolist()
            if not names:
                continue
            train_result, test_result = evaluate_portfolio_names(
                prices,
                candidates,
                names,
                train_start,
                train_end,
                test_start,
                test_end,
            )
            rows.append({
                "split": split_name,
                "selector": selector,
                "top_n": len(names),
                **metrics_prefix(train_result, "portfolio_train"),
                **metrics_prefix(test_result, "portfolio_test"),
                "names": ";".join(names),
            })

    for corr_limit in [0.25, 0.35, 0.50]:
        for top_n in TOP_N_VALUES:
            names = greedy_low_corr_names(
                prices,
                candidates,
                train_start,
                train_end,
                top_n,
                corr_limit,
            )
            if not names:
                continue
            train_result, test_result = evaluate_portfolio_names(
                prices,
                candidates,
                names,
                train_start,
                train_end,
                test_start,
                test_end,
            )
            rows.append({
                "split": split_name,
                "selector": f"greedy_corr_{corr_limit:.2f}",
                "top_n": len(names),
                **metrics_prefix(train_result, "portfolio_train"),
                **metrics_prefix(test_result, "portfolio_test"),
                "names": ";".join(names),
            })

    return pd.DataFrame(rows)


def summarize_walk_forward(selection_results):
    grouped = (
        selection_results.groupby(["selector", "top_n"])
        .agg(
            avg_test_score=("portfolio_test_score", "mean"),
            median_test_score=("portfolio_test_score", "median"),
            min_test_score=("portfolio_test_score", "min"),
            neg_test_windows=("portfolio_test_score", lambda values: int((values < 0).sum())),
            avg_test_mean_pl=("portfolio_test_mean_pl", "mean"),
            avg_test_std_pl=("portfolio_test_std_pl", "mean"),
            avg_test_sharpe=("portfolio_test_sharpe", "mean"),
        )
        .reset_index()
        .sort_values(
            ["neg_test_windows", "avg_test_score", "min_test_score"],
            ascending=[True, False, False],
        )
    )
    return grouped


def main():
    prices = load_prices()
    split_results = []
    candidate_result_frames = []

    for split in [PRIMARY_SPLIT, *WALK_FORWARD_SPLITS]:
        candidates = candidate_rows_for_split(prices, split)
        candidate_result_frames.append(candidates)
        split_results.append(build_selection_rows(prices, split, candidates))

    candidate_results = pd.concat(candidate_result_frames, ignore_index=True)
    selection_results = pd.concat(split_results, ignore_index=True)

    primary_name = PRIMARY_SPLIT[0]
    primary_results = selection_results[selection_results["split"] == primary_name].copy()
    walk_forward_results = selection_results[selection_results["split"] != primary_name].copy()
    walk_forward_summary = summarize_walk_forward(walk_forward_results)

    candidate_results.to_csv("signal_selection_candidate_results.csv", index=False)
    selection_results.to_csv("signal_selection_portfolio_results.csv", index=False)
    primary_results.to_csv("signal_selection_primary_results.csv", index=False)
    walk_forward_summary.to_csv("signal_selection_walk_forward_summary.csv", index=False)

    print("\nPrimary no-peek selection: train 1-250, test 250-500")
    print(
        primary_results[
            [
                "selector",
                "top_n",
                "portfolio_train_score",
                "portfolio_train_std_pl",
                "portfolio_test_score",
                "portfolio_test_mean_pl",
                "portfolio_test_std_pl",
                "portfolio_test_sharpe",
            ]
        ]
        .sort_values(["portfolio_test_score"], ascending=False)
        .round(3)
        .head(20)
        .to_string(index=False)
    )

    print("\nWalk-forward selector summary")
    print(
        walk_forward_summary[
            [
                "selector",
                "top_n",
                "avg_test_score",
                "median_test_score",
                "min_test_score",
                "neg_test_windows",
                "avg_test_mean_pl",
                "avg_test_std_pl",
                "avg_test_sharpe",
            ]
        ]
        .round(3)
        .head(30)
        .to_string(index=False)
    )

    print("\nBest primary basket names")
    best = primary_results.sort_values("portfolio_test_score", ascending=False).iloc[0]
    print(f"{best['selector']} top {int(best['top_n'])}")
    for name in str(best["names"]).split(";"):
        print(f"- {name}")


if __name__ == "__main__":
    main()
