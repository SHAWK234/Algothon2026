import importlib
import itertools

import numpy as np
import pandas as pd

import teamName


prices_df = pd.read_csv("prices.txt", sep=r"\s+")
prices = prices_df.values.T
returns_df = prices_df.pct_change()
assets = list(prices_df.columns)
asset_to_idx = {asset: idx for idx, asset in enumerate(assets)}

n_inst, n_days = prices.shape
TRAIN_DAYS = 400
start_day = TRAIN_DAYS

comm_rate = np.full(n_inst, 0.0001)
comm_rate[0] = 0.00002
dlr_pos_limit = np.full(n_inst, 10_000)
dlr_pos_limit[0] = 100_000


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250) * mean_pl / std_pl
    frac = sharpe**2 / (sharpe**2 + 1)
    return mean_pl * frac


def evaluate_module(
    cap=None,
    momentum_weight=None,
    pairs_weight=None,
    tuple_weight=None,
    momentum_horizon_weights=None,
    pairs=None,
    tuples=None,
):
    importlib.reload(teamName)
    if cap is not None:
        teamName.GROSS_DOLLAR_EXPOSURE = cap
    if momentum_weight is not None:
        teamName.MOMENTUM_WEIGHT = momentum_weight
    if pairs_weight is not None:
        teamName.PAIRS_WEIGHT = pairs_weight
    if tuple_weight is not None:
        teamName.TUPLE_WEIGHT = tuple_weight
    if momentum_horizon_weights is not None:
        teamName.MOMENTUM_HORIZON_WEIGHTS = momentum_horizon_weights
    if pairs is not None:
        teamName.PAIRS = pairs
    if tuples is not None:
        teamName.TUPLES = tuples

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    total_dvolume = 0.0
    value = 0.0
    comm = 0.0
    daily_pl = []

    for t in range(start_day, n_days + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]

        if t < n_days:
            new_pos_orig = teamName.getMyPosition(hist)
            pos_limits = (dlr_pos_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + comm

        dvolumes = cur_prices * np.abs(delta_pos)
        total_dvolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * comm_rate)

        cur_pos = np.array(new_pos)
        today_pl = cash + cur_pos.dot(cur_prices) - value
        value = cash + cur_pos.dot(cur_prices)

        if t > start_day:
            daily_pl.append(today_pl)

    daily_pl = np.array(daily_pl)
    mean_pl = daily_pl.mean()
    std_pl = daily_pl.std()
    sharpe = 0.0 if std_pl == 0 else np.sqrt(250) * mean_pl / std_pl
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": sharpe,
        "value": value,
        "dvolume": total_dvolume,
        "score": score(mean_pl, std_pl),
    }


def print_result(name, result, extra=None):
    extra = "" if extra is None else f" {extra}"
    print(
        f"{name:<32}{extra:<35}"
        f" mean={result['mean_pl']:8.2f}"
        f" sd={result['std_pl']:8.2f}"
        f" sharpe={result['sharpe']:6.3f}"
        f" score={result['score']:8.2f}"
        f" value={result['value']:9.2f}"
    )


def run_split_grid():
    caps = [2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]
    splits = [
        (0.55, 0.15, 0.30),
        (0.50, 0.20, 0.30),
        (0.50, 0.10, 0.40),
        (0.45, 0.15, 0.40),
        (0.60, 0.10, 0.30),
        (0.60, 0.20, 0.20),
        (0.45, 0.25, 0.30),
        (0.40, 0.30, 0.30),
        (0.35, 0.35, 0.30),
        (0.30, 0.40, 0.30),
    ]
    rows = []
    for mw, pw, tw in splits:
        for cap in caps:
            result = evaluate_module(cap=cap, momentum_weight=mw, pairs_weight=pw, tuple_weight=tw)
            rows.append({"kind": "split", "mw": mw, "pw": pw, "tw": tw, "cap": cap, **result})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def run_ablation():
    caps = [5_000_000, 10_000_000, 20_000_000]
    base_pairs = list(teamName.PAIRS)
    base_tuples = list(teamName.TUPLES)
    rows = []

    for cap in caps:
        rows.append({"kind": "none", "removed": "none", "cap": cap, **evaluate_module(cap=cap)})

    for i, pair in enumerate(base_pairs):
        pairs = [p for j, p in enumerate(base_pairs) if j != i]
        for cap in caps:
            rows.append({
                "kind": "remove_pair",
                "removed": str(pair[:2]),
                "cap": cap,
                **evaluate_module(cap=cap, pairs=pairs),
            })

    for i, tup in enumerate(base_tuples):
        tuples = [t for j, t in enumerate(base_tuples) if j != i]
        for cap in caps:
            rows.append({
                "kind": "remove_tuple",
                "removed": str(tup[:3]),
                "cap": cap,
                **evaluate_module(cap=cap, tuples=tuples),
            })

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def lead_lag_relationships(max_lag=30, top_n=8):
    train_returns = returns_df.iloc[:TRAIN_DAYS]
    rows = []
    for lag in range(1, max_lag + 1):
        lagged = train_returns.shift(lag)
        for leader in assets:
            x = lagged[leader]
            for follower in assets:
                if leader == follower:
                    continue
                y = train_returns[follower]
                valid = x.notna() & y.notna()
                if valid.sum() < 80:
                    continue
                corr = x[valid].corr(y[valid])
                rows.append({
                    "leader": leader,
                    "follower": follower,
                    "lag": lag,
                    "corr": corr,
                    "abs_corr": abs(corr),
                })
    rels = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    selected = []
    used = set()
    for _, row in rels.iterrows():
        key = (row["leader"], row["follower"])
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= top_n:
            break
    return pd.DataFrame(selected)


def lead_lag_weights(prc_so_far, rels):
    _, nt = prc_so_far.shape
    weights = np.zeros(n_inst)
    latest_returns = prc_so_far[:, 1:] / prc_so_far[:, :-1] - 1.0
    signals = np.zeros(n_inst)

    for _, row in rels.iterrows():
        lag = int(row["lag"])
        if latest_returns.shape[1] <= lag:
            continue
        leader_idx = asset_to_idx[row["leader"]]
        follower_idx = asset_to_idx[row["follower"]]
        corr = row["corr"]
        signal = np.sign(corr) * latest_returns[leader_idx, -lag]
        signals[follower_idx] += abs(corr) * signal

    gross = np.sum(np.abs(signals))
    if gross <= 0 or not np.isfinite(gross):
        return weights
    return signals / gross


def evaluate_with_lead_lag(lead_weight, rels, cap):
    importlib.reload(teamName)
    teamName.GROSS_DOLLAR_EXPOSURE = cap

    cash = 0.0
    cur_pos = np.zeros(n_inst)
    total_dvolume = 0.0
    value = 0.0
    comm = 0.0
    daily_pl = []

    for t in range(start_day, n_days + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]

        if t < n_days:
            base_weights = (
                teamName.MOMENTUM_WEIGHT * teamName._momentum_weights(hist)
                + teamName.PAIRS_WEIGHT * teamName._pairs_weights(hist)
                + teamName.TUPLE_WEIGHT * teamName._tuple_weights(hist)
            )
            ll_weights = lead_lag_weights(hist, rels)
            combined = (1 - lead_weight) * base_weights + lead_weight * ll_weights
            dollar_positions = teamName.GROSS_DOLLAR_EXPOSURE * combined
            new_pos_orig = np.divide(
                dollar_positions,
                cur_prices,
                out=np.zeros_like(dollar_positions),
                where=cur_prices > 0,
            ).astype(int)
            pos_limits = (dlr_pos_limit / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + comm
        dvolumes = cur_prices * np.abs(delta_pos)
        total_dvolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * comm_rate)
        cur_pos = np.array(new_pos)
        today_pl = cash + cur_pos.dot(cur_prices) - value
        value = cash + cur_pos.dot(cur_prices)
        if t > start_day:
            daily_pl.append(today_pl)

    daily_pl = np.array(daily_pl)
    mean_pl = daily_pl.mean()
    std_pl = daily_pl.std()
    sharpe = 0.0 if std_pl == 0 else np.sqrt(250) * mean_pl / std_pl
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": sharpe,
        "value": value,
        "dvolume": total_dvolume,
        "score": score(mean_pl, std_pl),
    }


def run_lead_lag():
    rels = lead_lag_relationships()
    rows = []
    for lead_weight in [0.02, 0.05, 0.10, 0.15]:
        for cap in [5_000_000, 10_000_000, 20_000_000]:
            rows.append({
                "kind": "lead_lag",
                "lead_weight": lead_weight,
                "cap": cap,
                **evaluate_with_lead_lag(lead_weight, rels, cap),
            })
    return rels, pd.DataFrame(rows).sort_values("score", ascending=False)


if __name__ == "__main__":
    baseline = evaluate_module()
    print_result("baseline", baseline)

    split_results = run_split_grid()
    print("\nTop split/grid results")
    print(split_results.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    ablation_results = run_ablation()
    print("\nTop ablation results")
    print(ablation_results.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    rels, lead_lag_results = run_lead_lag()
    print("\nLead-lag relationships used")
    print(rels.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop lead-lag results")
    print(lead_lag_results.head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
