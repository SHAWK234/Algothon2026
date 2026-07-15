import itertools

import numpy as np
import pandas as pd


ASSETS = [
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT", "HUXZ",
    "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET", "ANSO", "DIHO",
    "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF", "RRES", "CTGI", "ALUT",
    "ACAC", "SRTX", "GARI", "RCRI", "ACIX", "CCNS", "MTNS", "IHOZ", "NAYO",
    "FWWG", "EELT", "HRND", "AETS", "ULXY", "BLBT", "BENI", "ITPA", "HTRK",
    "NGTE", "ILVX", "FCSG", "FARS", "MHRM", "EAFC",
]

N_INST = len(ASSETS)
LOOKBACK = 2
SHORT_THRESHOLD = 0.0015
LONG_THRESHOLD = 0.0075
MIN_SCALE = 0.25
MAX_SCALE = 1.50
SCALE_SPEED = 800.0

POSITION_LIMITS = np.full(N_INST, 10_000.0)
POSITION_LIMITS[0] = 100_000.0
COMM_RATES = np.full(N_INST, 0.0001)
COMM_RATES[0] = 0.00002

ROLLING_WINDOWS = [(f"{start}-{start + 100}", start, start + 100) for start in range(50, 401, 25)]


def load_prices():
    return pd.read_csv("prices.txt", sep=r"\s+").to_numpy().T


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250.0) * mean_pl / std_pl
    return mean_pl * sharpe**2 / (sharpe**2 + 1.0)


def summary(pl):
    pl = np.asarray(pl, dtype=float)
    mean_pl = float(np.mean(pl))
    std_pl = float(np.std(pl))
    sharpe = 0.0 if std_pl <= 0 else float(np.sqrt(250.0) * mean_pl / std_pl)
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": sharpe,
        "score": float(score(mean_pl, std_pl)),
        "worst_day": float(np.min(pl)),
        "q05": float(np.quantile(pl, 0.05)),
        "q95": float(np.quantile(pl, 0.95)),
    }


def market_direction_and_scale(prices, t):
    if t <= LOOKBACK:
        return 0.0, 0.0

    current = prices[:, t - 1]
    old = prices[:, t - 1 - LOOKBACK]
    moves = np.divide(current, old, out=np.ones_like(current), where=old > 0) - 1.0
    market_move = float(np.mean(moves))

    if market_move > SHORT_THRESHOLD:
        direction = -1.0
        excess = market_move - SHORT_THRESHOLD
    elif market_move < -LONG_THRESHOLD:
        direction = 1.0
        excess = abs(market_move) - LONG_THRESHOLD
    else:
        return 0.0, 0.0

    scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * np.tanh(SCALE_SPEED * max(0.0, excess))
    return direction, scale


def raw_positions(prices, t, multipliers):
    direction, scale = market_direction_and_scale(prices, t)
    if direction == 0.0:
        return np.zeros(N_INST, dtype=int)

    current = prices[:, t - 1]
    dollar_targets = direction * scale * POSITION_LIMITS * multipliers
    positions = np.divide(
        dollar_targets,
        current,
        out=np.zeros_like(dollar_targets),
        where=current > 0,
    ).astype(int)
    pos_limits = (POSITION_LIMITS / current).astype(int)
    return np.clip(positions, -pos_limits, pos_limits).astype(int)


def precompute_raw_next_pl(prices, multipliers):
    raw_pos = np.zeros((prices.shape[1] + 1, N_INST), dtype=int)
    raw_next_asset_pl = np.full((prices.shape[1] + 1, N_INST), np.nan)

    for t in range(1, prices.shape[1]):
        pos = raw_positions(prices, t, multipliers)
        raw_pos[t] = pos
        current = prices[:, t - 1]
        nxt = prices[:, t]
        trade_commission = current * np.abs(pos) * COMM_RATES
        raw_next_asset_pl[t] = pos * (nxt - current) - trade_commission

    return raw_pos, raw_next_asset_pl


def desired_positions_from_gate(raw_pos, raw_next_asset_pl, gate_window=None, gate_threshold=0.0):
    desired = raw_pos.copy()
    if gate_window is None:
        return desired

    total_raw_pl = np.nansum(raw_next_asset_pl, axis=1)
    for t in range(1, len(desired)):
        start = max(1, t - gate_window)
        recent = total_raw_pl[start:t]
        recent = recent[np.isfinite(recent)]
        if len(recent) < max(5, min(20, gate_window // 2)):
            continue
        if float(np.mean(recent)) < gate_threshold:
            desired[t] = 0
    return desired


def evaluate_desired(prices, desired, start_day, end_day, collect_asset_pl=False):
    cash = 0.0
    cur_pos = np.zeros(N_INST, dtype=int)
    value = 0.0
    commission = 0.0
    prev_commission_by_asset = np.zeros(N_INST)
    prev_prices = prices[:, start_day - 1]
    daily_pl = []
    asset_pl = []

    for t in range(start_day, end_day + 1):
        current = prices[:, t - 1]
        if t < end_day:
            new_pos = desired[t].astype(int)
            pos_limits = (POSITION_LIMITS / current).astype(int)
            new_pos = np.clip(new_pos, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= float(current.dot(delta)) + commission
        dvolume_by_asset = current * np.abs(delta)
        commission_by_asset = dvolume_by_asset * COMM_RATES
        commission = float(np.sum(commission_by_asset))

        previous_pos = cur_pos.copy()
        previous_value = value
        cur_pos = new_pos.copy()
        value = cash + float(cur_pos.dot(current))
        today_pl = value - previous_value

        if t > start_day:
            daily_pl.append(today_pl)
            if collect_asset_pl:
                asset_pl.append(previous_pos * (current - prev_prices) - prev_commission_by_asset)

        prev_prices = current
        prev_commission_by_asset = commission_by_asset

    result = summary(daily_pl)
    if collect_asset_pl:
        result["daily_pl"] = np.asarray(daily_pl)
        result["asset_pl"] = np.asarray(asset_pl)
    return result


def rolling_stats(prices, desired):
    rows = []
    for name, start, end in ROLLING_WINDOWS:
        result = evaluate_desired(prices, desired, start, end)
        rows.append({"window": name, **result})
    df = pd.DataFrame(rows)
    return {
        "roll_mean_score": float(df["score"].mean()),
        "roll_median_score": float(df["score"].median()),
        "roll_min_score": float(df["score"].min()),
        "roll_neg_count": int((df["score"] < 0).sum()),
        "roll_mean_std": float(df["std_pl"].mean()),
        "roll_mean_worst_day": float(df["worst_day"].mean()),
    }


def standalone_asset_report(prices):
    rows = []
    for idx, asset in enumerate(ASSETS):
        multipliers = np.zeros(N_INST)
        multipliers[idx] = 1.0
        raw_pos, raw_asset_pl = precompute_raw_next_pl(prices, multipliers)

        for gate_name, gate_window, gate_threshold in [
            ("raw", None, 0.0),
            ("gate20", 20, 0.0),
            ("gate20_loose", 20, -250.0),
        ]:
            desired = desired_positions_from_gate(raw_pos, raw_asset_pl, gate_window, gate_threshold)
            full = evaluate_desired(prices, desired, 250, 500)
            rstats = rolling_stats(prices, desired)
            rows.append({
                "asset_idx": idx,
                "asset": asset,
                "variant": gate_name,
                **{f"full_{k}": v for k, v in full.items()},
                **rstats,
            })
    return pd.DataFrame(rows)


def portfolio_report(prices, multipliers, name, gate_window=None, gate_threshold=0.0):
    raw_pos, raw_asset_pl = precompute_raw_next_pl(prices, multipliers)
    desired = desired_positions_from_gate(raw_pos, raw_asset_pl, gate_window, gate_threshold)
    full = evaluate_desired(prices, desired, 250, 500, collect_asset_pl=True)
    rstats = rolling_stats(prices, desired)
    asset_pl = full["asset_pl"]
    daily_pl = full["daily_pl"]

    total_var = float(np.var(daily_pl))
    rows = []
    worst_day_indices = np.argsort(daily_pl)[:20]
    for idx, asset in enumerate(ASSETS):
        contribution = asset_pl[:, idx]
        cov_contribution = 0.0
        if total_var > 0:
            cov_contribution = float(np.cov(contribution, daily_pl, ddof=0)[0, 1] / total_var)
        rows.append({
            "portfolio": name,
            "asset_idx": idx,
            "asset": asset,
            "multiplier": float(multipliers[idx]),
            **{f"asset_{k}": v for k, v in summary(contribution).items()},
            "variance_contribution_frac": cov_contribution,
            "avg_contribution_on_worst20": float(np.mean(contribution[worst_day_indices])),
            "worst20_total_avg": float(np.mean(daily_pl[worst_day_indices])),
        })

    port_row = {
        "portfolio": name,
        "gate_window": gate_window if gate_window is not None else 0,
        "gate_threshold": gate_threshold,
        **{f"full_{k}": v for k, v in full.items() if k not in ("daily_pl", "asset_pl")},
        **rstats,
    }
    return port_row, pd.DataFrame(rows)


def build_weight_sets(standalone):
    gate20 = standalone[standalone["variant"] == "gate20"].copy()
    raw = standalone[standalone["variant"] == "raw"].copy()
    weights = {}

    weights["all_full"] = np.ones(N_INST)

    top_robust = gate20.sort_values(
        ["roll_neg_count", "roll_min_score", "full_sharpe"],
        ascending=[True, False, False],
    ).head(20)
    w = np.zeros(N_INST)
    w[top_robust["asset_idx"].to_numpy()] = 1.0
    weights["top20_robust_assets"] = w

    positive_recent = gate20[
        (gate20["full_mean_pl"] > 0)
        & (gate20["roll_neg_count"] <= 1)
        & (gate20["roll_min_score"] > -50)
    ]
    w = np.zeros(N_INST)
    w[positive_recent["asset_idx"].to_numpy()] = 1.0
    weights["positive_low_bad_window"] = w

    # Inverse-risk weights keep the broad signal but prevent a few noisy names from
    # dominating the daily PnL standard deviation.
    w = np.zeros(N_INST)
    std = raw["full_std_pl"].replace(0, np.nan).to_numpy()
    inv = np.nanmedian(std) / std
    inv = np.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0)
    w[raw["asset_idx"].to_numpy()] = np.clip(inv, 0.15, 1.0)
    weights["inverse_standalone_std"] = w

    w = np.zeros(N_INST)
    # Risk-adjust by downside, but keep all assets at least lightly active.
    downside = raw["full_q05"].abs().replace(0, np.nan).to_numpy()
    inv_downside = np.nanmedian(downside) / downside
    inv_downside = np.nan_to_num(inv_downside, nan=0.0, posinf=0.0, neginf=0.0)
    w[raw["asset_idx"].to_numpy()] = np.clip(inv_downside, 0.10, 1.0)
    weights["inverse_downside_q05"] = w

    return weights


def main():
    prices = load_prices()
    standalone = standalone_asset_report(prices)
    standalone.to_csv("high_std_standalone_assets.csv", index=False)

    print("Worst standalone asset std, gate20, 250-500")
    gate20 = standalone[standalone["variant"] == "gate20"]
    print(
        gate20.sort_values("full_std_pl", ascending=False)
        .head(12)[
            [
                "asset_idx", "asset", "full_mean_pl", "full_std_pl",
                "full_sharpe", "full_score", "full_worst_day",
                "roll_neg_count", "roll_min_score",
            ]
        ]
        .to_string(index=False)
    )

    print("\nBest robust standalone assets, gate20")
    print(
        gate20.sort_values(
            ["roll_neg_count", "roll_min_score", "full_score"],
            ascending=[True, False, False],
        )
        .head(20)[
            [
                "asset_idx", "asset", "full_mean_pl", "full_std_pl",
                "full_sharpe", "full_score", "full_worst_day",
                "roll_neg_count", "roll_min_score",
            ]
        ]
        .to_string(index=False)
    )

    weight_sets = build_weight_sets(standalone)
    portfolio_rows = []
    attribution_rows = []
    for weight_name, multipliers in weight_sets.items():
        for gate_name, gate_window, gate_threshold in [
            ("raw", None, 0.0),
            ("gate20", 20, 0.0),
            ("gate20_loose", 20, -250.0),
            ("gate40_loose", 40, -500.0),
        ]:
            name = f"{weight_name}_{gate_name}"
            port_row, attr = portfolio_report(prices, multipliers, name, gate_window, gate_threshold)
            active = int(np.sum(multipliers > 0))
            port_row["active_assets"] = active
            port_row["avg_multiplier"] = float(np.mean(multipliers[multipliers > 0])) if active else 0.0
            portfolio_rows.append(port_row)
            attribution_rows.append(attr)

    portfolio_df = pd.DataFrame(portfolio_rows).sort_values(
        ["roll_neg_count", "full_score", "roll_min_score"],
        ascending=[True, False, False],
    )
    attribution_df = pd.concat(attribution_rows, ignore_index=True)
    portfolio_df.to_csv("high_std_portfolio_candidates.csv", index=False)
    attribution_df.to_csv("high_std_portfolio_attribution.csv", index=False)

    print("\nPortfolio candidates")
    print(
        portfolio_df[
            [
                "portfolio", "active_assets", "avg_multiplier", "full_score",
                "full_mean_pl", "full_std_pl", "full_sharpe", "full_worst_day",
                "roll_neg_count", "roll_min_score", "roll_mean_score",
            ]
        ]
        .head(24)
        .to_string(index=False)
    )

    best_portfolio = portfolio_df.iloc[0]["portfolio"]
    print(f"\nTop variance contributors for {best_portfolio}")
    print(
        attribution_df[attribution_df["portfolio"] == best_portfolio]
        .sort_values("variance_contribution_frac", ascending=False)
        .head(15)[
            [
                "asset_idx", "asset", "multiplier", "asset_mean_pl",
                "asset_std_pl", "asset_sharpe", "asset_worst_day",
                "variance_contribution_frac", "avg_contribution_on_worst20",
            ]
        ]
        .to_string(index=False)
    )

    print("\nWorst worst-day contributors under all_full_gate20")
    print(
        attribution_df[attribution_df["portfolio"] == "all_full_gate20"]
        .sort_values("avg_contribution_on_worst20")
        .head(15)[
            [
                "asset_idx", "asset", "multiplier", "asset_mean_pl",
                "asset_std_pl", "asset_worst_day", "variance_contribution_frac",
                "avg_contribution_on_worst20",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
