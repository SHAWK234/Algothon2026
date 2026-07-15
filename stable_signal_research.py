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
POSITION_LIMITS = np.full(N_INST, 10_000.0)
POSITION_LIMITS[0] = 100_000.0
COMM_RATES = np.full(N_INST, 0.0001)
COMM_RATES[0] = 0.00002

ROLLING_WINDOWS = [(f"{start}-{start + 100}", start, start + 100) for start in range(50, 401, 25)]
MAIN_WINDOWS = [("250-500", 250, 500), ("250-400", 250, 400), ("400-500", 400, 500)]


def load_prices():
    prices_df = pd.read_csv("prices.txt", sep=r"\s+")
    return prices_df.to_numpy().T


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250.0) * mean_pl / std_pl
    return mean_pl * sharpe**2 / (sharpe**2 + 1.0)


def summarize(pl):
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
    }


def evaluate_dollars(prices, target_dollars, start_day, end_day, return_daily=False):
    cash = 0.0
    cur_pos = np.zeros(N_INST, dtype=int)
    value = 0.0
    commission = 0.0
    daily_pl = []

    for t in range(start_day, end_day + 1):
        current = prices[:, t - 1]
        if t < end_day:
            dollars = np.clip(target_dollars[t], -POSITION_LIMITS, POSITION_LIMITS)
            new_pos = np.divide(
                dollars,
                current,
                out=np.zeros_like(dollars),
                where=current > 0,
            ).astype(int)
            pos_limits = (POSITION_LIMITS / current).astype(int)
            new_pos = np.clip(new_pos, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = cur_pos.copy()

        delta = new_pos - cur_pos
        cash -= float(current.dot(delta)) + commission
        dvolumes = current * np.abs(delta)
        commission = float(np.sum(dvolumes * COMM_RATES))

        cur_pos = new_pos.copy()
        new_value = cash + float(cur_pos.dot(current))
        today_pl = new_value - value
        value = new_value

        if t > start_day:
            daily_pl.append(today_pl)

    result = summarize(daily_pl)
    if return_daily:
        result["daily_pl"] = np.asarray(daily_pl, dtype=float)
    return result


def rolling_metrics(prices, target_dollars):
    rows = []
    for name, start, end in ROLLING_WINDOWS:
        result = evaluate_dollars(prices, target_dollars, start, end)
        rows.append({"window": name, **result})
    df = pd.DataFrame(rows)
    return {
        "roll_mean_score": float(df["score"].mean()),
        "roll_median_score": float(df["score"].median()),
        "roll_min_score": float(df["score"].min()),
        "roll_neg_count": int((df["score"] < 0).sum()),
        "roll_positive_count": int((df["score"] > 0).sum()),
    }


def market_reversal_targets(prices, abs_cap=0.0175):
    n_days = prices.shape[1]
    targets = np.zeros((n_days + 1, N_INST))
    for t in range(1, n_days):
        if t <= 2:
            continue
        current = prices[:, t - 1]
        old = prices[:, t - 3]
        moves = np.divide(current, old, out=np.ones_like(current), where=old > 0) - 1.0
        market_move = float(np.mean(moves))
        if abs(market_move) > abs_cap:
            continue
        if market_move > 0.0015:
            direction = -1.0
            excess = market_move - 0.0015
        elif market_move < -0.0075:
            direction = 1.0
            excess = abs(market_move) - 0.0075
        else:
            continue
        scale = 0.25 + (1.50 - 0.25) * np.tanh(800.0 * max(0.0, excess))
        targets[t] = direction * scale * POSITION_LIMITS
    return targets


def set_single_asset_target(targets, t, asset_idx, signal, threshold=0.0):
    if not np.isfinite(signal) or abs(signal) < threshold:
        return
    targets[t, asset_idx] = np.sign(signal) * POSITION_LIMITS[asset_idx]


def candidate_targets(prices, spec):
    n_days = prices.shape[1]
    targets = np.zeros((n_days + 1, N_INST))
    idx = spec["asset_idx"]
    family = spec["family"]
    mode = spec["mode"]
    sign_mult = 1.0 if mode == "momentum" else -1.0

    returns = np.full_like(prices, np.nan, dtype=float)
    returns[:, 1:] = prices[:, 1:] / prices[:, :-1] - 1.0

    market_norm = (prices / prices[:, [0]]).mean(axis=0)
    asset_norm = prices[idx] / prices[idx, 0]
    spread = np.log(asset_norm) - np.log(market_norm)

    for t in range(1, n_days):
        signal = np.nan
        if family in ("absolute_return", "relative_return"):
            lookback = spec["lookback"]
            lag = 1 if spec["lag_signal"] else 0
            end = t - 1 - lag
            start = end - lookback
            if start >= 0:
                asset_move = prices[idx, end] / prices[idx, start] - 1.0
                if family == "absolute_return":
                    signal = asset_move
                else:
                    other = np.arange(N_INST) != idx
                    other_moves = prices[other, end] / prices[other, start] - 1.0
                    signal = asset_move - float(np.mean(other_moves))
        elif family == "own_return_z":
            lookback = spec["lookback"]
            vol_window = spec["vol_window"]
            end = t - 1
            if end - max(lookback, vol_window) >= 0:
                recent_sum = np.nansum(returns[idx, end - lookback + 1 : end + 1])
                vol = np.nanstd(returns[idx, end - vol_window + 1 : end + 1])
                if vol > 0:
                    signal = recent_sum / vol
        elif family == "market_spread_z":
            z_window = spec["z_window"]
            end = t - 1
            start = end - z_window + 1
            if start >= 0:
                recent = spread[start : end + 1]
                std = float(np.std(recent, ddof=1))
                if std > 0:
                    signal = (spread[end] - float(np.mean(recent))) / std
        else:
            raise ValueError(f"Unknown family: {family}")

        set_single_asset_target(
            targets,
            t,
            idx,
            sign_mult * signal,
            threshold=spec.get("threshold", 0.0),
        )

    return targets


def lead_lag_targets(prices, leader_idx, follower_idx, lag, corr_sign):
    n_days = prices.shape[1]
    targets = np.zeros((n_days + 1, N_INST))
    returns = np.full_like(prices, np.nan, dtype=float)
    returns[:, 1:] = prices[:, 1:] / prices[:, :-1] - 1.0
    for t in range(1, n_days):
        signal_idx = t - lag
        if signal_idx <= 0:
            continue
        signal = corr_sign * returns[leader_idx, signal_idx]
        set_single_asset_target(targets, t, follower_idx, signal)
    return targets


def build_specs():
    specs = []
    for asset_idx, asset in enumerate(ASSETS):
        for family in ["absolute_return", "relative_return"]:
            for lookback in [2, 5, 10, 20, 40, 60]:
                for mode in ["momentum", "reversal"]:
                    for lag_signal in [False, True]:
                        specs.append({
                            "name": f"{asset}_{family}_w{lookback}_{mode}_{'lag' if lag_signal else 'cur'}",
                            "asset_idx": asset_idx,
                            "asset": asset,
                            "family": family,
                            "lookback": lookback,
                            "mode": mode,
                            "lag_signal": lag_signal,
                            "threshold": 0.0,
                        })

        for lookback, vol_window in itertools.product([2, 5, 10, 20], [20, 40, 60]):
            for mode in ["momentum", "reversal"]:
                for threshold in [0.5, 1.0]:
                    specs.append({
                        "name": f"{asset}_own_z_l{lookback}_v{vol_window}_{mode}_e{threshold}",
                        "asset_idx": asset_idx,
                        "asset": asset,
                        "family": "own_return_z",
                        "lookback": lookback,
                        "vol_window": vol_window,
                        "mode": mode,
                        "threshold": threshold,
                    })

        for z_window in [10, 20, 40, 60]:
            for mode in ["momentum", "reversal"]:
                for threshold in [0.5, 1.0]:
                    specs.append({
                        "name": f"{asset}_market_spread_z_w{z_window}_{mode}_e{threshold}",
                        "asset_idx": asset_idx,
                        "asset": asset,
                        "family": "market_spread_z",
                        "z_window": z_window,
                        "mode": mode,
                        "threshold": threshold,
                    })
    return specs


def evaluate_candidate(prices, spec, market_daily):
    targets = candidate_targets(prices, spec)
    full = evaluate_dollars(prices, targets, 250, 500, return_daily=True)
    roll = rolling_metrics(prices, targets)
    daily = full.pop("daily_pl")
    corr_to_market = 0.0
    if np.std(daily) > 0 and np.std(market_daily) > 0:
        corr_to_market = float(np.corrcoef(daily, market_daily)[0, 1])
    return {**spec, **{f"full_{k}": v for k, v in full.items()}, **roll, "corr_to_market": corr_to_market}, targets


def evaluate_lead_lag_candidates(prices, market_daily):
    returns = prices[:, 1:250] / prices[:, :249] - 1.0
    raw = []
    for lag in [1, 2, 3, 5, 10]:
        for leader_idx, follower_idx in itertools.permutations(range(N_INST), 2):
            x = returns[leader_idx, :-lag]
            y = returns[follower_idx, lag:]
            if len(x) < 80 or np.std(x) <= 0 or np.std(y) <= 0:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            raw.append((abs(corr), corr, leader_idx, follower_idx, lag))
    raw.sort(reverse=True)

    rows = []
    targets_by_name = {}
    for abs_corr, corr, leader_idx, follower_idx, lag in raw[:250]:
        corr_sign = np.sign(corr)
        targets = lead_lag_targets(prices, leader_idx, follower_idx, lag, corr_sign)
        name = f"{ASSETS[leader_idx]}_lead{lag}_{ASSETS[follower_idx]}_corr{corr_sign:+.0f}"
        full = evaluate_dollars(prices, targets, 250, 500, return_daily=True)
        roll = rolling_metrics(prices, targets)
        daily = full.pop("daily_pl")
        corr_to_market = 0.0
        if np.std(daily) > 0 and np.std(market_daily) > 0:
            corr_to_market = float(np.corrcoef(daily, market_daily)[0, 1])
        rows.append({
            "name": name,
            "asset_idx": follower_idx,
            "asset": ASSETS[follower_idx],
            "family": "lead_lag",
            "mode": "momentum",
            "leader": ASSETS[leader_idx],
            "lag": lag,
            "train_corr": corr,
            **{f"full_{k}": v for k, v in full.items()},
            **roll,
            "corr_to_market": corr_to_market,
        })
        targets_by_name[name] = targets
    return rows, targets_by_name


def portfolio_from_candidates(targets_by_name, candidate_names):
    if not candidate_names:
        return None
    total = None
    for name in candidate_names:
        if total is None:
            total = targets_by_name[name].copy()
        else:
            total += targets_by_name[name]
    return np.clip(total, -POSITION_LIMITS, POSITION_LIMITS)


def main():
    prices = load_prices()
    market_targets = market_reversal_targets(prices, abs_cap=0.0175)
    market_full = evaluate_dollars(prices, market_targets, 250, 500, return_daily=True)
    market_daily = market_full["daily_pl"]

    rows = []
    targets_by_name = {}
    for spec in build_specs():
        row, targets = evaluate_candidate(prices, spec, market_daily)
        rows.append(row)
        targets_by_name[spec["name"]] = targets

    lead_rows, lead_targets = evaluate_lead_lag_candidates(prices, market_daily)
    rows.extend(lead_rows)
    targets_by_name.update(lead_targets)

    candidates = pd.DataFrame(rows)
    candidates.to_csv("stable_signal_candidates.csv", index=False)

    stable = candidates[
        (candidates["roll_neg_count"] <= 2)
        & (candidates["roll_min_score"] > -10)
        & (candidates["full_score"] > 5)
        & (candidates["full_sharpe"] > 1.0)
    ].copy()
    stable["abs_corr_to_market"] = stable["corr_to_market"].abs()
    stable = stable.sort_values(
        ["roll_neg_count", "roll_min_score", "full_score"],
        ascending=[True, False, False],
    )
    stable.to_csv("stable_signal_candidates_filtered.csv", index=False)

    print("Current market core, 250-500")
    print(
        pd.DataFrame([{k: v for k, v in market_full.items() if k != "daily_pl"}])
        .round(3)
        .to_string(index=False)
    )

    print("\nMost stable individual signals")
    print(
        stable[
            [
                "name", "asset", "family", "mode", "full_score", "full_mean_pl",
                "full_std_pl", "full_sharpe", "roll_neg_count", "roll_min_score",
                "roll_mean_score", "corr_to_market",
            ]
        ]
        .head(30)
        .round(3)
        .to_string(index=False)
    )

    print("\nStable low-correlation signals")
    low_corr = stable[stable["abs_corr_to_market"] < 0.25]
    print(
        low_corr[
            [
                "name", "asset", "family", "mode", "full_score", "full_mean_pl",
                "full_std_pl", "full_sharpe", "roll_neg_count", "roll_min_score",
                "roll_mean_score", "corr_to_market",
            ]
        ]
        .head(30)
        .round(3)
        .to_string(index=False)
    )

    # Keep one signal per asset so a portfolio is not just the same asset clipped repeatedly.
    selected = (
        stable.sort_values(
            ["roll_neg_count", "roll_min_score", "full_score"],
            ascending=[True, False, False],
        )
        .drop_duplicates("asset_idx")
    )
    selected_low_corr = (
        low_corr.sort_values(
            ["roll_neg_count", "roll_min_score", "full_score"],
            ascending=[True, False, False],
        )
        .drop_duplicates("asset_idx")
    )

    portfolio_rows = []
    for label, source in [("stable", selected), ("low_corr", selected_low_corr)]:
        for k in [3, 5, 10, 15, 20, 30]:
            names = source["name"].head(k).tolist()
            targets = portfolio_from_candidates(targets_by_name, names)
            if targets is None:
                continue
            full = evaluate_dollars(prices, targets, 250, 500)
            roll = rolling_metrics(prices, targets)
            portfolio_rows.append({
                "portfolio": f"{label}_top{k}",
                "n_signals": len(names),
                **{f"full_{key}": value for key, value in full.items()},
                **roll,
                "signals": ";".join(names),
            })

            for market_weight, overlay_weight in [(0.75, 0.25), (0.50, 0.50), (0.25, 0.75)]:
                combo = market_weight * market_targets + overlay_weight * targets
                combo_full = evaluate_dollars(prices, combo, 250, 500)
                combo_roll = rolling_metrics(prices, combo)
                portfolio_rows.append({
                    "portfolio": f"market{market_weight:.2f}_{label}_top{k}_{overlay_weight:.2f}",
                    "n_signals": len(names),
                    **{f"full_{key}": value for key, value in combo_full.items()},
                    **combo_roll,
                    "signals": ";".join(names),
                })

    portfolios = pd.DataFrame(portfolio_rows).sort_values(
        ["roll_neg_count", "full_score", "roll_min_score"],
        ascending=[True, False, False],
    )
    portfolios.to_csv("stable_signal_portfolios.csv", index=False)

    print("\nSignal portfolios")
    print(
        portfolios[
            [
                "portfolio", "n_signals", "full_score", "full_mean_pl",
                "full_std_pl", "full_sharpe", "roll_neg_count", "roll_min_score",
                "roll_mean_score",
            ]
        ]
        .head(30)
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
