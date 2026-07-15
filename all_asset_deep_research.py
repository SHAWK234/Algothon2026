import itertools
from collections import defaultdict

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
ASSET_TO_IDX = {asset: idx for idx, asset in enumerate(ASSETS)}

POSITION_LIMITS = np.full(N_INST, 10_000.0)
POSITION_LIMITS[0] = 100_000.0

COMM_RATES = np.full(N_INST, 0.0001)
COMM_RATES[0] = 0.00002

CV_WINDOWS = [
    ("250-350", 250, 350),
    ("300-400", 300, 400),
    ("350-450", 350, 450),
    ("400-500", 400, 500),
]


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250) * mean_pl / std_pl
    return mean_pl * sharpe**2 / (sharpe**2 + 1.0)


def sharpe(pl):
    pl = np.asarray(pl, dtype=float)
    std = np.std(pl)
    if std <= 0:
        return 0.0
    return np.sqrt(250) * np.mean(pl) / std


def evaluate_interval(prices, start_day, end_day, position_fn):
    cash = 0.0
    cur_pos = np.zeros(N_INST)
    total_dvolume = 0.0
    value = 0.0
    commission = 0.0
    daily_pl = []

    for t in range(start_day, end_day + 1):
        hist = prices[:, :t]
        cur_prices = hist[:, -1]

        if t < end_day:
            new_pos_orig = position_fn(hist)
            pos_limits = (POSITION_LIMITS / cur_prices).astype(int)
            new_pos = np.clip(new_pos_orig, -pos_limits, pos_limits).astype(int)
        else:
            new_pos = np.array(cur_pos)

        delta_pos = new_pos - cur_pos
        cash -= cur_prices.dot(delta_pos) + commission

        dvolumes = cur_prices * np.abs(delta_pos)
        total_dvolume += np.sum(dvolumes)
        commission = np.sum(dvolumes * COMM_RATES)

        cur_pos = np.array(new_pos)
        today_pl = cash + cur_pos.dot(cur_prices) - value
        value = cash + cur_pos.dot(cur_prices)

        if t > start_day:
            daily_pl.append(today_pl)

    daily_pl = np.array(daily_pl)
    mean_pl = float(np.mean(daily_pl))
    std_pl = float(np.std(daily_pl))
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": sharpe(daily_pl),
        "score": score(mean_pl, std_pl),
        "value": value,
        "dvolume": total_dvolume,
    }


def target_position_from_dollars(dollar_targets, current_prices):
    return np.divide(
        dollar_targets,
        current_prices,
        out=np.zeros_like(dollar_targets),
        where=current_prices > 0,
    ).astype(int)


def asset_profile(prices_df):
    returns = prices_df.pct_change()
    market = returns.mean(axis=1)
    rows = []

    for asset in ASSETS:
        r = returns[asset].dropna()
        market_ex = returns.drop(columns=[asset]).mean(axis=1)
        aligned = pd.concat([returns[asset], market_ex], axis=1).dropna()
        asset_r = aligned.iloc[:, 0]
        market_r = aligned.iloc[:, 1]
        beta = np.cov(asset_r, market_r)[0, 1] / np.var(market_r) if np.var(market_r) > 0 else np.nan
        residual = asset_r - beta * market_r
        rows.append(
            {
                "asset": asset,
                "mean": r.mean(),
                "vol": r.std(),
                "buy_hold_sharpe": np.sqrt(250) * r.mean() / r.std() if r.std() > 0 else np.nan,
                "corr_market": asset_r.corr(market_r),
                "beta_market": beta,
                "residual_vol": residual.std(),
                "ac1": r.autocorr(1),
                "ac2": r.autocorr(2),
                "ac5": r.autocorr(5),
                "ac10": r.autocorr(10),
                "abs_ac_sum": sum(abs(r.autocorr(lag)) for lag in [1, 2, 3, 5, 10]),
            }
        )

    profile = pd.DataFrame(rows).sort_values("abs_ac_sum", ascending=False)
    return profile, market


def market_factor_summary(prices_df):
    returns = prices_df.pct_change().dropna()
    corr = returns.corr()
    avg_abs_corr = (
        corr.where(~np.eye(len(corr), dtype=bool))
        .abs()
        .stack()
        .mean()
    )

    demeaned = returns - returns.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(demeaned.to_numpy(), full_matrices=False)
    explained = singular_values**2 / np.sum(singular_values**2)
    return {
        "avg_abs_pair_corr": avg_abs_corr,
        "pc1_var_share": explained[0],
        "pc2_var_share": explained[1],
        "pc3_var_share": explained[2],
    }


def signed_feature_position(asset_idx, feature_fn, mode):
    def position_fn(hist):
        feature = feature_fn(hist, asset_idx)
        if not np.isfinite(feature) or feature == 0:
            return np.zeros(N_INST)
        sign = np.sign(feature)
        if mode == "reversal":
            sign = -sign
        dollars = np.zeros(N_INST)
        dollars[asset_idx] = sign * POSITION_LIMITS[asset_idx]
        return target_position_from_dollars(dollars, hist[:, -1])

    return position_fn


def absolute_return_feature(window, lag_signal):
    def feature(hist, asset_idx):
        signal_prices = hist[:, :-1] if lag_signal else hist
        if signal_prices.shape[1] <= window:
            return np.nan
        return signal_prices[asset_idx, -1] / signal_prices[asset_idx, -1 - window] - 1.0

    return feature


def relative_return_feature(window, lag_signal):
    def feature(hist, asset_idx):
        signal_prices = hist[:, :-1] if lag_signal else hist
        if signal_prices.shape[1] <= window:
            return np.nan
        asset_move = signal_prices[asset_idx, -1] / signal_prices[asset_idx, -1 - window] - 1.0
        other_idx = [idx for idx in range(N_INST) if idx != asset_idx]
        other_moves = signal_prices[other_idx, -1] / signal_prices[other_idx, -1 - window] - 1.0
        return asset_move - np.mean(other_moves)

    return feature


def top_correlated_basket(prices, asset_idx, cutoff, top_n):
    returns = prices[:, 1:cutoff] / prices[:, : cutoff - 1] - 1.0
    target = returns[asset_idx]
    rows = []
    for idx in range(N_INST):
        if idx == asset_idx:
            continue
        corr = np.corrcoef(target, returns[idx])[0, 1]
        if np.isfinite(corr):
            rows.append((abs(corr), idx))
    rows.sort(reverse=True)
    return [idx for _, idx in rows[:top_n]]


def residual_z_series(hist, asset_idx, basket_indices, window, lag_signal):
    signal_prices = hist[:, :-1] if lag_signal else hist
    if signal_prices.shape[1] < window + 2:
        return np.array([])

    basket_norm = (
        signal_prices[basket_indices] / signal_prices[basket_indices][:, [0]]
    ).mean(axis=0)
    basket_price = basket_norm * signal_prices[asset_idx, 0]
    spread = np.log(signal_prices[asset_idx]) - np.log(basket_price)

    z_scores = []
    for end_idx in range(window - 1, len(spread)):
        recent = spread[end_idx - window + 1 : end_idx + 1]
        std = np.std(recent, ddof=1)
        if not np.isfinite(std) or std <= 0:
            z_scores.append(np.nan)
        else:
            z_scores.append((spread[end_idx] - np.mean(recent)) / std)
    return np.array(z_scores)


def replay_threshold_state(z_scores, entry_z, exit_z, mode):
    state = 0
    for z in z_scores:
        if not np.isfinite(z):
            continue
        rich_state = -1 if mode == "reversal" else 1
        cheap_state = 1 if mode == "reversal" else -1

        if state == 0:
            if z >= entry_z:
                state = rich_state
            elif z <= -entry_z:
                state = cheap_state
        elif state == 1:
            if z >= entry_z:
                state = rich_state
            elif z >= -exit_z:
                state = 0
        elif state == -1:
            if z <= -entry_z:
                state = cheap_state
            elif z <= exit_z:
                state = 0
    return state


def transition_threshold_state(previous_state, z, entry_z, exit_z, mode):
    if not np.isfinite(z):
        return previous_state

    rich_state = -1 if mode == "reversal" else 1
    cheap_state = 1 if mode == "reversal" else -1

    if previous_state == 0:
        if z >= entry_z:
            return rich_state
        if z <= -entry_z:
            return cheap_state
        return 0

    if previous_state == 1:
        if z >= entry_z:
            return rich_state
        if z >= -exit_z:
            return 0
        return 1

    if z <= -entry_z:
        return cheap_state
    if z <= exit_z:
        return 0
    return -1


def residual_z_position(asset_idx, basket_indices, window, entry_z, exit_z, mode, lag_signal):
    state = 0

    def position_fn(hist):
        nonlocal state
        signal_prices = hist[:, :-1] if lag_signal else hist
        if signal_prices.shape[1] < window + 2:
            return np.zeros(N_INST)

        basket_norm = (
            signal_prices[basket_indices] / signal_prices[basket_indices][:, [0]]
        ).mean(axis=0)
        basket_price = basket_norm * signal_prices[asset_idx, 0]
        spread = np.log(signal_prices[asset_idx]) - np.log(basket_price)
        recent = spread[-window:]
        spread_std = np.std(recent, ddof=1)
        if not np.isfinite(spread_std) or spread_std <= 0:
            return np.zeros(N_INST)

        z = (spread[-1] - np.mean(recent)) / spread_std
        state = transition_threshold_state(state, z, entry_z, exit_z, mode)
        dollars = np.zeros(N_INST)
        dollars[asset_idx] = state * POSITION_LIMITS[asset_idx]
        return target_position_from_dollars(dollars, hist[:, -1])

    return position_fn


def cross_sectional_position(window, top_k, mode, lag_signal):
    def position_fn(hist):
        signal_prices = hist[:, :-1] if lag_signal else hist
        if signal_prices.shape[1] <= window:
            return np.zeros(N_INST)
        moves = signal_prices[:, -1] / signal_prices[:, -1 - window] - 1.0
        ranks = np.argsort(moves)
        low = ranks[:top_k]
        high = ranks[-top_k:]
        dollars = np.zeros(N_INST)
        if mode == "momentum":
            dollars[high] = POSITION_LIMITS[high]
            dollars[low] = -POSITION_LIMITS[low]
        else:
            dollars[high] = -POSITION_LIMITS[high]
            dollars[low] = POSITION_LIMITS[low]
        return target_position_from_dollars(dollars, hist[:, -1])

    return position_fn


def market_direction_position(window, mode, lag_signal):
    def position_fn(hist):
        signal_prices = hist[:, :-1] if lag_signal else hist
        if signal_prices.shape[1] <= window:
            return np.zeros(N_INST)
        moves = signal_prices[:, -1] / signal_prices[:, -1 - window] - 1.0
        signal = np.sign(np.mean(moves))
        if mode == "reversal":
            signal = -signal
        dollars = signal * POSITION_LIMITS
        return target_position_from_dollars(dollars, hist[:, -1])

    return position_fn


def evaluate_candidate(prices, name, family, params, factory):
    rows = []
    scores = []
    means = []
    sharpes = []

    for label, start_day, end_day in CV_WINDOWS:
        position_fn = factory(start_day)
        result = evaluate_interval(prices, start_day, end_day, position_fn)
        scores.append(result["score"])
        means.append(result["mean_pl"])
        sharpes.append(result["sharpe"])
        rows.append(
            {
                "name": name,
                "family": family,
                "window_label": label,
                **params,
                **result,
            }
        )

    summary = {
        "name": name,
        "family": family,
        **params,
        "min_score": float(np.min(scores)),
        "avg_score": float(np.mean(scores)),
        "last_score": float(scores[-1]),
        "positive_windows": int(np.sum(np.array(scores) > 0)),
        "avg_mean_pl": float(np.mean(means)),
        "avg_sharpe": float(np.mean(sharpes)),
    }
    return summary, rows


def scan_strategies(prices):
    summaries = []
    detail_rows = []

    for asset_idx, window, feature_kind, mode, lag_signal in itertools.product(
        range(N_INST),
        [2, 3, 5, 10, 20, 40, 60],
        ["absolute", "relative"],
        ["momentum", "reversal"],
        [True, False],
    ):
        feature_fn = (
            absolute_return_feature(window, lag_signal)
            if feature_kind == "absolute"
            else relative_return_feature(window, lag_signal)
        )
        params = {
            "asset": ASSETS[asset_idx],
            "asset_idx": asset_idx,
            "lookback": window,
            "feature_kind": feature_kind,
            "mode": mode,
            "lag_signal": lag_signal,
        }
        name = f"{ASSETS[asset_idx]}_{feature_kind}_{mode}_{window}_{'lag' if lag_signal else 'cur'}"
        summary, rows = evaluate_candidate(
            prices,
            name,
            "single_asset_return",
            params,
            lambda _start, asset_idx=asset_idx, feature_fn=feature_fn, mode=mode: signed_feature_position(
                asset_idx, feature_fn, mode
            ),
        )
        summaries.append(summary)
        detail_rows.extend(rows)

    for asset_idx, top_n, window, entry_z, exit_z, mode, lag_signal in itertools.product(
        range(N_INST),
        [10],
        [10, 20, 40],
        [0.5, 1.0],
        [0.0],
        ["reversal", "momentum"],
        [False],
    ):
        if exit_z > entry_z:
            continue
        params = {
            "asset": ASSETS[asset_idx],
            "asset_idx": asset_idx,
            "top_n": top_n,
            "z_window": window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "mode": mode,
            "lag_signal": lag_signal,
        }
        name = f"{ASSETS[asset_idx]}_resid_{mode}_top{top_n}_w{window}_e{entry_z}_x{exit_z}_{'lag' if lag_signal else 'cur'}"

        def factory(start_day, asset_idx=asset_idx, top_n=top_n, window=window, entry_z=entry_z, exit_z=exit_z, mode=mode, lag_signal=lag_signal):
            basket = top_correlated_basket(prices, asset_idx, start_day, top_n)
            return residual_z_position(asset_idx, basket, window, entry_z, exit_z, mode, lag_signal)

        summary, rows = evaluate_candidate(prices, name, "residual_z", params, factory)
        summaries.append(summary)
        detail_rows.extend(rows)

    for window, top_k, mode, lag_signal in itertools.product(
        [2, 3, 5, 10, 20, 40],
        [3, 5, 10, 15],
        ["momentum", "reversal"],
        [True, False],
    ):
        params = {
            "lookback": window,
            "top_k": top_k,
            "mode": mode,
            "lag_signal": lag_signal,
        }
        name = f"cross_sectional_{mode}_w{window}_k{top_k}_{'lag' if lag_signal else 'cur'}"
        summary, rows = evaluate_candidate(
            prices,
            name,
            "cross_sectional",
            params,
            lambda _start, window=window, top_k=top_k, mode=mode, lag_signal=lag_signal: cross_sectional_position(
                window, top_k, mode, lag_signal
            ),
        )
        summaries.append(summary)
        detail_rows.extend(rows)

    for window, mode, lag_signal in itertools.product(
        [2, 3, 5, 10, 20, 40],
        ["momentum", "reversal"],
        [True, False],
    ):
        params = {
            "lookback": window,
            "mode": mode,
            "lag_signal": lag_signal,
        }
        name = f"market_{mode}_w{window}_{'lag' if lag_signal else 'cur'}"
        summary, rows = evaluate_candidate(
            prices,
            name,
            "market_direction",
            params,
            lambda _start, window=window, mode=mode, lag_signal=lag_signal: market_direction_position(
                window, mode, lag_signal
            ),
        )
        summaries.append(summary)
        detail_rows.extend(rows)

    return pd.DataFrame(summaries), pd.DataFrame(detail_rows)


def summarise_asset_opportunities(summary):
    by_asset_rows = []
    asset_rows = summary[summary["asset"].notna()].copy()
    for asset, group in asset_rows.groupby("asset"):
        best = group.sort_values(["min_score", "avg_score"], ascending=False).iloc[0]
        by_asset_rows.append(
            {
                "asset": asset,
                "best_family": best["family"],
                "best_name": best["name"],
                "best_min_score": best["min_score"],
                "best_avg_score": best["avg_score"],
                "best_last_score": best["last_score"],
                "positive_windows": best["positive_windows"],
            }
        )
    return pd.DataFrame(by_asset_rows).sort_values(["best_min_score", "best_avg_score"], ascending=False)


def main():
    prices_df = pd.read_csv("prices.txt", sep=r"\s+")
    prices = prices_df.values.T

    profile, market = asset_profile(prices_df)
    factor = market_factor_summary(prices_df)
    summary, detail = scan_strategies(prices)
    asset_opps = summarise_asset_opportunities(summary)

    profile.to_csv("all_asset_profile.csv", index=False)
    summary.sort_values(["min_score", "avg_score"], ascending=False).to_csv(
        "all_asset_strategy_summary.csv", index=False
    )
    detail.to_csv("all_asset_strategy_detail.csv", index=False)
    asset_opps.to_csv("all_asset_opportunities.csv", index=False)

    print("\nUniverse factor summary")
    for key, value in factor.items():
        print(f"{key}: {value:.4f}")

    print("\nMost autocorrelated assets")
    print(
        profile[
            [
                "asset",
                "buy_hold_sharpe",
                "corr_market",
                "beta_market",
                "residual_vol",
                "ac1",
                "ac2",
                "ac5",
                "ac10",
                "abs_ac_sum",
            ]
        ]
        .head(15)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )

    print("\nTop robust candidates across CV windows")
    display_cols = [
        "name",
        "family",
        "asset",
        "min_score",
        "avg_score",
        "last_score",
        "positive_windows",
        "avg_mean_pl",
        "avg_sharpe",
    ]
    print(
        summary.sort_values(["min_score", "avg_score"], ascending=False)
        .head(30)[display_cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )

    print("\nBest opportunity per asset")
    print(
        asset_opps.head(25).to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )

    print("\nTop cross-sectional / market candidates")
    print(
        summary[summary["family"].isin(["cross_sectional", "market_direction"])]
        .sort_values(["min_score", "avg_score"], ascending=False)
        .head(20)[display_cols]
        .to_string(index=False, float_format=lambda value: f"{value:.2f}")
    )


if __name__ == "__main__":
    main()
