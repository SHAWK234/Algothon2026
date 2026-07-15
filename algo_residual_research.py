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

ALGO_IDX = 0
TRAIN_DAYS = 400
DEFAULT_NUM_TEST_DAYS = 100

POSITION_LIMITS = np.full(len(ASSETS), 10_000.0)
POSITION_LIMITS[ALGO_IDX] = 100_000.0

COMM_RATES = np.full(len(ASSETS), 0.0001)
COMM_RATES[ALGO_IDX] = 0.00002


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250) * mean_pl / std_pl
    return mean_pl * sharpe**2 / (sharpe**2 + 1.0)


def annualised_sharpe(pl):
    pl = np.asarray(pl, dtype=float)
    std = np.std(pl)
    if std <= 0:
        return 0.0
    return np.sqrt(250) * np.mean(pl) / std


def make_basket_indices(prices, mode, top_n):
    if mode == "all":
        return [idx for idx in range(len(ASSETS)) if idx != ALGO_IDX]

    returns = prices[:, 1:TRAIN_DAYS] / prices[:, : TRAIN_DAYS - 1] - 1.0
    algo_returns = returns[ALGO_IDX]
    rows = []
    for idx in range(len(ASSETS)):
        if idx == ALGO_IDX:
            continue
        corr = np.corrcoef(algo_returns, returns[idx])[0, 1]
        if np.isfinite(corr):
            rows.append((abs(corr), idx))
    rows.sort(reverse=True)
    return [idx for _, idx in rows[:top_n]]


def residual_z_score(prices_so_far, basket_indices, window, lag_signal):
    signal_prices = prices_so_far[:, :-1] if lag_signal else prices_so_far
    if signal_prices.shape[1] < window + 2:
        return np.nan

    normalised_basket = (
        signal_prices[basket_indices] / signal_prices[basket_indices][:, [0]]
    ).mean(axis=0)
    basket_price = normalised_basket * signal_prices[ALGO_IDX, 0]
    spread = np.log(signal_prices[ALGO_IDX]) - np.log(basket_price)
    recent = spread[-window:]
    spread_std = np.std(recent, ddof=1)
    if not np.isfinite(spread_std) or spread_std <= 0:
        return np.nan
    return (spread[-1] - np.mean(recent)) / spread_std


def threshold_state(z_score, previous_state, entry_z, exit_z):
    if not np.isfinite(z_score):
        return previous_state

    if previous_state == 0:
        if z_score >= entry_z:
            return -1
        if z_score <= -entry_z:
            return 1
        return 0

    if previous_state == 1:
        if z_score >= entry_z:
            return -1
        if z_score >= -exit_z:
            return 0
        return 1

    if z_score <= -entry_z:
        return 1
    if z_score <= exit_z:
        return 0
    return -1


def algo_residual_position(
    prices_so_far,
    basket_indices,
    window,
    entry_z,
    exit_z,
    hedge_ratio,
    lag_signal,
    previous_state,
):
    current_prices = prices_so_far[:, -1]
    z_score = residual_z_score(prices_so_far, basket_indices, window, lag_signal)
    state = threshold_state(z_score, previous_state, entry_z, exit_z)

    dollar_targets = np.zeros(len(ASSETS))
    algo_dollars = state * POSITION_LIMITS[ALGO_IDX]
    dollar_targets[ALGO_IDX] = algo_dollars

    hedge_dollars = -hedge_ratio * algo_dollars
    if state != 0 and hedge_ratio > 0:
        per_asset_hedge = hedge_dollars / len(basket_indices)
        for idx in basket_indices:
            dollar_targets[idx] = np.clip(
                per_asset_hedge,
                -POSITION_LIMITS[idx],
                POSITION_LIMITS[idx],
            )

    positions = np.divide(
        dollar_targets,
        current_prices,
        out=np.zeros_like(dollar_targets),
        where=current_prices > 0,
    )
    return positions.astype(int), state


def evaluate_strategy(
    prices,
    num_test_days,
    basket_mode,
    top_n,
    window,
    entry_z,
    exit_z,
    hedge_ratio,
    lag_signal,
):
    basket_indices = make_basket_indices(prices, basket_mode, top_n)

    cash = 0.0
    current_position = np.zeros(len(ASSETS))
    total_dvolume = 0.0
    value = 0.0
    commission = 0.0
    daily_pl = []
    state = 0

    _, n_days = prices.shape
    start_day = n_days - num_test_days

    for t in range(start_day, n_days + 1):
        prices_so_far = prices[:, :t]
        current_prices = prices_so_far[:, -1]

        if t < n_days:
            new_position, state = algo_residual_position(
                prices_so_far,
                basket_indices,
                window,
                entry_z,
                exit_z,
                hedge_ratio,
                lag_signal,
                state,
            )
            position_limits = (POSITION_LIMITS / current_prices).astype(int)
            new_position = np.clip(new_position, -position_limits, position_limits)
        else:
            new_position = np.array(current_position)

        delta_position = new_position - current_position
        cash -= current_prices.dot(delta_position) + commission

        dvolumes = current_prices * np.abs(delta_position)
        total_dvolume += np.sum(dvolumes)
        commission = np.sum(dvolumes * COMM_RATES)

        current_position = np.array(new_position)
        today_pl = cash + current_position.dot(current_prices) - value
        value = cash + current_position.dot(current_prices)

        if t > start_day:
            daily_pl.append(today_pl)

    daily_pl = np.array(daily_pl)
    mean_pl = float(np.mean(daily_pl))
    std_pl = float(np.std(daily_pl))
    return {
        "basket_mode": basket_mode,
        "top_n": top_n,
        "window": window,
        "entry_z": entry_z,
        "exit_z": exit_z,
        "hedge_ratio": hedge_ratio,
        "lag_signal": lag_signal,
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": annualised_sharpe(daily_pl),
        "score": score(mean_pl, std_pl),
        "dvolume": total_dvolume,
        "value": value,
    }


def run_grid():
    prices_df = pd.read_csv("prices.txt", sep=r"\s+")
    prices = prices_df.values.T

    rows = []
    grid = itertools.product(
        ["all", "top"],
        [10, 20, 30, 50],
        [10, 20, 40, 60, 100],
        [0.5, 1.0, 1.5],
        [0.0, 0.5],
        [0.0, 0.5, 1.0],
        [True, False],
    )

    for basket_mode, top_n, window, entry_z, exit_z, hedge_ratio, lag_signal in grid:
        if basket_mode == "all" and top_n != 50:
            continue
        if exit_z > entry_z:
            continue
        rows.append(
            evaluate_strategy(
                prices,
                DEFAULT_NUM_TEST_DAYS,
                basket_mode,
                top_n,
                window,
                entry_z,
                exit_z,
                hedge_ratio,
                lag_signal,
            )
        )

    results = pd.DataFrame(rows).sort_values("score", ascending=False)
    print("\nTop ALGO residual strategies on days 400-500")
    print(results.head(25).to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    robust = results[
        (results["entry_z"] >= 1.0)
        & (results["lag_signal"])
        & (results["hedge_ratio"] > 0)
    ].sort_values("score", ascending=False)
    print("\nTop conservative candidates: lagged signal, hedged, entry >= 1")
    print(robust.head(15).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run_grid()
