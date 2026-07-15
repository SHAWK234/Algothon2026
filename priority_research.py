import itertools

import numpy as np
import pandas as pd


ASSET_NAMES = [
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT", "HUXZ",
    "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET", "ANSO", "DIHO",
    "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF", "RRES", "CTGI", "ALUT",
    "ACAC", "SRTX", "GARI", "RCRI", "ACIX", "CCNS", "MTNS", "IHOZ", "NAYO",
    "FWWG", "EELT", "HRND", "AETS", "ULXY", "BLBT", "BENI", "ITPA", "HTRK",
    "NGTE", "ILVX", "FCSG", "FARS", "MHRM", "EAFC",
]

N_INST = len(ASSET_NAMES)
POSITION_LIMITS = np.full(N_INST, 10_000.0)
POSITION_LIMITS[0] = 100_000.0
COMM_RATES = np.full(N_INST, 0.0001)
COMM_RATES[0] = 0.00002

BASE_PARAMS = {
    "lookback": 2,
    "short_threshold": 0.0015,
    "long_threshold": 0.0075,
    "excluded": (1, 5, 17, 18, 34, 36),
    "min_scale": 0.25,
    "max_scale": 1.50,
    "scale_speed": 800.0,
}

CV_WINDOWS = [
    ("250-350", 250, 350),
    ("300-400", 300, 400),
    ("350-450", 350, 450),
    ("400-500", 400, 500),
]


def load_prices():
    prices_df = pd.read_csv("prices.txt", sep=r"\s+")
    return prices_df.to_numpy().T


def score(mean_pl, std_pl):
    if mean_pl <= 0 or std_pl < 1e-10:
        return mean_pl
    sharpe = np.sqrt(250.0) * mean_pl / std_pl
    return mean_pl * sharpe**2 / (sharpe**2 + 1.0)


def summarize_pl(pl, value=0.0, dvolume=0.0):
    pl = np.asarray(pl, dtype=float)
    mean_pl = float(np.mean(pl))
    std_pl = float(np.std(pl))
    sharpe = 0.0 if std_pl <= 0 else float(np.sqrt(250.0) * mean_pl / std_pl)
    return {
        "mean_pl": mean_pl,
        "std_pl": std_pl,
        "sharpe": sharpe,
        "score": float(score(mean_pl, std_pl)),
        "value": float(value),
        "dvolume": float(dvolume),
    }


def target_position_from_dollars(dollar_targets, current_prices):
    return np.divide(
        dollar_targets,
        current_prices,
        out=np.zeros_like(dollar_targets),
        where=current_prices > 0,
    ).astype(int)


def signal_from_market_move(market_move, short_threshold, long_threshold):
    if market_move > short_threshold:
        return -1.0, market_move - short_threshold
    if market_move < -long_threshold:
        return 1.0, abs(market_move) - long_threshold
    return 0.0, 0.0


def current_strategy_position(hist, params, mode="both", included=None):
    n_inst, nt = hist.shape
    lookback = params["lookback"]
    if nt <= lookback:
        return np.zeros(n_inst)

    current_prices = hist[:, -1]
    old_prices = hist[:, -1 - lookback]
    asset_moves = np.divide(
        current_prices,
        old_prices,
        out=np.ones_like(current_prices),
        where=old_prices > 0,
    ) - 1.0
    market_move = float(np.mean(asset_moves))
    direction, signal_excess = signal_from_market_move(
        market_move,
        params["short_threshold"],
        params["long_threshold"],
    )
    if direction == 0.0:
        return np.zeros(n_inst)
    if mode == "long_only" and direction < 0:
        return np.zeros(n_inst)
    if mode == "short_only" and direction > 0:
        return np.zeros(n_inst)

    scale = params["min_scale"] + (
        params["max_scale"] - params["min_scale"]
    ) * np.tanh(params["scale_speed"] * max(0.0, signal_excess))

    dollar_limits = POSITION_LIMITS.copy()
    dollar_limits[list(params.get("excluded", ()))] = 0.0
    if included is not None:
        mask = np.zeros(n_inst, dtype=bool)
        mask[list(included)] = True
        dollar_limits[~mask] = 0.0

    dollar_targets = direction * scale * dollar_limits
    return target_position_from_dollars(dollar_targets, current_prices)


def evaluate_interval(prices, start_day, end_day, position_fn, collect_asset_pl=False):
    cash = 0.0
    cur_pos = np.zeros(N_INST)
    total_dvolume = 0.0
    value = 0.0
    commission = 0.0
    daily_pl = []
    asset_pl = []

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
        total_dvolume += float(np.sum(dvolumes))
        commission_by_asset = dvolumes * COMM_RATES
        commission = float(np.sum(commission_by_asset))

        prev_pos = np.array(cur_pos)
        prev_value = value
        cur_pos = np.array(new_pos)
        pos_value = float(cur_pos.dot(cur_prices))
        value = cash + pos_value
        today_pl = value - prev_value

        if t > start_day:
            daily_pl.append(today_pl)
            if collect_asset_pl:
                # Attribute daily PnL by instrument. This is approximate but
                # matches the same fills, marks, and per-asset commissions.
                price_change = cur_prices - prev_prices
                asset_pl.append(prev_pos * price_change - prev_commission_by_asset)

        prev_prices = cur_prices
        prev_commission_by_asset = commission_by_asset

    result = summarize_pl(daily_pl, value=value, dvolume=total_dvolume)
    if collect_asset_pl:
        result["asset_pl"] = np.asarray(asset_pl)
    return result


def evaluate_cv(prices, position_fn, windows=CV_WINDOWS):
    rows = []
    for name, start, end in windows:
        result = evaluate_interval(prices, start, end, position_fn)
        rows.append({"window": name, **result})
    df = pd.DataFrame(rows)
    return {
        "cv_mean_score": float(df["score"].mean()),
        "cv_min_score": float(df["score"].min()),
        "cv_mean_pl": float(df["mean_pl"].mean()),
        "cv_mean_sharpe": float(df["sharpe"].mean()),
        "windows": df,
    }


def format_result(result):
    return (
        f"score={result['score']:.2f} mean={result['mean_pl']:.2f} "
        f"std={result['std_pl']:.2f} sharpe={result['sharpe']:.2f}"
    )


def base_position_fn(params=None, mode="both", included=None):
    params = dict(BASE_PARAMS if params is None else params)
    return lambda hist: current_strategy_position(hist, params, mode=mode, included=included)


def run_asset_exclusion_research(prices):
    rows = []
    base_params = dict(BASE_PARAMS)
    base_pos = base_position_fn(base_params)
    base_full = evaluate_interval(prices, 250, 500, base_pos, collect_asset_pl=True)
    base_cv = evaluate_cv(prices, base_pos)

    asset_pl = base_full["asset_pl"]
    for idx, name in enumerate(ASSET_NAMES):
        pl = asset_pl[:, idx]
        rows.append({
            "experiment": "base_attribution",
            "asset_idx": idx,
            "asset": name,
            **summarize_pl(pl),
        })

    current_excluded = set(base_params["excluded"])
    for idx, name in enumerate(ASSET_NAMES):
        params = dict(base_params)
        if idx in current_excluded:
            params["excluded"] = tuple(sorted(current_excluded - {idx}))
            action = "add_back"
        else:
            params["excluded"] = tuple(sorted(current_excluded | {idx}))
            action = "remove"
        pos = base_position_fn(params)
        full = evaluate_interval(prices, 250, 500, pos)
        cv = evaluate_cv(prices, pos)
        rows.append({
            "experiment": action,
            "asset_idx": idx,
            "asset": name,
            "excluded": params["excluded"],
            "full_score": full["score"],
            "full_mean": full["mean_pl"],
            "full_std": full["std_pl"],
            "full_sharpe": full["sharpe"],
            **{k: v for k, v in cv.items() if k != "windows"},
        })

    return base_full, base_cv, pd.DataFrame(rows)


def run_direction_research(prices):
    rows = []
    for mode in ["both", "long_only", "short_only"]:
        pos = base_position_fn(mode=mode)
        full = evaluate_interval(prices, 250, 500, pos)
        cv = evaluate_cv(prices, pos)
        rows.append({
            "mode": mode,
            "full_score": full["score"],
            "full_mean": full["mean_pl"],
            "full_std": full["std_pl"],
            "full_sharpe": full["sharpe"],
            **{k: v for k, v in cv.items() if k != "windows"},
        })
    return pd.DataFrame(rows)


def run_lookback_research(prices):
    threshold_rows = []
    lookbacks = [1, 2, 3, 4, 5, 7, 10]
    short_grid = [0.0005, 0.0010, 0.0015, 0.0025, 0.0040, 0.0060, 0.0080]
    long_grid = [0.0015, 0.0030, 0.0050, 0.0075, 0.0100, 0.0125, 0.0150]

    for lookback in lookbacks:
        params = dict(BASE_PARAMS)
        params["lookback"] = lookback
        same = evaluate_interval(prices, 400, 500, base_position_fn(params))
        same_full = evaluate_interval(prices, 250, 500, base_position_fn(params))
        threshold_rows.append({
            "kind": "same_thresholds",
            "lookback": lookback,
            "short_threshold": params["short_threshold"],
            "long_threshold": params["long_threshold"],
            "train_score": np.nan,
            "test_score": same["score"],
            "test_mean": same["mean_pl"],
            "test_std": same["std_pl"],
            "test_sharpe": same["sharpe"],
            "full_score": same_full["score"],
        })

        best = None
        for short_threshold, long_threshold in itertools.product(short_grid, long_grid):
            params = dict(BASE_PARAMS)
            params["lookback"] = lookback
            params["short_threshold"] = short_threshold
            params["long_threshold"] = long_threshold
            train = evaluate_interval(prices, 250, 400, base_position_fn(params))
            item = (train["score"], short_threshold, long_threshold, train)
            if best is None or item[0] > best[0]:
                best = item
        _, short_threshold, long_threshold, train = best
        params = dict(BASE_PARAMS)
        params["lookback"] = lookback
        params["short_threshold"] = short_threshold
        params["long_threshold"] = long_threshold
        test = evaluate_interval(prices, 400, 500, base_position_fn(params))
        full = evaluate_interval(prices, 250, 500, base_position_fn(params))
        threshold_rows.append({
            "kind": "train250_400_best_thresholds",
            "lookback": lookback,
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "train_score": train["score"],
            "test_score": test["score"],
            "test_mean": test["mean_pl"],
            "test_std": test["std_pl"],
            "test_sharpe": test["sharpe"],
            "full_score": full["score"],
        })

    return pd.DataFrame(threshold_rows)


def historical_signal_pl(prices, params, end_t, window):
    start_t = max(params["lookback"] + 1, end_t - window)
    pl = []
    for t in range(start_t, end_t):
        hist = prices[:, :t]
        pos = current_strategy_position(hist, params)
        cur_prices = prices[:, t - 1]
        next_prices = prices[:, t]
        pos_limits = (POSITION_LIMITS / cur_prices).astype(int)
        pos = np.clip(pos, -pos_limits, pos_limits)
        dvolumes = cur_prices * np.abs(pos)
        comm = float(np.sum(dvolumes * COMM_RATES))
        pl.append(float(pos.dot(next_prices - cur_prices) - comm))
    return np.asarray(pl)


def performance_filter_position(prices, params, window, min_recent_mean):
    def position_fn(hist):
        end_t = hist.shape[1]
        recent_pl = historical_signal_pl(prices, params, end_t, window)
        if len(recent_pl) < max(10, window // 2):
            return current_strategy_position(hist, params)
        if float(np.mean(recent_pl)) < min_recent_mean:
            return np.zeros(N_INST)
        return current_strategy_position(hist, params)

    return position_fn


def run_regime_filter_research(prices):
    rows = []
    params = dict(BASE_PARAMS)
    for window, min_recent_mean in itertools.product(
        [20, 40, 60, 100, 150],
        [-250.0, 0.0, 250.0, 500.0, 750.0],
    ):
        train_pos = performance_filter_position(prices, params, window, min_recent_mean)
        train = evaluate_interval(prices, 250, 400, train_pos)
        test = evaluate_interval(prices, 400, 500, train_pos)
        full = evaluate_interval(prices, 250, 500, train_pos)
        rows.append({
            "window": window,
            "min_recent_mean": min_recent_mean,
            "train_score": train["score"],
            "test_score": test["score"],
            "test_mean": test["mean_pl"],
            "test_std": test["std_pl"],
            "test_sharpe": test["sharpe"],
            "full_score": full["score"],
        })
    return pd.DataFrame(rows)


def hold_decay_position(params, hold_days, decay):
    def position_fn(hist):
        best_pos = np.zeros(N_INST)
        n_inst, nt = hist.shape
        for age in range(0, hold_days):
            cut = nt - age
            if cut <= params["lookback"]:
                continue
            pos = current_strategy_position(hist[:, :cut], params)
            if np.any(pos):
                if age == 0:
                    return pos
                current_prices = hist[:, -1]
                cut_prices = hist[:, cut - 1]
                dollar_targets = pos * cut_prices * (decay ** age)
                return target_position_from_dollars(dollar_targets, current_prices)
        return best_pos

    return position_fn


def run_hold_decay_research(prices):
    rows = []
    params = dict(BASE_PARAMS)
    for hold_days, decay in itertools.product([1, 2, 3, 5, 8], [0.25, 0.50, 0.75, 1.00]):
        pos = hold_decay_position(params, hold_days, decay)
        train = evaluate_interval(prices, 250, 400, pos)
        test = evaluate_interval(prices, 400, 500, pos)
        full = evaluate_interval(prices, 250, 500, pos)
        rows.append({
            "hold_days": hold_days,
            "decay": decay,
            "train_score": train["score"],
            "test_score": test["score"],
            "test_mean": test["mean_pl"],
            "test_std": test["std_pl"],
            "test_sharpe": test["sharpe"],
            "full_score": full["score"],
        })
    return pd.DataFrame(rows)


def dispersion_filtered_position(params, max_dispersion=None, min_coherence=None):
    def position_fn(hist):
        lookback = params["lookback"]
        if hist.shape[1] <= lookback:
            return np.zeros(N_INST)
        current_prices = hist[:, -1]
        old_prices = hist[:, -1 - lookback]
        moves = np.divide(
            current_prices,
            old_prices,
            out=np.ones_like(current_prices),
            where=old_prices > 0,
        ) - 1.0
        dispersion = float(np.std(moves))
        coherence = abs(float(np.mean(moves))) / dispersion if dispersion > 0 else 0.0
        if max_dispersion is not None and dispersion > max_dispersion:
            return np.zeros(N_INST)
        if min_coherence is not None and coherence < min_coherence:
            return np.zeros(N_INST)
        return current_strategy_position(hist, params)

    return position_fn


def run_dispersion_research(prices):
    params = dict(BASE_PARAMS)
    lookback = params["lookback"]
    train_moves = prices[:, lookback:400] / prices[:, : 400 - lookback] - 1.0
    train_dispersions = np.std(train_moves, axis=0)
    train_coherence = np.abs(np.mean(train_moves, axis=0)) / np.where(
        train_dispersions > 0,
        train_dispersions,
        np.nan,
    )
    dispersion_thresholds = np.nanquantile(train_dispersions, [0.50, 0.60, 0.70, 0.80, 0.90])
    coherence_thresholds = np.nanquantile(train_coherence, [0.10, 0.20, 0.30, 0.40, 0.50])
    rows = []

    for threshold in dispersion_thresholds:
        pos = dispersion_filtered_position(params, max_dispersion=float(threshold))
        train = evaluate_interval(prices, 250, 400, pos)
        test = evaluate_interval(prices, 400, 500, pos)
        full = evaluate_interval(prices, 250, 500, pos)
        rows.append({
            "kind": "max_dispersion",
            "threshold": float(threshold),
            "train_score": train["score"],
            "test_score": test["score"],
            "test_mean": test["mean_pl"],
            "test_std": test["std_pl"],
            "test_sharpe": test["sharpe"],
            "full_score": full["score"],
        })

    for threshold in coherence_thresholds:
        pos = dispersion_filtered_position(params, min_coherence=float(threshold))
        train = evaluate_interval(prices, 250, 400, pos)
        test = evaluate_interval(prices, 400, 500, pos)
        full = evaluate_interval(prices, 250, 500, pos)
        rows.append({
            "kind": "min_coherence",
            "threshold": float(threshold),
            "train_score": train["score"],
            "test_score": test["score"],
            "test_mean": test["mean_pl"],
            "test_std": test["std_pl"],
            "test_sharpe": test["sharpe"],
            "full_score": full["score"],
        })

    return pd.DataFrame(rows)


def save_and_print(title, df, path, sort_col, n=10):
    df.to_csv(path, index=False)
    print(f"\n{title}")
    print(f"saved: {path}")
    cols = [c for c in df.columns if c != "excluded"]
    print(df.sort_values(sort_col, ascending=False)[cols].head(n).to_string(index=False))


def main():
    prices = load_prices()
    print(f"Loaded prices: {prices.shape[0]} assets x {prices.shape[1]} days")

    base_full, base_cv, exclusion_df = run_asset_exclusion_research(prices)
    print(f"\nBASE 250-500: {format_result(base_full)}")
    print(
        "BASE CV: "
        f"mean_score={base_cv['cv_mean_score']:.2f} "
        f"min_score={base_cv['cv_min_score']:.2f} "
        f"mean_sharpe={base_cv['cv_mean_sharpe']:.2f}"
    )
    exclusion_df.to_csv("priority_asset_exclusion_research.csv", index=False)
    print("\nWorst current asset attribution, 250-500:")
    print(
        exclusion_df[exclusion_df["experiment"] == "base_attribution"]
        .sort_values("mean_pl")
        .head(12)[["asset_idx", "asset", "mean_pl", "std_pl", "sharpe", "score"]]
        .to_string(index=False)
    )
    print("\nBest one-asset exclusion/add-back changes by CV mean score:")
    change_rows = exclusion_df[exclusion_df["experiment"].isin(["remove", "add_back"])]
    print(
        change_rows.sort_values("cv_mean_score", ascending=False)
        .head(12)[
            [
                "experiment", "asset_idx", "asset", "full_score", "full_mean",
                "full_std", "full_sharpe", "cv_mean_score", "cv_min_score",
            ]
        ]
        .to_string(index=False)
    )

    direction_df = run_direction_research(prices)
    save_and_print("Long/short attribution", direction_df, "priority_direction_research.csv", "full_score")

    lookback_df = run_lookback_research(prices)
    save_and_print("Lookback and threshold sweep", lookback_df, "priority_lookback_research.csv", "test_score")

    regime_df = run_regime_filter_research(prices)
    save_and_print("Recent-performance filter sweep", regime_df, "priority_regime_filter_research.csv", "test_score")

    hold_df = run_hold_decay_research(prices)
    save_and_print("Hold/decay sweep", hold_df, "priority_hold_decay_research.csv", "test_score")

    dispersion_df = run_dispersion_research(prices)
    save_and_print("Dispersion/coherence filter sweep", dispersion_df, "priority_dispersion_research.csv", "test_score")


if __name__ == "__main__":
    main()
