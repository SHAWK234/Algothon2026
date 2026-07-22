import numpy as np
from collections import deque

nInst = 51
currentPos = np.zeros(nInst)

LOOKBACK = 120
IC_LOOKBACK = 120
IC_PRIOR = 0.03
REG_HALF_LIFE = 125.0
REG_DECAY = 2.0 ** (-1.0 / REG_HALF_LIFE)
REG_RIDGE = 1.0
FIRST_FEATURE_NT = LOOKBACK + 3
N_FAST = 3

ALGO_REGIME_LOOKBACK = 120
ALGO_TRADE_LOOKBACK = 60
N_ALGO_SIGNALS = 4

PAIR_ENTRY_Z = 1.0
PAIR_MAX_HOLD = 5

_reg_wxx = np.zeros((N_FAST, N_FAST))
_reg_wxy = np.zeros(N_FAST)
_ic_history = deque(maxlen=IC_LOOKBACK)
_algo_pnl_history = deque(maxlen=ALGO_REGIME_LOOKBACK)
_last_nt = None
_last_features = None
_last_algo_candidates = None
_active_pairs = {}


def _reset_state():
    global _reg_wxx, _reg_wxy, _ic_history, _algo_pnl_history
    global _last_nt, _last_features, _last_algo_candidates, _active_pairs
    _reg_wxx = np.zeros((N_FAST, N_FAST))
    _reg_wxy = np.zeros(N_FAST)
    _ic_history = deque(maxlen=IC_LOOKBACK)
    _algo_pnl_history = deque(maxlen=ALGO_REGIME_LOOKBACK)
    _last_nt = None
    _last_features = None
    _last_algo_candidates = None
    _active_pairs = {}


def _correlation_matrix(left, right):
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    left_std = left.std(axis=0)
    right_std = right.std(axis=0)
    left_std[left_std < 1e-12] = 1.0
    right_std[right_std < 1e-12] = 1.0
    left = left / left_std
    right = right / right_std
    return left.T.dot(right) / max(left.shape[0] - 1, 1)


def _return_betas(log_returns):
    algo = log_returns[0]
    algo_centered = algo - algo.mean()
    algo_var = np.mean(algo_centered * algo_centered)
    if algo_var < 1e-12:
        return np.ones(log_returns.shape[0] - 1)
    constituents = log_returns[1:].T
    constituents_centered = constituents - constituents.mean(axis=0)
    return np.mean(
        algo_centered[:, None] * constituents_centered,
        axis=0,
    ) / algo_var


def _standardised_residuals(log_returns, betas):
    residuals = np.empty_like(log_returns)
    residuals[0] = log_returns[0]
    residuals[1:] = (
        log_returns[1:]
        - betas[:, None] * log_returns[0][None, :]
    )
    residual_std = residuals.std(axis=1)
    residual_std[residual_std < 1e-12] = 1.0
    return (residuals / residual_std[:, None]).T


def _sparse_lead_lag_forecast(standardised_returns):
    leaders = standardised_returns[:-1]
    followers = standardised_returns[1:]
    n_samples = leaders.shape[0]
    full_corr = _correlation_matrix(leaders, followers)
    midpoint = n_samples // 2
    first_corr = _correlation_matrix(
        leaders[:midpoint], followers[:midpoint]
    )
    second_corr = _correlation_matrix(
        leaders[midpoint:], followers[midpoint:]
    )
    stable = (
        (np.sign(first_corr) == np.sign(second_corr))
        & (np.abs(full_corr) >= 2.0 / np.sqrt(n_samples))
    )
    np.fill_diagonal(stable, False)
    return standardised_returns[-1].dot(full_corr * stable)


def _ridge_lead_lag_forecast(standardised_residuals):
    leaders = standardised_residuals[:-1]
    followers = standardised_residuals[1:]
    n_samples = leaders.shape[0]
    gram = leaders.T.dot(leaders)
    coefficients = np.linalg.solve(
        gram + n_samples * np.eye(nInst),
        leaders.T.dot(followers),
    )
    np.fill_diagonal(coefficients, 0.0)
    return standardised_residuals[-1].dot(coefficients)


def _pair_information(prcSoFar):
    log_prices = np.log(prcSoFar[1:])
    log_returns = np.diff(log_prices, axis=1)
    return_corr = np.corrcoef(log_returns)
    np.fill_diagonal(return_corr, np.nan)
    partners = np.nanargmax(np.abs(return_corr), axis=1)
    betas = np.zeros(nInst - 1)
    z_scores = np.zeros(nInst - 1)

    for i, partner in enumerate(partners):
        y = log_prices[i]
        x = log_prices[partner]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        beta = np.mean(x_centered * y_centered) / (
            np.mean(x_centered * x_centered) + 1e-12
        )
        spread = y_centered - beta * x_centered
        recent = spread[-LOOKBACK:]
        spread_std = recent.std()
        betas[i] = beta
        if spread_std > 1e-12:
            z_scores[i] = (
                spread[-1] - recent.mean()
            ) / spread_std

    return partners, betas, z_scores


def _pair_directional_forecast(z_scores):
    forecast = np.zeros(nInst)
    forecast[1:] = -z_scores
    return forecast


def _overvaluation_forecast(prcSoFar):
    normalised_constituents = prcSoFar[1:] / prcSoFar[1:, [0]]
    normalised_algo = prcSoFar[0] / prcSoFar[0, 0]
    relative_log_price = np.log(
        normalised_constituents / normalised_algo[None, :]
    )
    recent = relative_log_price[:, -LOOKBACK:]
    relative_std = recent.std(axis=1)
    relative_std[relative_std < 1e-12] = 1.0
    z_score = (
        relative_log_price[:, -1] - recent.mean(axis=1)
    ) / relative_std
    forecast = np.zeros(nInst)
    forecast[1:] = np.where(z_score > 0.0, -z_score, 0.0)
    return forecast


def _cross_sectional_scale(forecast):
    scale = forecast.std()
    if scale < 1e-12:
        return np.zeros_like(forecast)
    return forecast / scale


def _rolling_log_vol(prices, window=LOOKBACK):
    returns = np.diff(np.log(prices))[-window:]
    return max(returns.std(), 1e-6)


def _algo_candidates(prcSoFar, sparse_residual, sparse_raw):
    algo_prices = prcSoFar[0]
    basket_prices = prcSoFar[1:].mean(axis=0)

    algo_return_5 = algo_prices[-1] / algo_prices[-6] - 1.0
    basket_return_2 = basket_prices[-1] / basket_prices[-3] - 1.0

    algo_vol = _rolling_log_vol(algo_prices)
    basket_vol = _rolling_log_vol(basket_prices)

    return np.array([
        sparse_residual[0] / max(sparse_residual.std(), 1e-12),
        sparse_raw[0] / max(sparse_raw.std(), 1e-12),
        -algo_return_5 / (np.sqrt(5.0) * algo_vol),
        -basket_return_2 / (np.sqrt(2.0) * basket_vol),
    ])


def _build_signals(prcSoFar):
    log_returns = np.diff(np.log(prcSoFar), axis=1)
    betas = _return_betas(log_returns)
    residual_returns = _standardised_residuals(log_returns, betas)

    raw_std = log_returns.std(axis=1)
    raw_std[raw_std < 1e-12] = 1.0
    standardised_raw = (log_returns / raw_std[:, None]).T

    sparse_residual = _sparse_lead_lag_forecast(residual_returns)
    sparse_raw = _sparse_lead_lag_forecast(standardised_raw)
    ridge_residual = _ridge_lead_lag_forecast(residual_returns)
    partners, pair_betas, pair_z = _pair_information(prcSoFar)
    pair_directional = _pair_directional_forecast(pair_z)
    overvaluation = _overvaluation_forecast(prcSoFar)

    features = np.column_stack([
        _cross_sectional_scale(sparse_residual)[1:],
        _cross_sectional_scale(sparse_raw)[1:],
        _cross_sectional_scale(ridge_residual)[1:],
        _cross_sectional_scale(pair_directional)[1:],
        _cross_sectional_scale(overvaluation)[1:],
    ])

    algo_candidates = _algo_candidates(
        prcSoFar,
        sparse_residual,
        sparse_raw,
    )

    return features, algo_candidates, partners, pair_betas, pair_z


def _rankdata(values):
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1
        ranks[order[start:end]] = (
            0.5 * (start + end - 1) + 1.0
        )
        start = end
    return ranks


def _daily_ics(features, realised_returns):
    y = _rankdata(realised_returns)
    y -= y.mean()
    yy = y.dot(y)
    result = np.zeros(5)
    for k in range(5):
        x = _rankdata(features[:, k])
        x -= x.mean()
        denominator = np.sqrt(x.dot(x) * yy)
        if denominator > 1e-12:
            result[k] = x.dot(y) / denominator
    return result


def _update_training(
    features,
    realised_returns,
    algo_candidates,
    realised_algo_return,
):
    global _reg_wxx, _reg_wxy, _ic_history, _algo_pnl_history

    _reg_wxx *= REG_DECAY
    _reg_wxy *= REG_DECAY

    target = 10_000.0 * realised_returns
    weight = np.abs(target)
    fast = features[:, :N_FAST]
    _reg_wxx += fast.T.dot(weight[:, None] * fast)
    _reg_wxy += fast.T.dot(weight * target)
    _ic_history.append(_daily_ics(features, realised_returns))

    algo_target = 100_000.0 * realised_algo_return
    _algo_pnl_history.append(
        np.sign(algo_candidates) * algo_target
    )


def _fit_fast_regression():
    total_weight_scale = max(
        np.trace(_reg_wxx) / N_FAST,
        1e-12,
    )
    return np.linalg.solve(
        _reg_wxx
        + REG_RIDGE
        * total_weight_scale
        * np.eye(N_FAST),
        _reg_wxy,
    )


def _ic_weights():
    if not _ic_history:
        return np.ones(5)
    trailing_ic = np.mean(np.asarray(_ic_history), axis=0)
    raw = np.maximum(trailing_ic + IC_PRIOR, 0.0)
    if raw.mean() < 1e-12:
        return np.ones(5)
    return raw / raw.mean()


def _algo_ensemble_forecast(candidates, lookback):
    if not _algo_pnl_history:
        weights = np.ones(N_ALGO_SIGNALS)
    else:
        history = np.asarray(_algo_pnl_history)
        history = history[-min(lookback, len(history)):]
        mean_pnl = history.mean(axis=0)
        raw = np.maximum(mean_pnl, 0.0)
        if raw.mean() < 1e-12:
            weights = np.ones(N_ALGO_SIGNALS)
        else:
            weights = raw / raw.mean()
    return weights.dot(np.sign(candidates))


def _update_pair_state(partners, betas, z_scores):
    global _active_pairs
    next_state = {}

    for key, trade in _active_pairs.items():
        i, j = key
        current_z = z_scores[i]
        crossed = np.sign(current_z) != np.sign(trade["entry_z"])
        age = trade["age"] + 1
        if not crossed and age < PAIR_MAX_HOLD:
            trade = trade.copy()
            trade["age"] = age
            next_state[key] = trade

    occupied = {index for key in next_state for index in key}

    for i in range(nInst - 1):
        j = int(partners[i])
        if (
            j <= i
            or int(partners[j]) != i
            or i in occupied
            or j in occupied
            or abs(z_scores[i]) < PAIR_ENTRY_Z
        ):
            continue
        next_state[(i, j)] = {
            "entry_z": z_scores[i],
            "direction": -np.sign(z_scores[i]),
            "beta": betas[i],
            "age": 0,
        }
        occupied.add(i)
        occupied.add(j)

    _active_pairs = next_state


def _active_pair_directions():
    directions = np.zeros(nInst - 1)
    for (i, j), trade in _active_pairs.items():
        direction = trade["direction"]
        beta = trade["beta"]
        directions[i] += direction
        directions[j] += (
            -direction
            * np.sign(beta if abs(beta) > 1e-12 else 1.0)
        )
    return np.sign(directions)


def _bootstrap(prcSoFar):
    global _last_nt, _last_features, _last_algo_candidates

    _reset_state()
    nt = prcSoFar.shape[1]

    for prefix_len in range(FIRST_FEATURE_NT, nt):
        (
            features,
            algo_candidates,
            partners,
            pair_betas,
            pair_z,
        ) = _build_signals(prcSoFar[:, :prefix_len])

        realised_returns = (
            prcSoFar[1:, prefix_len]
            / prcSoFar[1:, prefix_len - 1]
            - 1.0
        )
        realised_algo_return = (
            prcSoFar[0, prefix_len]
            / prcSoFar[0, prefix_len - 1]
            - 1.0
        )
        _update_training(
            features,
            realised_returns,
            algo_candidates,
            realised_algo_return,
        )
        _update_pair_state(partners, pair_betas, pair_z)

    (
        features,
        algo_candidates,
        partners,
        pair_betas,
        pair_z,
    ) = _build_signals(prcSoFar)
    _update_pair_state(partners, pair_betas, pair_z)

    _last_nt = nt
    _last_features = features
    _last_algo_candidates = algo_candidates
    return features, algo_candidates


def getMyPosition(prcSoFar):
    global currentPos, _last_nt, _last_features, _last_algo_candidates

    nins, nt = prcSoFar.shape
    if nt < FIRST_FEATURE_NT:
        currentPos = np.zeros(nins, dtype=int)
        return currentPos

    if _last_nt is None or nt != _last_nt + 1:
        features, algo_candidates = _bootstrap(prcSoFar)
    else:
        realised_returns = (
            prcSoFar[1:, -1]
            / prcSoFar[1:, -2]
            - 1.0
        )
        realised_algo_return = (
            prcSoFar[0, -1]
            / prcSoFar[0, -2]
            - 1.0
        )
        _update_training(
            _last_features,
            realised_returns,
            _last_algo_candidates,
            realised_algo_return,
        )

        (
            features,
            algo_candidates,
            partners,
            pair_betas,
            pair_z,
        ) = _build_signals(prcSoFar)
        _update_pair_state(partners, pair_betas, pair_z)

        _last_nt = nt
        _last_features = features
        _last_algo_candidates = algo_candidates

    ic_forecast = features.dot(_ic_weights())
    regression_forecast = features[:, :N_FAST].dot(
        _fit_fast_regression()
    )
    ic_forecast = _cross_sectional_scale(ic_forecast)
    regression_forecast = _cross_sectional_scale(
        regression_forecast
    )

    # The longer ALGO ensemble is a market-regime classifier.
    # In predicted-down regimes the local constituent P&L regression is
    # disabled, leaving the more robust IC forecast in control.
    algo_regime_forecast = _algo_ensemble_forecast(
        algo_candidates,
        ALGO_REGIME_LOOKBACK,
    )
    regression_weight = (
        1.0 if algo_regime_forecast >= 0.0 else 0.0
    )
    combined = ic_forecast + regression_weight * regression_forecast

    pair_direction = _active_pair_directions()
    weak = np.abs(combined) < np.median(np.abs(combined))
    conflict = (
        weak
        & (pair_direction != 0.0)
        & (np.sign(combined) != pair_direction)
    )
    combined[conflict] = 0.0

    # Keep the original sparse-residual ALGO forecast. The longer ensemble
    # is used only as a regime classifier for the constituent regression.
    algo_trade_forecast = algo_candidates[0]

    target_dollars = np.zeros(nins)
    target_dollars[0] = (
        100_000.0 * np.sign(algo_trade_forecast)
    )
    target_dollars[1:] = 10_000.0 * np.sign(combined)

    currentPos = np.trunc(
        target_dollars / prcSoFar[:, -1]
    ).astype(int)
    return currentPos
