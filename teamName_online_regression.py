import numpy as np

nInst = 51
currentPos = np.zeros(nInst)

LOOKBACK = 120
FIRST_FEATURE_NT = LOOKBACK + 3
N_FEATURES = 5

# Expanding weighted-ridge sufficient statistics.
_sum_weight = 0.0
_sum_wx = np.zeros(N_FEATURES)
_sum_wy = 0.0
_sum_wxx = np.zeros((N_FEATURES, N_FEATURES))
_sum_wxy = np.zeros(N_FEATURES)

_last_nt = None
_last_regular_features = None


def _reset_model():
    global _sum_weight, _sum_wx, _sum_wy, _sum_wxx, _sum_wxy
    global _last_nt, _last_regular_features

    _sum_weight = 0.0
    _sum_wx = np.zeros(N_FEATURES)
    _sum_wy = 0.0
    _sum_wxx = np.zeros((N_FEATURES, N_FEATURES))
    _sum_wxy = np.zeros(N_FEATURES)

    _last_nt = None
    _last_regular_features = None


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
        leaders[:midpoint],
        followers[:midpoint],
    )
    second_corr = _correlation_matrix(
        leaders[midpoint:],
        followers[midpoint:],
    )

    stable = (
        (np.sign(first_corr) == np.sign(second_corr))
        & (np.abs(full_corr) >= 2.0 / np.sqrt(n_samples))
    )

    np.fill_diagonal(stable, False)
    coefficients = full_corr * stable

    return standardised_returns[-1].dot(coefficients)


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


def _nearest_pair_forecast(prcSoFar):
    log_prices = np.log(prcSoFar[1:])
    log_returns = np.diff(log_prices, axis=1)

    return_corr = np.corrcoef(log_returns)
    np.fill_diagonal(return_corr, np.nan)

    forecast = np.zeros(nInst)

    for i in range(nInst - 1):
        partner = int(np.nanargmax(np.abs(return_corr[i])))

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

        if spread_std > 1e-12:
            forecast[i + 1] = -(
                spread[-1] - recent.mean()
            ) / spread_std

    return forecast


def _overvaluation_forecast(prcSoFar):
    normalised_constituents = (
        prcSoFar[1:] / prcSoFar[1:, [0]]
    )
    normalised_algo = (
        prcSoFar[0] / prcSoFar[0, 0]
    )

    relative_log_price = np.log(
        normalised_constituents
        / normalised_algo[None, :]
    )

    recent = relative_log_price[:, -LOOKBACK:]
    relative_std = recent.std(axis=1)
    relative_std[relative_std < 1e-12] = 1.0

    z_score = (
        relative_log_price[:, -1]
        - recent.mean(axis=1)
    ) / relative_std

    forecast = np.zeros(nInst)
    forecast[1:] = np.where(
        z_score > 0.0,
        -z_score,
        0.0,
    )

    return forecast


def _cross_sectional_scale(forecast):
    scale = forecast.std()

    if scale < 1e-12:
        return np.zeros_like(forecast)

    return forecast / scale


def _build_features(prcSoFar):
    log_returns = np.diff(
        np.log(prcSoFar),
        axis=1,
    )

    betas = _return_betas(log_returns)

    residual_returns = _standardised_residuals(
        log_returns,
        betas,
    )

    raw_std = log_returns.std(axis=1)
    raw_std[raw_std < 1e-12] = 1.0

    standardised_raw_returns = (
        log_returns / raw_std[:, None]
    ).T

    sparse_residual = _sparse_lead_lag_forecast(
        residual_returns
    )
    sparse_raw = _sparse_lead_lag_forecast(
        standardised_raw_returns
    )
    ridge_residual = _ridge_lead_lag_forecast(
        residual_returns
    )
    pair_forecast = _nearest_pair_forecast(
        prcSoFar
    )
    overvaluation = _overvaluation_forecast(
        prcSoFar
    )

    regular_features = np.column_stack([
        _cross_sectional_scale(sparse_residual)[1:],
        _cross_sectional_scale(sparse_raw)[1:],
        _cross_sectional_scale(ridge_residual)[1:],
        _cross_sectional_scale(pair_forecast)[1:],
        _cross_sectional_scale(overvaluation)[1:],
    ])

    return regular_features, sparse_residual[0]


def _add_training_day(features, realised_returns):
    """
    Add the 50 regular-instrument rows from one newly labelled day.

    Target:
        y = 10,000 * next-day return

    Weight:
        w = abs(y)

    This matches:
        sum w * (y - intercept - X beta)^2
        + sum(w) * ||beta||^2
    """
    global _sum_weight, _sum_wx, _sum_wy
    global _sum_wxx, _sum_wxy

    target = 10_000.0 * realised_returns
    weight = np.abs(target)

    _sum_weight += weight.sum()
    _sum_wx += (weight[:, None] * features).sum(axis=0)
    _sum_wy += weight.dot(target)

    _sum_wxx += features.T.dot(
        weight[:, None] * features
    )
    _sum_wxy += features.T.dot(
        weight * target
    )


def _fit_weighted_ridge():
    if _sum_weight < 1e-12:
        return 0.0, np.zeros(N_FEATURES)

    mean_x = _sum_wx / _sum_weight
    mean_y = _sum_wy / _sum_weight

    centered_xx = (
        _sum_wxx
        - _sum_weight * np.outer(mean_x, mean_x)
    )
    centered_xy = (
        _sum_wxy
        - _sum_weight * mean_x * mean_y
    )

    # Dividing the whole objective by sum(weight) gives a fixed
    # unit-strength ridge penalty, independent of sample count.
    coefficients = np.linalg.solve(
        centered_xx
        + _sum_weight * np.eye(N_FEATURES),
        centered_xy,
    )

    intercept = (
        mean_y - mean_x.dot(coefficients)
    )

    return intercept, coefficients


def _bootstrap_model(prcSoFar):
    """
    Build all labelled feature/target rows available at the first call.

    A feature built with prefix length p predicts the return from
    price index p-1 to price index p.
    """
    global _last_nt, _last_regular_features

    _reset_model()
    _, nt = prcSoFar.shape

    for prefix_len in range(
        FIRST_FEATURE_NT,
        nt,
    ):
        features, _ = _build_features(
            prcSoFar[:, :prefix_len]
        )

        realised_returns = (
            prcSoFar[1:, prefix_len]
            / prcSoFar[1:, prefix_len - 1]
            - 1.0
        )

        _add_training_day(
            features,
            realised_returns,
        )

    current_features, algo_forecast = (
        _build_features(prcSoFar)
    )

    _last_nt = nt
    _last_regular_features = current_features

    return current_features, algo_forecast


def getMyPosition(prcSoFar):
    global currentPos
    global _last_nt, _last_regular_features

    nins, nt = prcSoFar.shape

    if nt < FIRST_FEATURE_NT:
        currentPos = np.zeros(nins, dtype=int)
        return currentPos

    # Normal evaluator use is sequential. Rebuild safely if the process
    # starts at a later day, history shrinks, or calls are non-sequential.
    if (
        _last_nt is None
        or nt != _last_nt + 1
    ):
        regular_features, algo_forecast = (
            _bootstrap_model(prcSoFar)
        )
    else:
        # The previous call's prediction features are now labelled by
        # the newly observed final return.
        realised_returns = (
            prcSoFar[1:, -1]
            / prcSoFar[1:, -2]
            - 1.0
        )

        _add_training_day(
            _last_regular_features,
            realised_returns,
        )

        regular_features, algo_forecast = (
            _build_features(prcSoFar)
        )

        _last_nt = nt
        _last_regular_features = regular_features

    intercept, coefficients = (
        _fit_weighted_ridge()
    )

    predicted_pnl = (
        intercept
        + regular_features.dot(coefficients)
    )

    target_dollars = np.zeros(nins)
    target_dollars[0] = (
        100_000.0 * np.sign(algo_forecast)
    )
    target_dollars[1:] = (
        10_000.0 * np.sign(predicted_pnl)
    )

    currentPos = np.trunc(
        target_dollars / prcSoFar[:, -1]
    ).astype(int)

    return currentPos
