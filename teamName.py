import numpy as np


MARKET_LOOKBACK = 2
SHORT_AFTER_RALLY_THRESHOLD = 0.0015
LONG_AFTER_DROP_THRESHOLD = 0.0075
MAX_ABS_MARKET_MOVE = 0.0175
EXCLUDED_ASSET_INDICES = np.array([], dtype=int)
MIN_POSITION_SCALE = 0.25
MAX_POSITION_SCALE = 1.50
SIGNAL_SCALE_SPEED = 800.0
SHORT_SIDE_MULTIPLIER = 1.0

DEFAULT_DOLLAR_LIMIT = 10_000.0
INST0_DOLLAR_LIMIT = 100_000.0
MARKET_CORE_WEIGHT = 0.75
STABLE_OVERLAY_WEIGHT = 0.25


def _dollar_limits(nins):
    dollar_limits = np.full(nins, DEFAULT_DOLLAR_LIMIT)
    dollar_limits[0] = INST0_DOLLAR_LIMIT
    valid_exclusions = EXCLUDED_ASSET_INDICES[EXCLUDED_ASSET_INDICES < nins]
    dollar_limits[valid_exclusions] = 0.0
    return dollar_limits


def _set_target(dollars, idx, signal, threshold, dollar_limits):
    if idx >= len(dollars) or not np.isfinite(signal) or abs(signal) < threshold:
        return
    dollars[idx] = np.sign(signal) * dollar_limits[idx]


def _absolute_return(prcSoFar, idx, lookback, lag_signal):
    end = prcSoFar.shape[1] - 1 - int(lag_signal)
    start = end - lookback
    if idx >= prcSoFar.shape[0] or start < 0:
        return np.nan
    old_price = prcSoFar[idx, start]
    if old_price <= 0:
        return np.nan
    return prcSoFar[idx, end] / old_price - 1.0


def _relative_return(prcSoFar, idx, lookback, lag_signal):
    end = prcSoFar.shape[1] - 1 - int(lag_signal)
    start = end - lookback
    if idx >= prcSoFar.shape[0] or start < 0:
        return np.nan
    old_prices = prcSoFar[:, start]
    new_prices = prcSoFar[:, end]
    moves = np.divide(
        new_prices,
        old_prices,
        out=np.ones_like(new_prices),
        where=old_prices > 0,
    ) - 1.0
    if len(moves) <= 1:
        return np.nan
    return moves[idx] - (np.sum(moves) - moves[idx]) / (len(moves) - 1)


def _own_return_z(prcSoFar, idx, lookback, vol_window):
    if idx >= prcSoFar.shape[0] or prcSoFar.shape[1] <= max(lookback, vol_window):
        return np.nan
    prices = prcSoFar[idx]
    rets = prices[1:] / prices[:-1] - 1.0
    vol = np.std(rets[-vol_window:])
    if vol <= 0:
        return np.nan
    return np.sum(rets[-lookback:]) / vol


def _market_spread_z(prcSoFar, idx, z_window):
    if idx >= prcSoFar.shape[0] or prcSoFar.shape[1] < z_window:
        return np.nan
    base_prices = prcSoFar[:, [0]]
    normalized = np.divide(
        prcSoFar,
        base_prices,
        out=np.ones_like(prcSoFar),
        where=base_prices > 0,
    )
    market_norm = np.mean(normalized, axis=0)
    asset_norm = normalized[idx]
    spread = np.log(asset_norm) - np.log(market_norm)
    recent = spread[-z_window:]
    std = np.std(recent, ddof=1)
    if std <= 0:
        return np.nan
    return (spread[-1] - np.mean(recent)) / std


def _lead_lag_return(prcSoFar, leader_idx, lag, corr_sign):
    signal_idx = prcSoFar.shape[1] - lag
    prev_idx = signal_idx - 1
    if leader_idx >= prcSoFar.shape[0] or prev_idx < 0:
        return np.nan
    old_price = prcSoFar[leader_idx, prev_idx]
    if old_price <= 0:
        return np.nan
    return corr_sign * (prcSoFar[leader_idx, signal_idx] / old_price - 1.0)


def _market_core_dollars(prcSoFar, dollar_limits):
    nins, nt = prcSoFar.shape
    if nt <= MARKET_LOOKBACK:
        return np.zeros(nins)

    current_prices = prcSoFar[:, -1]
    old_prices = prcSoFar[:, -1 - MARKET_LOOKBACK]

    asset_moves = np.divide(
        current_prices,
        old_prices,
        out=np.ones_like(current_prices),
        where=old_prices > 0,
    ) - 1.0
    market_move = np.mean(asset_moves)
    if abs(market_move) > MAX_ABS_MARKET_MOVE:
        return np.zeros(nins)

    if market_move > SHORT_AFTER_RALLY_THRESHOLD:
        direction = -1.0
        signal_excess = market_move - SHORT_AFTER_RALLY_THRESHOLD
    elif market_move < -LONG_AFTER_DROP_THRESHOLD:
        direction = 1.0
        signal_excess = abs(market_move) - LONG_AFTER_DROP_THRESHOLD
    else:
        return np.zeros(nins)

    position_scale = MIN_POSITION_SCALE + (
        MAX_POSITION_SCALE - MIN_POSITION_SCALE
    ) * np.tanh(SIGNAL_SCALE_SPEED * max(0.0, signal_excess))
    if direction < 0:
        position_scale *= SHORT_SIDE_MULTIPLIER

    return direction * position_scale * dollar_limits


def _stable_overlay_dollars(prcSoFar, dollar_limits):
    dollars = np.zeros(prcSoFar.shape[0])

    _set_target(dollars, 35, -_market_spread_z(prcSoFar, 35, 60), 1.0, dollar_limits)
    _set_target(dollars, 40, -_own_return_z(prcSoFar, 40, 2, 20), 0.5, dollar_limits)
    _set_target(dollars, 29, _lead_lag_return(prcSoFar, 4, 1, 1.0), 0.0, dollar_limits)
    _set_target(dollars, 14, _lead_lag_return(prcSoFar, 32, 1, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 3, _absolute_return(prcSoFar, 3, 10, True), 0.0, dollar_limits)
    _set_target(dollars, 47, _lead_lag_return(prcSoFar, 33, 2, 1.0), 0.0, dollar_limits)
    _set_target(dollars, 17, _lead_lag_return(prcSoFar, 7, 2, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 41, -_own_return_z(prcSoFar, 41, 5, 40), 0.5, dollar_limits)
    _set_target(dollars, 22, -_own_return_z(prcSoFar, 22, 5, 60), 1.0, dollar_limits)
    _set_target(dollars, 25, _lead_lag_return(prcSoFar, 32, 2, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 18, -_relative_return(prcSoFar, 18, 20, False), 0.0, dollar_limits)
    _set_target(dollars, 37, -_market_spread_z(prcSoFar, 37, 60), 1.0, dollar_limits)
    _set_target(dollars, 26, _lead_lag_return(prcSoFar, 33, 10, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 9, _absolute_return(prcSoFar, 9, 60, True), 0.0, dollar_limits)
    _set_target(dollars, 1, _lead_lag_return(prcSoFar, 40, 2, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 36, -_absolute_return(prcSoFar, 36, 20, False), 0.0, dollar_limits)
    _set_target(dollars, 33, -_absolute_return(prcSoFar, 33, 60, True), 0.0, dollar_limits)
    _set_target(dollars, 45, -_relative_return(prcSoFar, 45, 5, True), 0.0, dollar_limits)
    _set_target(dollars, 11, _relative_return(prcSoFar, 11, 5, True), 0.0, dollar_limits)
    _set_target(dollars, 8, _own_return_z(prcSoFar, 8, 5, 60), 0.5, dollar_limits)
    _set_target(dollars, 5, -_absolute_return(prcSoFar, 5, 10, True), 0.0, dollar_limits)
    _set_target(dollars, 27, -_absolute_return(prcSoFar, 27, 2, False), 0.0, dollar_limits)
    _set_target(dollars, 50, -_absolute_return(prcSoFar, 50, 2, True), 0.0, dollar_limits)
    _set_target(dollars, 24, _relative_return(prcSoFar, 24, 10, True), 0.0, dollar_limits)
    _set_target(dollars, 20, _absolute_return(prcSoFar, 20, 40, False), 0.0, dollar_limits)
    _set_target(dollars, 2, _lead_lag_return(prcSoFar, 3, 2, -1.0), 0.0, dollar_limits)
    _set_target(dollars, 42, -_absolute_return(prcSoFar, 42, 10, True), 0.0, dollar_limits)
    _set_target(dollars, 4, _lead_lag_return(prcSoFar, 33, 1, 1.0), 0.0, dollar_limits)
    _set_target(dollars, 16, -_absolute_return(prcSoFar, 16, 10, False), 0.0, dollar_limits)
    _set_target(dollars, 44, -_own_return_z(prcSoFar, 44, 2, 60), 1.0, dollar_limits)

    return np.clip(dollars, -dollar_limits, dollar_limits)


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt <= MARKET_LOOKBACK:
        return np.zeros(nins)

    dollar_limits = _dollar_limits(nins)
    dollar_targets = (
        MARKET_CORE_WEIGHT * _market_core_dollars(prcSoFar, dollar_limits)
        + STABLE_OVERLAY_WEIGHT * _stable_overlay_dollars(prcSoFar, dollar_limits)
    )
    dollar_targets = np.clip(dollar_targets, -dollar_limits, dollar_limits)

    current_prices = prcSoFar[:, -1]
    positions = np.divide(
        dollar_targets,
        current_prices,
        out=np.zeros_like(dollar_targets),
        where=current_prices > 0,
    )
    return positions.astype(int)
