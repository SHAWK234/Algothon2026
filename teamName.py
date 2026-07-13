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


def _dollar_limits(nins):
    dollar_limits = np.full(nins, DEFAULT_DOLLAR_LIMIT)
    dollar_limits[0] = INST0_DOLLAR_LIMIT
    valid_exclusions = EXCLUDED_ASSET_INDICES[EXCLUDED_ASSET_INDICES < nins]
    dollar_limits[valid_exclusions] = 0.0
    return dollar_limits


def _raw_position(prcSoFar):
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

    dollar_targets = direction * position_scale * _dollar_limits(nins)

    positions = np.divide(
        dollar_targets,
        current_prices,
        out=np.zeros_like(dollar_targets),
        where=current_prices > 0,
    )
    return positions.astype(int)


def getMyPosition(prcSoFar):
    nins, nt = prcSoFar.shape
    if nt <= MARKET_LOOKBACK:
        return np.zeros(nins)

    return _raw_position(prcSoFar)
