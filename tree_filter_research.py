import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import all_asset_deep_research as research


TRAIN_END = 350
TEST_END = 500
FEATURE_LOOKBACK = 60
BASE_WINDOW = 2
BASE_THRESHOLD = 0.0015
RANDOM_STATE = 20260712


def score(mean_pl, std_pl):
    return research.score(mean_pl, std_pl)


def load_prices():
    return pd.read_csv("prices.txt", sep=r"\s+").values.T


def base_direction(hist, threshold=BASE_THRESHOLD):
    if hist.shape[1] <= BASE_WINDOW:
        return 0.0
    market_move = np.mean(hist[:, -1] / hist[:, -1 - BASE_WINDOW] - 1.0)
    if abs(market_move) <= threshold:
        return 0.0
    return -np.sign(market_move)


def dollars_to_positions(dollars, prices):
    return research.target_position_from_dollars(dollars, prices)


def base_position_fn(threshold=BASE_THRESHOLD, trade_indices=None):
    if trade_indices is None:
        trade_indices = np.arange(research.N_INST)
    trade_indices = np.asarray(trade_indices, dtype=int)

    def position_fn(hist):
        direction = base_direction(hist, threshold)
        dollars = np.zeros(research.N_INST)
        if direction != 0:
            dollars[trade_indices] = direction * research.POSITION_LIMITS[trade_indices]
        return dollars_to_positions(dollars, hist[:, -1])

    return position_fn


def safe_return(hist, idx, window):
    if hist.shape[1] <= window:
        return 0.0
    return hist[idx, -1] / hist[idx, -1 - window] - 1.0


def market_features(hist):
    returns = hist[:, 1:] / hist[:, :-1] - 1.0
    market_returns = np.mean(returns, axis=0)
    latest_returns = returns[:, -1]

    features = {}
    for window in [1, 2, 3, 5, 10, 20, 40]:
        if hist.shape[1] > window:
            asset_moves = hist[:, -1] / hist[:, -1 - window] - 1.0
            features[f"market_ret_{window}"] = np.mean(asset_moves)
            features[f"market_abs_ret_{window}"] = abs(features[f"market_ret_{window}"])
            features[f"dispersion_ret_{window}"] = np.std(asset_moves)
            features[f"frac_positive_{window}"] = np.mean(asset_moves > 0)
        else:
            features[f"market_ret_{window}"] = 0.0
            features[f"market_abs_ret_{window}"] = 0.0
            features[f"dispersion_ret_{window}"] = 0.0
            features[f"frac_positive_{window}"] = 0.5

    for window in [5, 10, 20, 40, 60]:
        if len(market_returns) >= window:
            recent_market = market_returns[-window:]
            recent_dispersion = np.std(returns[:, -window:], axis=0)
            features[f"market_vol_{window}"] = np.std(recent_market)
            features[f"market_mean_{window}"] = np.mean(recent_market)
            features[f"dispersion_mean_{window}"] = np.mean(recent_dispersion)
        else:
            features[f"market_vol_{window}"] = 0.0
            features[f"market_mean_{window}"] = 0.0
            features[f"dispersion_mean_{window}"] = 0.0

    features["latest_dispersion"] = np.std(latest_returns)
    features["latest_frac_positive"] = np.mean(latest_returns > 0)
    features["base_direction"] = base_direction(hist, BASE_THRESHOLD)
    return features


MARKET_FEATURE_NAMES = sorted(market_features(np.ones((research.N_INST, FEATURE_LOOKBACK + 2))).keys())


def market_feature_vector(hist):
    features = market_features(hist)
    return np.array([features[name] for name in MARKET_FEATURE_NAMES], dtype=float)


def asset_feature_vector(hist, asset_idx, market=None):
    if market is None:
        market = market_features(hist)
    returns = hist[:, 1:] / hist[:, :-1] - 1.0
    market_returns = np.mean(returns, axis=0)
    asset_returns = returns[asset_idx]
    latest_asset_return = asset_returns[-1]
    latest_market_return = market_returns[-1]

    features = dict(market)
    features["asset_idx_norm"] = asset_idx / (research.N_INST - 1)
    features["is_algo"] = 1.0 if asset_idx == 0 else 0.0
    features["limit_norm"] = research.POSITION_LIMITS[asset_idx] / np.max(research.POSITION_LIMITS)
    features["asset_latest_return"] = latest_asset_return
    features["asset_latest_relative"] = latest_asset_return - latest_market_return

    for window in [1, 2, 3, 5, 10, 20, 40]:
        asset_move = safe_return(hist, asset_idx, window)
        if hist.shape[1] > window:
            market_move = np.mean(hist[:, -1] / hist[:, -1 - window] - 1.0)
        else:
            market_move = 0.0
        features[f"asset_ret_{window}"] = asset_move
        features[f"asset_abs_ret_{window}"] = abs(asset_move)
        features[f"asset_rel_ret_{window}"] = asset_move - market_move

    for window in [5, 10, 20, 40, 60]:
        if len(asset_returns) >= window:
            features[f"asset_vol_{window}"] = np.std(asset_returns[-window:])
            features[f"asset_mean_{window}"] = np.mean(asset_returns[-window:])
        else:
            features[f"asset_vol_{window}"] = 0.0
            features[f"asset_mean_{window}"] = 0.0

    return features


ASSET_FEATURE_NAMES = sorted(asset_feature_vector(np.ones((research.N_INST, FEATURE_LOOKBACK + 2)), 0).keys())


def asset_feature_array(hist, asset_idx, market=None):
    features = asset_feature_vector(hist, asset_idx, market)
    return np.array([features[name] for name in ASSET_FEATURE_NAMES], dtype=float)


def all_asset_feature_matrix(hist):
    market = market_features(hist)
    return np.vstack(
        [asset_feature_array(hist, asset_idx, market) for asset_idx in range(research.N_INST)]
    )


def build_day_dataset(prices, start_t, end_t, threshold=BASE_THRESHOLD):
    x_rows = []
    y_rows = []
    pnl_rows = []

    for t in range(max(FEATURE_LOOKBACK + 1, start_t), end_t):
        hist = prices[:, :t]
        direction = base_direction(hist, threshold)
        if direction == 0:
            continue
        next_returns = prices[:, t] / prices[:, t - 1] - 1.0
        gross_pnl = np.sum(direction * research.POSITION_LIMITS * next_returns)
        x_rows.append(market_feature_vector(hist))
        y_rows.append(1 if gross_pnl > 0 else 0)
        pnl_rows.append(gross_pnl)

    return np.array(x_rows), np.array(y_rows), np.array(pnl_rows)


def build_asset_dataset(prices, start_t, end_t, threshold=BASE_THRESHOLD):
    x_rows = []
    y_rows = []
    pnl_rows = []

    for t in range(max(FEATURE_LOOKBACK + 1, start_t), end_t):
        hist = prices[:, :t]
        direction = base_direction(hist, threshold)
        if direction == 0:
            continue
        market = market_features(hist)
        next_returns = prices[:, t] / prices[:, t - 1] - 1.0
        for asset_idx in range(research.N_INST):
            gross_pnl = direction * research.POSITION_LIMITS[asset_idx] * next_returns[asset_idx]
            x_rows.append(asset_feature_array(hist, asset_idx, market))
            y_rows.append(1 if gross_pnl > 0 else 0)
            pnl_rows.append(gross_pnl)

    return np.array(x_rows), np.array(y_rows), np.array(pnl_rows)


def fit_tree(x_train, y_train, max_depth, min_samples_leaf, ccp_alpha):
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=2 * min_samples_leaf,
        ccp_alpha=ccp_alpha,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(np.nan_to_num(x_train), y_train)
    return model


def positive_probability(model, x):
    proba = model.predict_proba(np.nan_to_num(x.reshape(1, -1)))[0]
    if len(model.classes_) == 1:
        return 1.0 if model.classes_[0] == 1 else 0.0
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return float(proba[positive_index])


def positive_probabilities(model, x):
    proba = model.predict_proba(np.nan_to_num(x))
    if len(model.classes_) == 1:
        return np.full(len(x), 1.0 if model.classes_[0] == 1 else 0.0)
    positive_index = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, positive_index]


def day_tree_position_fn(model, probability_threshold, base_threshold=BASE_THRESHOLD):
    def position_fn(hist):
        direction = base_direction(hist, base_threshold)
        dollars = np.zeros(research.N_INST)
        if direction == 0:
            return dollars
        probability = positive_probability(model, market_feature_vector(hist))
        if probability >= probability_threshold:
            dollars = direction * research.POSITION_LIMITS
        return dollars_to_positions(dollars, hist[:, -1])

    return position_fn


def asset_tree_position_fn(model, probability_threshold, base_threshold=BASE_THRESHOLD):
    def position_fn(hist):
        direction = base_direction(hist, base_threshold)
        dollars = np.zeros(research.N_INST)
        if direction == 0:
            return dollars
        probabilities = positive_probabilities(model, all_asset_feature_matrix(hist))
        selected = probabilities >= probability_threshold
        dollars[selected] = direction * research.POSITION_LIMITS[selected]
        return dollars_to_positions(dollars, hist[:, -1])

    return position_fn


def evaluate_candidate(prices, position_fn, start_day, end_day):
    return research.evaluate_interval(prices, start_day, end_day, position_fn)


def scan_baselines(prices):
    rows = []
    for threshold in [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]:
        train_val = evaluate_candidate(prices, base_position_fn(threshold), 250, TRAIN_END)
        test = evaluate_candidate(prices, base_position_fn(threshold), TRAIN_END, TEST_END)
        rows.append(
            {
                "kind": "baseline",
                "threshold": threshold,
                "train_val_score": train_val["score"],
                "train_val_mean": train_val["mean_pl"],
                "train_val_sharpe": train_val["sharpe"],
                "test_score": test["score"],
                "test_mean": test["mean_pl"],
                "test_std": test["std_pl"],
                "test_sharpe": test["sharpe"],
                "test_dvolume": test["dvolume"],
            }
        )
    return pd.DataFrame(rows).sort_values("train_val_score", ascending=False)


def scan_day_trees(prices):
    x_fit, y_fit, _ = build_day_dataset(prices, 0, 250)
    x_final, y_final, _ = build_day_dataset(prices, 0, TRAIN_END)
    rows = []
    models = {}

    for max_depth in [1, 2, 3]:
        for min_samples_leaf in [25, 40, 60, 80]:
            if len(y_fit) < 2 * min_samples_leaf or len(np.unique(y_fit)) < 2:
                continue
            for ccp_alpha in [0.0, 0.001, 0.003, 0.01]:
                model = fit_tree(x_fit, y_fit, max_depth, min_samples_leaf, ccp_alpha)
                for prob_threshold in [0.50, 0.55, 0.60, 0.65]:
                    position_fn = day_tree_position_fn(model, prob_threshold)
                    train_val = evaluate_candidate(prices, position_fn, 250, TRAIN_END)
                    rows.append(
                        {
                            "kind": "day_tree",
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "ccp_alpha": ccp_alpha,
                            "prob_threshold": prob_threshold,
                            "train_val_score": train_val["score"],
                            "train_val_mean": train_val["mean_pl"],
                            "train_val_sharpe": train_val["sharpe"],
                        }
                    )

    results = pd.DataFrame(rows).sort_values("train_val_score", ascending=False)
    best = results.iloc[0].to_dict()
    final_model = fit_tree(
        x_final,
        y_final,
        int(best["max_depth"]),
        int(best["min_samples_leaf"]),
        float(best["ccp_alpha"]),
    )
    test = evaluate_candidate(
        prices,
        day_tree_position_fn(final_model, float(best["prob_threshold"])),
        TRAIN_END,
        TEST_END,
    )
    for key, value in test.items():
        best[f"test_{key}"] = value
    return results, best, final_model


def scan_asset_trees(prices):
    x_fit, y_fit, _ = build_asset_dataset(prices, 0, 250)
    x_final, y_final, _ = build_asset_dataset(prices, 0, TRAIN_END)
    rows = []

    for max_depth in [1, 2, 3]:
        for min_samples_leaf in [1000, 2000]:
            if len(y_fit) < 2 * min_samples_leaf or len(np.unique(y_fit)) < 2:
                continue
            for ccp_alpha in [0.0, 0.001, 0.003]:
                model = fit_tree(x_fit, y_fit, max_depth, min_samples_leaf, ccp_alpha)
                for prob_threshold in [0.50, 0.55, 0.60]:
                    position_fn = asset_tree_position_fn(model, prob_threshold)
                    train_val = evaluate_candidate(prices, position_fn, 250, TRAIN_END)
                    rows.append(
                        {
                            "kind": "asset_tree",
                            "max_depth": max_depth,
                            "min_samples_leaf": min_samples_leaf,
                            "ccp_alpha": ccp_alpha,
                            "prob_threshold": prob_threshold,
                            "train_val_score": train_val["score"],
                            "train_val_mean": train_val["mean_pl"],
                            "train_val_sharpe": train_val["sharpe"],
                        }
                    )

    results = pd.DataFrame(rows).sort_values("train_val_score", ascending=False)
    best = results.iloc[0].to_dict()
    final_model = fit_tree(
        x_final,
        y_final,
        int(best["max_depth"]),
        int(best["min_samples_leaf"]),
        float(best["ccp_alpha"]),
    )
    test = evaluate_candidate(
        prices,
        asset_tree_position_fn(final_model, float(best["prob_threshold"])),
        TRAIN_END,
        TEST_END,
    )
    for key, value in test.items():
        best[f"test_{key}"] = value
    return results, best, final_model


def main():
    prices = load_prices()

    baseline_results = scan_baselines(prices)
    day_results, best_day, day_model = scan_day_trees(prices)
    asset_results, best_asset, asset_model = scan_asset_trees(prices)

    baseline_results.to_csv("tree_filter_baseline_results.csv", index=False)
    day_results.to_csv("tree_filter_day_results.csv", index=False)
    asset_results.to_csv("tree_filter_asset_results.csv", index=False)

    print("\nBaseline threshold scan, selected using only days 250-350")
    print(baseline_results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nBest day-level tree selected on days 250-350, tested on 350-500")
    print(pd.Series(best_day).to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nDay-level tree")
    print(export_text(day_model, feature_names=MARKET_FEATURE_NAMES, decimals=5))

    print("\nBest asset-level tree selected on days 250-350, tested on 350-500")
    print(pd.Series(best_asset).to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nAsset-level tree")
    print(export_text(asset_model, feature_names=ASSET_FEATURE_NAMES, decimals=5))

    comparison = pd.DataFrame(
        [
            baseline_results.iloc[0].to_dict()
            | {
                "selected_model": "best_train_baseline",
                "test_score": baseline_results.iloc[0]["test_score"],
            },
            {
                "selected_model": "day_tree",
                "train_val_score": best_day["train_val_score"],
                "test_score": best_day["test_score"],
                "test_mean": best_day["test_mean_pl"],
                "test_std": best_day["test_std_pl"],
                "test_sharpe": best_day["test_sharpe"],
                "test_dvolume": best_day["test_dvolume"],
            },
            {
                "selected_model": "asset_tree",
                "train_val_score": best_asset["train_val_score"],
                "test_score": best_asset["test_score"],
                "test_mean": best_asset["test_mean_pl"],
                "test_std": best_asset["test_std_pl"],
                "test_sharpe": best_asset["test_sharpe"],
                "test_dvolume": best_asset["test_dvolume"],
            },
        ]
    )
    print("\nFinal comparison")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    comparison.to_csv("tree_filter_final_comparison.csv", index=False)


if __name__ == "__main__":
    main()
