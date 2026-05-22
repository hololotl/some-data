import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

from novaya_crutay_model_log import (
    LOCATION_COLUMN,
    build_engine,
    load_orders,
    load_work_hours_df,
    prepare_data,
    build_location_segments,
    build_hourly_dataset,
    expand_hourly_grid,
    add_time_features,
    add_is_open_flag,
    add_time_segment,
    build_segment_dataset,
    expand_segment_grid,
    create_segment_features,
    split_train_test,
    add_location_stats,
    build_feature_matrix,
    train_catboost,
    evaluate_predictions,
    print_segment_time_stats,
    predict_model,
    compute_peak_metrics,
    save_outputs,
)

OUTPUT_DIR_V2 = "ahuet_analiz_segment_csv_log_v2"

HARD_TIME_SEGMENTS = ["lunch", "deep_night", "opening", "dinner"]
HARD_REST_SEGMENTS = ["medium", "high", "mega"]


def add_v2_features(df):
    df = df.copy()
    df = df.sort_values(by=[LOCATION_COLUMN, "segment_datetime"])

    df["dayofyear"] = df["segment_datetime"].dt.dayofyear
    df["weekofyear"] = (
        df["segment_datetime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 366.0)

    grouped = df.groupby(LOCATION_COLUMN)["orders_count"]

    df["lag_16seg"] = grouped.shift(16)
    df["lag_24seg"] = grouped.shift(24)
    df["lag_112seg"] = grouped.shift(112)

    df["rolling_mean_14seg"] = grouped.transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean()
    )
    df["rolling_std_14seg"] = grouped.transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).std()
    )
    df["rolling_median_7d"] = grouped.transform(
        lambda x: x.shift(1).rolling(56, min_periods=1).median()
    )

    df["rolling_ratio_3_to_14"] = df["rolling_mean_3seg"] / (df["rolling_mean_14seg"] + 1.0)
    df["lag_diff_1_8"] = df["lag_1seg"] - df["lag_8seg"]
    df["lag_ratio_1_56"] = df["lag_1seg"] / (df["lag_56seg"] + 1.0)

    df["is_lunch_or_dinner"] = (
        df["time_segment"]
        .isin(["lunch", "dinner"])
        .astype(int)
    )
    df["is_hard_time_segment"] = (
        df["time_segment"]
        .isin(HARD_TIME_SEGMENTS)
        .astype(int)
    )

    fill_cols = [
        "lag_16seg",
        "lag_24seg",
        "lag_112seg",
        "rolling_mean_14seg",
        "rolling_std_14seg",
        "rolling_median_7d",
        "rolling_ratio_3_to_14",
        "lag_diff_1_8",
        "lag_ratio_1_56",
    ]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    return df


def build_feature_matrix_v2(df):
    features, cat_features = build_feature_matrix(df)

    extra_features = [
        "dayofyear",
        "weekofyear",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "doy_sin",
        "doy_cos",
        "lag_16seg",
        "lag_24seg",
        "lag_112seg",
        "rolling_mean_14seg",
        "rolling_std_14seg",
        "rolling_median_7d",
        "rolling_ratio_3_to_14",
        "lag_diff_1_8",
        "lag_ratio_1_56",
        "is_lunch_or_dinner",
        "is_hard_time_segment",
    ]

    for col in extra_features:
        if col in df.columns and col not in features and col not in cat_features:
            features.append(col)

    if LOCATION_COLUMN in df.columns and LOCATION_COLUMN not in cat_features:
        cat_features.append(LOCATION_COLUMN)

    return features, cat_features


def compute_sample_weight(train_df):
    peak_thresholds = (
        train_df.groupby("segment")["orders_count"]
        .quantile(0.97)
        .to_dict()
    )

    peak_flag = (
        train_df["orders_count"] >= train_df["segment"].map(peak_thresholds)
    ).astype(float)
    hard_time_flag = train_df["time_segment"].isin(HARD_TIME_SEGMENTS).astype(float)
    hard_seg_flag = train_df["segment"].isin(HARD_REST_SEGMENTS).astype(float)

    sample_weight = (
        1.0
        + 0.90 * peak_flag
        + 0.25 * hard_time_flag
        + 0.25 * hard_seg_flag
        + 0.40 * (peak_flag * hard_time_flag)
    )
    return sample_weight


def train_single_log_model(train_df, features, cat_features, params):
    X_train = train_df[features + cat_features]
    y_train = np.log1p(train_df["orders_count"])
    sample_weight = compute_sample_weight(train_df)

    cat_indices = [X_train.columns.get_loc(c) for c in cat_features]

    model = CatBoostRegressor(
        random_seed=42,
        verbose=100,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        sample_weight=sample_weight,
    )
    return model


def predict_single_log_model(model, df, features, cat_features):
    X = df[features + cat_features]
    pred = model.predict(X)
    pred = np.expm1(pred)
    pred = np.clip(pred, 0, None)
    return pred


def tuning_objective(y_true, y_pred, valid_df):
    mae = mean_absolute_error(y_true, y_pred)

    valid_peak_threshold = np.quantile(y_true, 0.95)
    peak_mask = y_true >= valid_peak_threshold
    hard_mask = (
        valid_df["time_segment"].isin(HARD_TIME_SEGMENTS)
        | valid_df["segment"].isin(HARD_REST_SEGMENTS)
    )

    peak_mae = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]) if peak_mask.any() else mae
    hard_mae = mean_absolute_error(y_true[hard_mask], y_pred[hard_mask]) if hard_mask.any() else mae

    return mae + 0.20 * peak_mae + 0.10 * hard_mae


def find_best_blend_weights(valid_df, pred_rmse, pred_mae, pred_quantile):
    y_true = valid_df["orders_count"].to_numpy()

    best_score = float("inf")
    best_weights = (0.65, 0.25, 0.10)

    for w_rmse in np.arange(0.45, 0.86, 0.05):
        for w_mae in np.arange(0.10, 0.46, 0.05):
            w_quantile = 1.0 - w_rmse - w_mae
            if w_quantile < 0.0:
                continue

            blended = (
                w_rmse * pred_rmse
                + w_mae * pred_mae
                + w_quantile * pred_quantile
            )

            score = tuning_objective(y_true, blended, valid_df)
            if score < best_score:
                best_score = score
                best_weights = (w_rmse, w_mae, w_quantile)

    return best_weights, best_score


def train_v2_ensemble(train_df, features, cat_features):
    split_point = train_df["segment_datetime"].quantile(0.85)
    subtrain = train_df[train_df["segment_datetime"] <= split_point].copy()
    valid = train_df[train_df["segment_datetime"] > split_point].copy()

    if len(subtrain) < 5000 or len(valid) < 1000:
        subtrain = train_df.copy()
        valid = train_df.tail(min(5000, len(train_df))).copy()

    rmse_params = {
        "iterations": 2300,
        "learning_rate": 0.028,
        "depth": 8,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "l2_leaf_reg": 9.0,
    }
    mae_params = {
        "iterations": 1800,
        "learning_rate": 0.035,
        "depth": 7,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "l2_leaf_reg": 11.0,
    }
    quantile_params = {
        "iterations": 1700,
        "learning_rate": 0.035,
        "depth": 7,
        "loss_function": "Quantile:alpha=0.65",
        "eval_metric": "Quantile:alpha=0.65",
        "l2_leaf_reg": 12.0,
    }

    print("\n=== TRAIN V2 MODEL: RMSE LOG ===")
    rmse_model_sub = train_single_log_model(subtrain, features, cat_features, rmse_params)

    print("\n=== TRAIN V2 MODEL: MAE LOG ===")
    mae_model_sub = train_single_log_model(subtrain, features, cat_features, mae_params)

    print("\n=== TRAIN V2 MODEL: QUANTILE LOG ===")
    quantile_model_sub = train_single_log_model(subtrain, features, cat_features, quantile_params)

    pred_rmse_valid = predict_single_log_model(rmse_model_sub, valid, features, cat_features)
    pred_mae_valid = predict_single_log_model(mae_model_sub, valid, features, cat_features)
    pred_quantile_valid = predict_single_log_model(quantile_model_sub, valid, features, cat_features)

    best_weights, best_score = find_best_blend_weights(
        valid,
        pred_rmse_valid,
        pred_mae_valid,
        pred_quantile_valid,
    )

    print("\n=== V2 BLEND SEARCH ===")
    print(
        "Best weights (rmse/mae/quantile): "
        f"{best_weights[0]:.2f} / {best_weights[1]:.2f} / {best_weights[2]:.2f}"
    )
    print(f"Validation objective score: {best_score:.6f}")

    print("\n=== RETRAIN V2 ENSEMBLE ON FULL TRAIN ===")
    rmse_model = train_single_log_model(train_df, features, cat_features, rmse_params)
    mae_model = train_single_log_model(train_df, features, cat_features, mae_params)
    quantile_model = train_single_log_model(train_df, features, cat_features, quantile_params)

    models = {
        "rmse": rmse_model,
        "mae": mae_model,
        "quantile": quantile_model,
    }
    return models, best_weights


def predict_v2_ensemble(models, weights, df, features, cat_features):
    pred_rmse = predict_single_log_model(models["rmse"], df, features, cat_features)
    pred_mae = predict_single_log_model(models["mae"], df, features, cat_features)
    pred_quantile = predict_single_log_model(models["quantile"], df, features, cat_features)

    prediction = (
        weights[0] * pred_rmse
        + weights[1] * pred_mae
        + weights[2] * pred_quantile
    )

    result_df = df.copy()
    result_df["prediction"] = np.clip(prediction, 0, None)
    result_df["error"] = result_df["prediction"] - result_df["orders_count"]
    result_df["abs_error"] = result_df["error"].abs()
    return result_df


def main():
    print("\n=== BUILD ENGINE ===")
    engine = build_engine()

    print("\n=== LOAD ORDERS ===")
    df = load_orders(engine, limit=1_000_000)
    work_hours_df = load_work_hours_df(engine)
    print(df.head())

    print("\n=== PREPARE DATA ===")
    df = prepare_data(df)

    print("\n=== BUILD LOCATION SEGMENTS ===")
    location_segments = build_location_segments(df)
    print(location_segments["segment"].value_counts())

    print("\n=== BUILD HOURLY DATASET ===")
    hourly_df = build_hourly_dataset(df)
    hourly_df = expand_hourly_grid(hourly_df)
    hourly_df = add_time_features(hourly_df)
    hourly_df = add_is_open_flag(hourly_df, work_hours_df)
    hourly_df = add_time_segment(hourly_df, work_hours_df)
    hourly_df = hourly_df.merge(location_segments, on=LOCATION_COLUMN, how="left")
    print(hourly_df.head())

    print("\n=== BUILD SEGMENT DATASET ===")
    segment_df = build_segment_dataset(hourly_df)
    segment_df = expand_segment_grid(segment_df)
    print(segment_df.head())

    print("\n=== CREATE BASE FEATURES ===")
    feature_df = create_segment_features(segment_df)

    print("\n=== SPLIT TRAIN / TEST ===")
    train_df, test_df = split_train_test(feature_df)
    print(len(train_df), len(test_df))

    feature_df = add_location_stats(train_df, feature_df)
    feature_df = add_v2_features(feature_df)
    train_df, test_df = split_train_test(feature_df)

    features, cat_features = build_feature_matrix_v2(feature_df)

    print("\n=== TRAIN BASELINE LOG MODEL ===")
    baseline_log = train_catboost(
        train_df,
        features,
        cat_features,
        use_log=True,
    )

    print("\n=== TRAIN V2 ENSEMBLE ===")
    v2_models, blend_weights = train_v2_ensemble(train_df, features, cat_features)

    print("\n=== EVALUATE BASELINE LOG MODEL ===")
    baseline_result = predict_model(
        baseline_log,
        test_df,
        features,
        cat_features,
        use_log=True,
        clip_negative=True,
    )
    (
        baseline_global,
        baseline_segment,
        baseline_time,
        baseline_segment_time,
    ) = evaluate_predictions(baseline_result)
    print_segment_time_stats(baseline_result, baseline_segment_time, "BASELINE_LOG")

    print("\n=== EVALUATE V2 ENSEMBLE ===")
    result_v2 = predict_v2_ensemble(v2_models, blend_weights, test_df, features, cat_features)
    (
        global_v2,
        segment_v2,
        time_v2,
        segment_time_v2,
    ) = evaluate_predictions(result_v2)
    print_segment_time_stats(result_v2, segment_time_v2, "V2_ENSEMBLE")

    train_v2_result = predict_v2_ensemble(v2_models, blend_weights, train_df, features, cat_features)
    train_metrics_v2, _, _, _ = evaluate_predictions(train_v2_result)

    v2_peak_metrics, v2_peak_examples = compute_peak_metrics(train_df, result_v2)
    save_outputs(
        v2_models["rmse"],
        result_v2,
        global_v2,
        segment_v2,
        segment_time_v2,
        v2_peak_metrics,
        v2_peak_examples,
        OUTPUT_DIR_V2,
    )

    print("\n=== COMPARISON: OVERALL ===")
    print(f"BASELINE LOG MAE: {baseline_global['mae']:.4f} | V2 MAE: {global_v2['mae']:.4f}")
    print(f"BASELINE LOG RMSE: {baseline_global['rmse']:.4f} | V2 RMSE: {global_v2['rmse']:.4f}")

    print("\n=== COMPARISON: RESTAURANT SEGMENT MAE ===")
    seg_compare = baseline_segment[["segment", "mae"]].merge(
        segment_v2[["segment", "mae"]],
        on="segment",
        suffixes=("_baseline", "_v2"),
    )
    seg_compare["delta"] = seg_compare["mae_v2"] - seg_compare["mae_baseline"]
    print(seg_compare.to_string(index=False))

    print("\n=== COMPARISON: TIME SEGMENT MAE ===")
    time_compare = baseline_time.merge(
        time_v2,
        on="time_segment",
        suffixes=("_baseline", "_v2"),
    )
    time_compare["delta"] = time_compare["mae_v2"] - time_compare["mae_baseline"]
    print(time_compare.to_string(index=False))

    print("\n=== OVERFITTING CHECK (V2) ===")
    print(f"V2 train/test RMSE: {train_metrics_v2['rmse']:.4f} / {global_v2['rmse']:.4f}")

    print("\n=== PEAK METRICS (P95 BY SEGMENT): V2 ===")
    print(v2_peak_metrics.to_string(index=False))

    print("\n=== SUMMARY ===")
    mae_delta = global_v2["mae"] - baseline_global["mae"]
    rmse_delta = global_v2["rmse"] - baseline_global["rmse"]
    if mae_delta < 0 and rmse_delta < 0:
        print("V2 ensemble improved both MAE and RMSE.")
    elif mae_delta > 0 and rmse_delta > 0:
        print("V2 ensemble worsened both MAE and RMSE.")
    else:
        print("V2 ensemble is mixed: one metric improved, another worsened.")


if __name__ == "__main__":
    main()
