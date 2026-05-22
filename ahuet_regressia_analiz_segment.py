import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

# =========================================================
# CONFIG
# =========================================================

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://courier:1337@localhost:1338/coffee"
)

TABLE_NAME = os.getenv("ORDERS_TABLE", "orders")
TIMESTAMP_COLUMN = os.getenv("ORDERS_TIMESTAMP_COLUMN", "prepare_until")
LOCATION_COLUMN = os.getenv("ORDERS_RESTAURANT_COLUMN", "location_id")
OUTPUT_DIR = "ahuet_analiz_segment_csv"

SEGMENT_ORDER = [
    "deep_night",
    "opening",
    "early_morning",
    "morning",
    "lunch",
    "afternoon",
    "dinner",
    "late_evening",
]
SEGMENT_RANK = {name: idx for idx, name in enumerate(SEGMENT_ORDER)}

# =========================================================
# DB
# =========================================================

def build_engine():
    return create_engine(DB_URL)

def load_work_hours_df(engine):
    wh_sql = (
        "SELECT location_id, weekday, working, start_hour, start_minutes, "
        "finish_hour, finish_minutes FROM work_hours where working = true"
    )
    try:
        work_hours_df = pd.read_sql_query(text(wh_sql), engine)
    except Exception:
        return None
    if work_hours_df.empty:
        return None
    work_hours_df["location_id"] = work_hours_df["location_id"].astype(str)
    return work_hours_df

# =========================================================
# LOAD DATA
# =========================================================

def load_orders(engine, limit=None):

    query = [
        f"SELECT {LOCATION_COLUMN}, {TIMESTAMP_COLUMN} FROM {TABLE_NAME}",
    ]
    query.append(f"ORDER BY {TIMESTAMP_COLUMN} DESC")
    if limit:
        query.append(f"LIMIT {int(limit)}")

    sql = " ".join(query)
    return pd.read_sql_query(text(sql), engine)

# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(df):

    df = df.copy()

    # UTC -> datetime
    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        unit="ms",
        errors="coerce",
        utc=True
    )

    # Moscow timezone
    df[TIMESTAMP_COLUMN] = (
        df[TIMESTAMP_COLUMN]
        .dt.tz_convert("Europe/Moscow")
    )

    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT

    # remove bad rows
    df = df.dropna(
        subset=[
            LOCATION_COLUMN,
            TIMESTAMP_COLUMN
        ]
    )

    # year filter
    df = df[
        (df[TIMESTAMP_COLUMN].dt.year >= 2024)
        & (df[TIMESTAMP_COLUMN].dt.year <= 2027)
    ]

    # string ids
    df[LOCATION_COLUMN] = df[LOCATION_COLUMN].astype(str)

    # floor to hour
    df["datetime_hour"] = (
        df[TIMESTAMP_COLUMN]
        .dt.floor("h")
    )

    return df

# =========================================================
# BUILD HOURLY DATASET
# =========================================================

def build_hourly_dataset(df):

    hourly_df = (
        df.groupby(
            [LOCATION_COLUMN, "datetime_hour"]
        )
        .size()
        .reset_index(name="orders_count")
    )

    return hourly_df

def expand_hourly_grid(hourly_df):
    hourly_df = (
        hourly_df
        .set_index([LOCATION_COLUMN, "datetime_hour"])
        .sort_index()
    )

    def _expand(group):
        dt_index = group.index.get_level_values(1)
        idx = pd.date_range(
            dt_index.min(),
            dt_index.max(),
            freq="h",
            tz=dt_index.tz
        )
        loc = group.index.get_level_values(0)[0]
        new_index = pd.MultiIndex.from_product(
            [[loc], idx],
            names=[LOCATION_COLUMN, "datetime_hour"]
        )
        return group.reindex(new_index)

    expanded = (
        hourly_df
        .groupby(level=0, group_keys=False)
        .apply(_expand)
        .reset_index()
    )

    expanded["orders_count"] = expanded["orders_count"].fillna(0)
    return expanded

def add_time_features(df):
    df["hour"] = df["datetime_hour"].dt.hour
    df["weekday"] = df["datetime_hour"].dt.dayofweek
    df["month"] = df["datetime_hour"].dt.month
    df["is_weekend"] = (
        df["weekday"]
        .isin([5, 6])
        .astype(int)
    )
    return df

# =========================================================
# WORK HOURS / SEGMENTS
# =========================================================

def is_open_hour(location_id, weekday, hour, work_hours_df):
    if work_hours_df is None or work_hours_df.empty:
        return True

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id)
        & (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return False

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    if start_hour < finish_hour:
        return start_hour <= hour < finish_hour

    return hour >= start_hour or hour < finish_hour

def add_is_open_flag(df, work_hours_df):
    df["is_open"] = df.apply(
        lambda row: is_open_hour(
            row[LOCATION_COLUMN],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    return df

def classify_hour(location_id, weekday, hour, work_hours_df):

    if work_hours_df is None or work_hours_df.empty:
        return _classify_by_hour(hour)

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id)
        & (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return "closed"

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    if start_hour < finish_hour:
        is_open = start_hour <= hour < finish_hour
    else:
        is_open = (
            hour >= start_hour
            or hour < finish_hour
        )

    if not is_open:
        return "closed"

    return _classify_by_hour(hour)

def _classify_by_hour(hour):
    if 0 <= hour < 6:
        return "deep_night"

    if 6 <= hour < 8:
        return "opening"

    if 8 <= hour < 9:
        return "early_morning"

    if 9 <= hour < 11:
        return "morning"

    if 11 <= hour < 15:
        return "lunch"

    if 15 <= hour < 18:
        return "afternoon"

    if 18 <= hour < 21:
        return "dinner"

    if 21 <= hour < 24:
        return "late_evening"

    return "other"

def add_time_segment(df, work_hours_df):
    df["time_segment"] = df.apply(
        lambda row: classify_hour(
            row[LOCATION_COLUMN],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    return df

# =========================================================
# BUILD SEGMENT DATASET
# =========================================================

def build_segment_dataset(hourly_df):
    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["datetime_hour"].dt.floor("D")
    hourly_df["segment_rank"] = hourly_df["time_segment"].map(SEGMENT_RANK)

    segment_df = (
        hourly_df.groupby(
            [LOCATION_COLUMN, "segment", "date", "time_segment", "segment_rank"]
        )
        .agg(
            orders_count=("orders_count", "sum"),
            open_hours=("is_open", "sum"),
        )
        .reset_index()
    )

    return segment_df

def expand_segment_grid(segment_df):
    segments = pd.DataFrame({
        "time_segment": SEGMENT_ORDER,
        "segment_rank": [SEGMENT_RANK[s] for s in SEGMENT_ORDER],
    })

    grids = []
    for location_id, group in segment_df.groupby(LOCATION_COLUMN):
        location_segment = group["segment"].iloc[0]
        date_index = pd.date_range(
            group["date"].min(),
            group["date"].max(),
            freq="D"
        )
        grid = pd.MultiIndex.from_product(
            [date_index, segments["time_segment"]],
            names=["date", "time_segment"]
        ).to_frame(index=False)
        grid["segment_rank"] = grid["time_segment"].map(SEGMENT_RANK)
        grid[LOCATION_COLUMN] = location_id
        grid["segment"] = location_segment
        grids.append(grid)

    grid_df = pd.concat(grids, ignore_index=True)

    merged = grid_df.merge(
        segment_df,
        on=[LOCATION_COLUMN, "segment", "date", "time_segment", "segment_rank"],
        how="left"
    )

    merged["orders_count"] = merged["orders_count"].fillna(0)
    merged["open_hours"] = merged["open_hours"].fillna(0)

    merged["segment_datetime"] = (
        pd.to_datetime(merged["date"])
        + pd.to_timedelta(merged["segment_rank"], unit="h")
    )

    merged["weekday"] = pd.to_datetime(merged["date"]).dt.dayofweek
    merged["month"] = pd.to_datetime(merged["date"]).dt.month
    merged["is_weekend"] = (
        merged["weekday"]
        .isin([5, 6])
        .astype(int)
    )

    return merged

# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_segment_features(df):

    df = df.copy()

    df = df.sort_values(
        by=[
            LOCATION_COLUMN,
            "segment_datetime"
        ]
    )

    df["open_orders"] = df["orders_count"].where(df["open_hours"] > 0)

    df["lag_1seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(1)
    )

    df["lag_2seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(2)
    )

    df["lag_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(8)
    )

    df["prev_open_1seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(1)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["prev_open_2seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(2)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["prev_open_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .shift(8)
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    df["rolling_mean_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    df["rolling_std_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .std()
        )
    )

    df["rolling_open_count_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_hours"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .sum()
        )
    )

    for col in [
        "lag_1seg",
        "lag_2seg",
        "lag_8seg",
        "rolling_mean_3seg",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
    ]:
        df[col] = df[col].fillna(0)

    df = df[df["open_hours"] > 0].copy()
    df = df.drop(columns=["open_orders"])

    return df

# =========================================================
# FEATURE MATRIX
# =========================================================

def build_feature_matrix(df):
    df = df.copy()

    seg_dummies = pd.get_dummies(
        df["time_segment"],
        prefix="seg",
        drop_first=True
    )
    df = pd.concat([df, seg_dummies], axis=1)

    features = [
        "weekday",
        "month",
        "is_weekend",
        "open_hours",

        "lag_1seg",
        "lag_2seg",
        "lag_8seg",

        "prev_open_1seg",
        "prev_open_2seg",
        "prev_open_8seg",

        "rolling_mean_3seg",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
    ]

    features += [c for c in df.columns if c.startswith("seg_")]

    return df, features

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def split_train_test(df):

    split_date = df["segment_datetime"].quantile(0.8)

    train_df = (
        df[df["segment_datetime"] <= split_date]
    )

    test_df = (
        df[df["segment_datetime"] > split_date]
    )

    return train_df, test_df

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(train_df, features):

    target = "orders_count"

    X_train = train_df[features]
    y_train = train_df[target]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model

# =========================================================
# EVALUATE
# =========================================================

def evaluate_model(
    model,
    test_df,
    features
):

    X_test = test_df[features]
    y_test = test_df["orders_count"]

    predictions = model.predict(X_test)

    result_df = test_df.copy()
    result_df["prediction"] = predictions
    result_df["error"] = result_df["prediction"] - result_df["orders_count"]
    result_df["abs_error"] = result_df["error"].abs()

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("\n=== GLOBAL METRICS ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    segment_metrics_df = (
        result_df.groupby("segment")
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
    )

    segment_time_metrics_df = (
        result_df.groupby(["segment", "time_segment"])
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
                "rmse": np.sqrt(mean_squared_error(g["orders_count"], g["prediction"])),
                "mean_orders": g["orders_count"].mean(),
                "mean_pred": g["prediction"].mean(),
            })
        )
        .reset_index()
    )

    print("\n=== METRICS BY RESTAURANT SEGMENT + TIME SEGMENT ===")

    for rest_segment in ["low", "medium", "high", "mega"]:
        rest_df = result_df[result_df["segment"] == rest_segment]
        if rest_df.empty:
            continue

        print(f"\nRESTAURANT SEGMENT: {rest_segment}")
        seg_time_df = segment_time_metrics_df[
            segment_time_metrics_df["segment"] == rest_segment
        ]
        print(
            seg_time_df.sort_values("time_segment")
            .to_string(index=False)
        )

        examples = (
            rest_df.sort_values("segment_datetime")
            .groupby("time_segment", group_keys=False)
            .head(3)
        )

        print("\nEXAMPLES:")
        print(
            examples[
                [
                    LOCATION_COLUMN,
                    "segment_datetime",
                    "time_segment",
                    "orders_count",
                    "prediction",
                ]
            ]
            .to_string(index=False)
        )

    global_metrics = {
        "mae": mae,
        "rmse": rmse,
        "rows": len(result_df),
    }

    return result_df, global_metrics, segment_metrics_df, segment_time_metrics_df

# =========================================================
# UTILS
# =========================================================

def build_location_segments(df):

    location_stats = (
        df.groupby(LOCATION_COLUMN)
        .size()
        .reset_index(name="orders_count")
    )

    p50 = location_stats["orders_count"].quantile(0.50)
    p90 = location_stats["orders_count"].quantile(0.90)
    p99 = location_stats["orders_count"].quantile(0.99)

    location_stats["segment"] = (
        location_stats["orders_count"]
        .apply(
            lambda x:
            classify_location(
                x,
                p50,
                p90,
                p99
            )
        )
    )

    return location_stats[
        [LOCATION_COLUMN, "segment"]
    ]

def print_feature_importance(
    model,
    features
):

    importance_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_
    })

    importance_df["abs_coef"] = (
        importance_df["coefficient"]
        .abs()
    )

    importance_df = (
        importance_df
        .sort_values(
            by="abs_coef",
            ascending=False
        )
    )

    print("\n=== FEATURE IMPORTANCE ===")

    print(
        importance_df[
            ["feature", "coefficient"]
        ]
        .to_string(index=False)
    )

    print("\n=== INTERCEPT ===")
    print(model.intercept_)

    return importance_df

def save_outputs(
    model,
    result_df,
    global_metrics,
    segment_metrics_df,
    segment_time_metrics_df,
    feature_importance_df,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    metrics_df = pd.DataFrame([
        {"metric": "mae", "value": global_metrics["mae"]},
        {"metric": "rmse", "value": global_metrics["rmse"]},
        {"metric": "rows", "value": global_metrics["rows"]},
    ])

    metrics_df.to_csv(
        os.path.join(output_dir, "global_metrics.csv"),
        index=False
    )

    segment_metrics_df.to_csv(
        os.path.join(output_dir, "segment_metrics.csv"),
        index=False
    )

    segment_time_metrics_df.to_csv(
        os.path.join(output_dir, "segment_time_metrics.csv"),
        index=False
    )

    feature_importance_df.to_csv(
        os.path.join(output_dir, "feature_importance.csv"),
        index=False
    )

    model_params_df = pd.DataFrame({
        "feature": feature_importance_df["feature"],
        "coefficient": feature_importance_df["coefficient"],
    })
    model_params_df = pd.concat(
        [
            pd.DataFrame([
                {"feature": "intercept", "coefficient": model.intercept_}
            ]),
            model_params_df,
        ],
        ignore_index=True
    )

    model_params_df.to_csv(
        os.path.join(output_dir, "model_params.csv"),
        index=False
    )

    errors_df = (
        result_df
        .sort_values("abs_error", ascending=False)
        .head(1000)
        [[
            LOCATION_COLUMN,
            "segment_datetime",
            "segment",
            "time_segment",
            "orders_count",
            "prediction",
            "error",
            "abs_error",
        ]]
    )

    errors_df.to_csv(
        os.path.join(output_dir, "errors_top_1000.csv"),
        index=False
    )

    examples_df = (
        result_df
        .sort_values("segment_datetime")
        .groupby(["segment", "time_segment"], group_keys=False)
        .head(5)
        [[
            LOCATION_COLUMN,
            "segment_datetime",
            "segment",
            "time_segment",
            "orders_count",
            "prediction",
        ]]
    )

    examples_df.to_csv(
        os.path.join(output_dir, "examples_by_segment_time.csv"),
        index=False
    )

def classify_location(order_count, p50, p90, p99):

    if order_count <= p50:
        return "low"

    elif order_count <= p90:
        return "medium"

    elif order_count <= p99:
        return "high"

    return "mega"

# =========================================================
# MAIN
# =========================================================

def main():

    print("\n=== BUILD ENGINE ===")
    engine = build_engine()

    print("\n=== LOAD ORDERS ===")
    df = load_orders(
        engine,
        limit=1_000_000
    )

    work_hours_df = load_work_hours_df(engine)

    print(df.head())

    print("\n=== PREPARE DATA ===")
    df = prepare_data(df)
    print("\n=== BUILD LOCATION SEGMENTS ===")

    location_segments = (
        build_location_segments(df)
    )

    print(
        location_segments["segment"]
        .value_counts()
    )

    print("\n=== BUILD HOURLY DATASET ===")
    hourly_df = build_hourly_dataset(df)
    hourly_df = expand_hourly_grid(hourly_df)
    hourly_df = add_time_features(hourly_df)
    hourly_df = add_is_open_flag(hourly_df, work_hours_df)
    hourly_df = add_time_segment(hourly_df, work_hours_df)
    hourly_df = hourly_df.merge(
        location_segments,
        on=LOCATION_COLUMN,
        how="left"
    )

    print(hourly_df.head())

    print("\n=== BUILD SEGMENT DATASET ===")
    segment_df = build_segment_dataset(hourly_df)
    segment_df = expand_segment_grid(segment_df)

    print(segment_df.head())

    print("\n=== CREATE FEATURES ===")
    feature_df = create_segment_features(segment_df)
    feature_df, features = build_feature_matrix(feature_df)

    print(feature_df.head())

    print("\n=== SPLIT TRAIN / TEST ===")
    train_df, test_df = split_train_test(
        feature_df
    )

    print(
        len(train_df),
        len(test_df)
    )

    print("\n=== TRAIN MODEL ===")
    model = train_model(
        train_df,
        features
    )

    print("\n=== EVALUATE MODEL ===")
    (
        result_df,
        global_metrics,
        segment_metrics_df,
        segment_time_metrics_df,
    ) = evaluate_model(
        model,
        test_df,
        features
    )
    feature_importance_df = print_feature_importance(
        model,
        features
    )

    save_outputs(
        model,
        result_df,
        global_metrics,
        segment_metrics_df,
        segment_time_metrics_df,
        feature_importance_df,
        OUTPUT_DIR
    )

if __name__ == "__main__":
    main()
