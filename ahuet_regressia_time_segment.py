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

    if 0 <= hour < 6:
        return "deep_night"

    elif 6 <= hour < 8:
        return "opening"

    elif 8 <= hour < 9:
        return "early_morning"

    elif 9 <= hour < 11:
        return "morning"

    elif 11 <= hour < 15:
        return "lunch"

    elif 15 <= hour < 18:
        return "afternoon"

    elif 18 <= hour < 21:
        return "dinner"

    elif 21 <= hour < 24:
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
# FEATURE ENGINEERING
# =========================================================

def create_features(df):

    df = df.copy()

    # sorting is IMPORTANT
    df = df.sort_values(
        by=[
            LOCATION_COLUMN,
            "datetime_hour"
        ]
    )

    df["open_orders"] = df["orders_count"].where(df["is_open"])

    # =====================================================
    # LAGS
    # =====================================================

    df["lag_1h"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(1)
    )

    df["lag_2h"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(2)
    )

    df["lag_24h"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(24)
    )

    df["prev_open_1h"] = (
        df.groupby(LOCATION_COLUMN)["is_open"]
        .shift(1)
        .fillna(False)
        .astype(int)
    )

    df["prev_open_2h"] = (
        df.groupby(LOCATION_COLUMN)["is_open"]
        .shift(2)
        .fillna(False)
        .astype(int)
    )

    df["prev_open_24h"] = (
        df.groupby(LOCATION_COLUMN)["is_open"]
        .shift(24)
        .fillna(False)
        .astype(int)
    )

    # =====================================================
    # ROLLING
    # =====================================================

    df["rolling_mean_3h"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .mean()
        )
    )

    df["rolling_std_3h"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .std()
        )
    )

    df["rolling_open_count_3h"] = (
        df.groupby(LOCATION_COLUMN)["is_open"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .sum()
        )
    )

    for col in [
        "lag_1h",
        "lag_2h",
        "lag_24h",
        "rolling_mean_3h",
        "rolling_std_3h",
        "rolling_open_count_3h",
    ]:
        df[col] = df[col].fillna(0)

    df = df[df["is_open"]].copy()
    df = df.drop(columns=["open_orders"])

    return df

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

def split_train_test(df):

    split_date = df["datetime_hour"].quantile(0.8)

    train_df = (
        df[df["datetime_hour"] <= split_date]
    )

    test_df = (
        df[df["datetime_hour"] > split_date]
    )

    return train_df, test_df

# =========================================================
# FEATURE MATRIX
# =========================================================

def build_feature_matrix(df, use_time_segment=True):
    df = df.copy()

    if use_time_segment:
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

        "lag_1h",
        "lag_2h",
        "lag_24h",

        "prev_open_1h",
        "prev_open_2h",
        "prev_open_24h",

        "rolling_mean_3h",
        "rolling_std_3h",
        "rolling_open_count_3h",
    ]

    if use_time_segment:
        features += [c for c in df.columns if c.startswith("seg_")]
    else:
        features += ["hour"]

    return df, features

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

    # ============================================
    # GLOBAL METRICS
    # ============================================

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

    # ============================================
    # METRICS BY SEGMENT
    # ============================================

    print("\n=== METRICS BY SEGMENT ===")

    for segment in [
        "low",
        "medium",
        "high",
        "mega"
    ]:

        seg_df = (
            result_df[
                result_df["segment"] == segment
            ]
        )

        if len(seg_df) == 0:
            continue

        seg_mae = mean_absolute_error(
            seg_df["orders_count"],
            seg_df["prediction"]
        )

        seg_rmse = np.sqrt(
            mean_squared_error(
                seg_df["orders_count"],
                seg_df["prediction"]
            )
        )

        print(f"\nSEGMENT: {segment}")
        print(f"rows: {len(seg_df)}")
        print(f"MAE: {seg_mae:.2f}")
        print(f"RMSE: {seg_rmse:.2f}")

        print(
            seg_df[
                [
                    LOCATION_COLUMN,
                    "datetime_hour",
                    "time_segment",
                    "orders_count",
                    "prediction",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("\n=== METRICS BY RESTAURANT SEGMENT + TIME SEGMENT ===")

    segment_time_stats = (
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

    for rest_segment in ["low", "medium", "high", "mega"]:
        seg_time_df = segment_time_stats[
            segment_time_stats["segment"] == rest_segment
        ]
        if seg_time_df.empty:
            continue

        print(f"\nRESTAURANT SEGMENT: {rest_segment}")
        print(
            seg_time_df.sort_values("time_segment")
            .to_string(index=False)
        )

        examples = (
            result_df[result_df["segment"] == rest_segment]
            .sort_values("datetime_hour")
            .groupby("time_segment", group_keys=False)
            .head(20)
        )

        print("\nEXAMPLES:")
        print(
            examples[
                [
                    LOCATION_COLUMN,
                    "datetime_hour",
                    "time_segment",
                    "orders_count",
                    "prediction",
                ]
            ]
            .to_string(index=False)
        )

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

    print("\n=== CREATE FEATURES ===")
    feature_df = create_features(hourly_df)
    feature_df, features = build_feature_matrix(feature_df, use_time_segment=True)

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
    evaluate_model(
        model,
        test_df,
        features
    )
    print_feature_importance(
        model,
        features
    )

if __name__ == "__main__":
    main()
