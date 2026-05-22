import os

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from catboost import CatBoostRegressor

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

OUTPUT_DIR_OLD = "ahuet_analiz_segment_csv_old"
OUTPUT_DIR_LOG = "ahuet_analiz_segment_csv_log"

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

CAT_FEATURES = ["time_segment", "segment", "weekday"]

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

    df[TIMESTAMP_COLUMN] = pd.to_datetime(
        df[TIMESTAMP_COLUMN],
        unit="ms",
        errors="coerce",
        utc=True
    )

    df[TIMESTAMP_COLUMN] = (
        df[TIMESTAMP_COLUMN]
        .dt.tz_convert("Europe/Moscow")
    )

    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT

    df = df.dropna(
        subset=[
            LOCATION_COLUMN,
            TIMESTAMP_COLUMN
        ]
    )

    df = df[
        (df[TIMESTAMP_COLUMN].dt.year >= 2024)
        & (df[TIMESTAMP_COLUMN].dt.year <= 2027)
    ]

    df[LOCATION_COLUMN] = df[LOCATION_COLUMN].astype(str)

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
        is_open = hour >= start_hour or hour < finish_hour

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
# SEGMENT DATASET
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

    df["lag_56seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .shift(56)
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

    df["rolling_mean_7d"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(56)
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

    df["rolling_max_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .max()
        )
    )

    df["rolling_max_8seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(8)
            .max()
        )
    )

    df["rolling_min_3seg"] = (
        df.groupby(LOCATION_COLUMN)["open_orders"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(3)
            .min()
        )
    )

    df["trend_3seg"] = df["rolling_mean_3seg"] - df["lag_8seg"]
    df["trend_ratio"] = df["lag_1seg"] / (df["lag_8seg"] + 1)
    df["surge_ratio"] = df["lag_1seg"] / (df["rolling_mean_3seg"] + 1)

    df["segment_sin"] = np.sin(2 * np.pi * df["segment_rank"] / 8)
    df["segment_cos"] = np.cos(2 * np.pi * df["segment_rank"] / 8)

    for col in [
        "lag_1seg",
        "lag_2seg",
        "lag_8seg",
        "lag_56seg",
        "rolling_mean_3seg",
        "rolling_mean_7d",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
        "rolling_max_3seg",
        "rolling_max_8seg",
        "rolling_min_3seg",
        "trend_3seg",
        "trend_ratio",
        "surge_ratio",
    ]:
        df[col] = df[col].fillna(0)

    df = df[df["open_hours"] > 0].copy()
    df = df.drop(columns=["open_orders"])

    return df

# =========================================================
# SPLIT
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
# LOCATION STATS
# =========================================================

def add_location_stats(train_df, full_df):
    stats = (
        train_df.groupby(LOCATION_COLUMN)["orders_count"]
        .agg(
            location_mean_orders="mean",
            location_std_orders="std",
            location_max_orders="max",
        )
        .reset_index()
    )

    seg_stats = (
        train_df.groupby([LOCATION_COLUMN, "time_segment"])["orders_count"]
        .agg(
            location_seg_mean_orders="mean",
            location_seg_std_orders="std",
            location_seg_max_orders="max",
        )
        .reset_index()
    )

    seg_defaults = (
        train_df.groupby("time_segment")["orders_count"]
        .agg(
            seg_mean_orders="mean",
            seg_std_orders="std",
            seg_max_orders="max",
        )
        .to_dict()
    )

    full_df = full_df.merge(stats, on=LOCATION_COLUMN, how="left")
    full_df = full_df.merge(
        seg_stats,
        on=[LOCATION_COLUMN, "time_segment"],
        how="left"
    )

    full_df["location_mean_orders"] = full_df["location_mean_orders"].fillna(
        train_df["orders_count"].mean()
    )
    full_df["location_std_orders"] = full_df["location_std_orders"].fillna(
        train_df["orders_count"].std()
    )
    full_df["location_max_orders"] = full_df["location_max_orders"].fillna(
        train_df["orders_count"].max()
    )

    full_df["location_seg_mean_orders"] = full_df[
        "location_seg_mean_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_mean_orders"])
    )
    full_df["location_seg_std_orders"] = full_df[
        "location_seg_std_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_std_orders"])
    )
    full_df["location_seg_max_orders"] = full_df[
        "location_seg_max_orders"
    ].fillna(
        full_df["time_segment"].map(seg_defaults["seg_max_orders"])
    )

    return full_df

# =========================================================
# FEATURE MATRIX
# =========================================================

def build_feature_matrix(df):
    raw_features = [
        "weekday",
        "month",
        "is_weekend",
        "open_hours",
        "segment_rank",

        "lag_1seg",
        "lag_2seg",
        "lag_8seg",
        "lag_56seg",

        "prev_open_1seg",
        "prev_open_2seg",
        "prev_open_8seg",

        "rolling_mean_3seg",
        "rolling_mean_7d",
        "rolling_std_3seg",
        "rolling_open_count_3seg",
        "rolling_max_3seg",
        "rolling_max_8seg",
        "rolling_min_3seg",

        "trend_3seg",
        "trend_ratio",
        "surge_ratio",

        "segment_sin",
        "segment_cos",

        "location_mean_orders",
        "location_std_orders",
        "location_max_orders",

        "location_seg_mean_orders",
        "location_seg_std_orders",
        "location_seg_max_orders",
    ]

    cat_features = [f for f in CAT_FEATURES if f in df.columns]

    # Ensure categorical features are not duplicated in the numeric list.
    features = [f for f in raw_features if f not in cat_features]

    return features, cat_features

# =========================================================
# MODELS
# =========================================================

def train_catboost(train_df, features, cat_features, use_log):
    X_train = train_df[features + cat_features]
    y_train = train_df["orders_count"]

    peak_thresholds = (
        train_df.groupby("segment")["orders_count"]
        .quantile(0.97)
        .to_dict()
    )
    peak_weight = 1.8
    sample_weight = train_df["segment"].map(peak_thresholds)
    sample_weight = (
        train_df["orders_count"] >= sample_weight
    ).astype(float) * (peak_weight - 1.0) + 1.0

    if use_log:
        y_train = np.log1p(y_train)

    cat_indices = [X_train.columns.get_loc(c) for c in cat_features]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        sample_weight=sample_weight
    )

    return model

# =========================================================
# EVALUATE
# =========================================================

def evaluate_predictions(result_df):
    mae = mean_absolute_error(
        result_df["orders_count"],
        result_df["prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            result_df["orders_count"],
            result_df["prediction"]
        )
    )

    global_metrics = {
        "mae": mae,
        "rmse": rmse,
        "rows": len(result_df),
    }

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

    time_segment_mae_df = (
        result_df.groupby("time_segment")
        .apply(
            lambda g: pd.Series({
                "rows": len(g),
                "mae": mean_absolute_error(g["orders_count"], g["prediction"]),
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

    return global_metrics, segment_metrics_df, time_segment_mae_df, segment_time_metrics_df


def print_segment_time_stats(result_df, segment_time_metrics_df, label):
    print(f"\n=== SEGMENT TIME STATS: {label} ===")

    for rest_segment in ["low", "medium", "high", "mega"]:
        seg_time_df = segment_time_metrics_df[
            segment_time_metrics_df["segment"] == rest_segment
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
            .sort_values("segment_datetime")
            .head(10)
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


def predict_model(model, df, features, cat_features, use_log, clip_negative):
    X = df[features + cat_features]
    pred = model.predict(X)
    if use_log:
        pred = np.expm1(pred)
    if clip_negative:
        pred = np.clip(pred, 0, None)

    result_df = df.copy()
    result_df["prediction"] = pred
    result_df["error"] = result_df["prediction"] - result_df["orders_count"]
    result_df["abs_error"] = result_df["error"].abs()

    return result_df

# =========================================================
# PEAK METRICS
# =========================================================

def compute_peak_metrics(train_df, result_df):
    thresholds = (
        train_df.groupby("segment")["orders_count"]
        .quantile(0.95)
        .to_dict()
    )

    rows = []
    for segment, threshold in thresholds.items():
        seg_df = result_df[result_df["segment"] == segment]
        peaks = seg_df[seg_df["orders_count"] >= threshold]
        if len(peaks) == 0:
            rows.append({
                "segment": segment,
                "threshold": threshold,
                "rows": 0,
                "mae": np.nan,
                "rmse": np.nan,
            })
            continue

        rows.append({
            "segment": segment,
            "threshold": threshold,
            "rows": len(peaks),
            "mae": mean_absolute_error(peaks["orders_count"], peaks["prediction"]),
            "rmse": np.sqrt(mean_squared_error(peaks["orders_count"], peaks["prediction"])),
        })

    peak_metrics_df = pd.DataFrame(rows)

    peak_examples_df = (
        result_df
        .copy()
        .assign(peak_threshold=result_df["segment"].map(thresholds))
    )
    peak_examples_df = peak_examples_df[
        peak_examples_df["orders_count"] >= peak_examples_df["peak_threshold"]
    ]
    peak_examples_df = (
        peak_examples_df
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
            "peak_threshold",
        ]]
    )

    return peak_metrics_df, peak_examples_df

# =========================================================
# OUTPUTS
# =========================================================

def save_outputs(
    model,
    result_df,
    global_metrics,
    segment_metrics_df,
    segment_time_metrics_df,
    peak_metrics_df,
    peak_examples_df,
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

    peak_metrics_df.to_csv(
        os.path.join(output_dir, "peak_metrics.csv"),
        index=False
    )

    peak_examples_df.to_csv(
        os.path.join(output_dir, "peak_examples.csv"),
        index=False
    )

    result_df[
        [
            LOCATION_COLUMN,
            "segment_datetime",
            "segment",
            "time_segment",
            "orders_count",
            "prediction",
            "error",
            "abs_error",
        ]
    ].to_csv(
        os.path.join(output_dir, "predictions_full.csv"),
        index=False
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
    p90 = location_stats["orders_count"].quantile(0.85)
    p99 = location_stats["orders_count"].quantile(0.95)

    location_stats["segment"] = (
        location_stats["orders_count"]
        .apply(
            lambda x: classify_location(
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

def classify_location(order_count, p50, p90, p99):
    if order_count <= p50:
        return "low"

    if order_count <= p90:
        return "medium"

    if order_count <= p99:
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

    location_segments = build_location_segments(df)

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

    print(feature_df.head())

    print("\n=== SPLIT TRAIN / TEST ===")
    train_df, test_df = split_train_test(feature_df)

    print(
        len(train_df),
        len(test_df)
    )

    feature_df = add_location_stats(train_df, feature_df)
    train_df, test_df = split_train_test(feature_df)

    features, cat_features = build_feature_matrix(feature_df)

    print("\n=== TRAIN CATBOOST (OLD) ===")
    model_old = train_catboost(
        train_df,
        features,
        cat_features,
        use_log=False
    )

    print("\n=== TRAIN CATBOOST (LOG) ===")
    model_log = train_catboost(
        train_df,
        features,
        cat_features,
        use_log=True
    )

    print("\n=== EVALUATE OLD MODEL ===")
    result_old = predict_model(
        model_old,
        test_df,
        features,
        cat_features,
        use_log=False,
        clip_negative=True
    )
    (
        global_old,
        segment_old,
        time_old,
        segment_time_old,
    ) = evaluate_predictions(result_old)

    print_segment_time_stats(result_old, segment_time_old, "OLD")

    print("\n=== EVALUATE LOG MODEL ===")
    result_log = predict_model(
        model_log,
        test_df,
        features,
        cat_features,
        use_log=True,
        clip_negative=True
    )
    (
        global_log,
        segment_log,
        time_log,
        segment_time_log,
    ) = evaluate_predictions(result_log)

    print_segment_time_stats(result_log, segment_time_log, "LOG")

    train_old = predict_model(
        model_old,
        train_df,
        features,
        cat_features,
        use_log=False,
        clip_negative=True
    )
    train_log = predict_model(
        model_log,
        train_df,
        features,
        cat_features,
        use_log=True,
        clip_negative=True
    )

    train_metrics_old, _, _, _ = evaluate_predictions(train_old)
    train_metrics_log, _, _, _ = evaluate_predictions(train_log)

    old_peak_metrics, old_peak_examples = compute_peak_metrics(
        train_df,
        result_old
    )
    log_peak_metrics, log_peak_examples = compute_peak_metrics(
        train_df,
        result_log
    )

    save_outputs(
        model_old,
        result_old,
        global_old,
        segment_old,
        segment_time_old,
        old_peak_metrics,
        old_peak_examples,
        OUTPUT_DIR_OLD
    )

    save_outputs(
        model_log,
        result_log,
        global_log,
        segment_log,
        segment_time_log,
        log_peak_metrics,
        log_peak_examples,
        OUTPUT_DIR_LOG
    )

    print("\n=== COMPARISON: OVERALL ===")
    print(f"OLD MAE: {global_old['mae']:.4f} | LOG MAE: {global_log['mae']:.4f}")
    print(f"OLD RMSE: {global_old['rmse']:.4f} | LOG RMSE: {global_log['rmse']:.4f}")

    print("\n=== COMPARISON: RESTAURANT SEGMENT MAE ===")
    seg_compare = segment_old[["segment", "mae"]].merge(
        segment_log[["segment", "mae"]],
        on="segment",
        suffixes=("_old", "_log")
    )
    seg_compare["delta"] = seg_compare["mae_log"] - seg_compare["mae_old"]
    print(seg_compare.to_string(index=False))

    print("\n=== COMPARISON: TIME SEGMENT MAE ===")
    time_compare = time_old.merge(
        time_log,
        on="time_segment",
        suffixes=("_old", "_log")
    )
    time_compare["delta"] = time_compare["mae_log"] - time_compare["mae_old"]
    print(time_compare.to_string(index=False))

    lunch_old = result_old[result_old["time_segment"] == "lunch"]
    lunch_log = result_log[result_log["time_segment"] == "lunch"]
    mega_old = result_old[result_old["segment"] == "mega"]
    mega_log = result_log[result_log["segment"] == "mega"]
    mega_lunch_old = result_old[
        (result_old["segment"] == "mega")
        & (result_old["time_segment"] == "lunch")
    ]
    mega_lunch_log = result_log[
        (result_log["segment"] == "mega")
        & (result_log["time_segment"] == "lunch")
    ]

    def _mae_safe(df):
        if len(df) == 0:
            return np.nan
        return mean_absolute_error(df["orders_count"], df["prediction"])

    print("\n=== COMPARISON: LUNCH / MEGA ===")
    print(f"LUNCH MAE old/log: {_mae_safe(lunch_old):.4f} / {_mae_safe(lunch_log):.4f}")
    print(f"MEGA MAE old/log: {_mae_safe(mega_old):.4f} / {_mae_safe(mega_log):.4f}")
    print(
        f"MEGA+LUNCH MAE old/log: {_mae_safe(mega_lunch_old):.4f} / {_mae_safe(mega_lunch_log):.4f}"
    )

    print("\n=== OVERFITTING CHECK ===")
    print(
        f"OLD train/test RMSE: {train_metrics_old['rmse']:.4f} / {global_old['rmse']:.4f}"
    )
    print(
        f"LOG train/test RMSE: {train_metrics_log['rmse']:.4f} / {global_log['rmse']:.4f}"
    )

    print("\n=== PEAK METRICS (P95 BY SEGMENT) ===")
    print("OLD:")
    print(old_peak_metrics.to_string(index=False))
    print("LOG:")
    print(log_peak_metrics.to_string(index=False))

    print("\n=== SUMMARY ===")
    mae_delta = global_log["mae"] - global_old["mae"]
    rmse_delta = global_log["rmse"] - global_old["rmse"]
    if mae_delta < 0 and rmse_delta < 0:
        print("LOG model improved overall metrics.")
    elif mae_delta > 0 and rmse_delta > 0:
        print("LOG model worsened overall metrics.")
    else:
        print("LOG model is mixed: one metric improved, another worsened.")

    if train_metrics_log["rmse"] < train_metrics_old["rmse"]:
        print("LOG model has lower train RMSE (possible regularization effect).")
    else:
        print("LOG model has higher train RMSE (may underfit relative to OLD).")

    peak_improved = (log_peak_metrics["rmse"] < old_peak_metrics["rmse"]).sum()
    print(f"Peak RMSE improved in {peak_improved} segments out of {len(old_peak_metrics)}.")

if __name__ == "__main__":
    main()
