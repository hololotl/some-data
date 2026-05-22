import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text


DB_URL = os.getenv("DB_URL", "postgresql://courier:1337@localhost:1338/coffee")
TABLE_NAME = os.getenv("ORDERS_TABLE", "orders")
TIMESTAMP_COLUMN = os.getenv("ORDERS_TIMESTAMP_COLUMN", "prepare_until")
RESTAURANT_COLUMN = os.getenv("ORDERS_RESTAURANT_COLUMN", "location_id")


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

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

def load_orders(engine, limit=None):
    query = [
        f"SELECT {RESTAURANT_COLUMN}, {TIMESTAMP_COLUMN} FROM {TABLE_NAME}",
    ]
    query.append(f"ORDER BY {TIMESTAMP_COLUMN} DESC")
    if limit:
        query.append(f"LIMIT {int(limit)}")

    sql = " ".join(query)
    return pd.read_sql_query(text(sql), engine)

def build_engine():
    return create_engine(DB_URL)

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
    ## ставим пусто где таймстам нет
    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT
    ## удаляет все строки, где в одной из этих колонок есть пустое значение NaN / NaT
    df = df.dropna(subset=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN])
    ##переводим в строку
    df = df[
        (df[TIMESTAMP_COLUMN].dt.year >= 2024) &
        (df[TIMESTAMP_COLUMN].dt.year <= 2027)
        ]
    df[RESTAURANT_COLUMN] = df[RESTAURANT_COLUMN].astype(str)

    ## добавляем признаки в df, добавляем дату час день недели день месяц
    df["date"] = df[TIMESTAMP_COLUMN].dt.date
    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["weekday"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    df["weekday_name"] = df[TIMESTAMP_COLUMN].dt.day_name()
    df["month"] = (
        df[TIMESTAMP_COLUMN]
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Order analytics dashboard for restaurant demand analysis")
    parser.add_argument("--limit", type=int, default=1000000, help="Limit rows loaded from the database")
    return parser.parse_args()

def filter_orders_by_location_order_count(data, orders_limit_min=0, orders_limit_max=None):
    # Считаем количество заказов по каждой локации
    location_order_counts = data.groupby("location_id").size()

    # Формируем условие фильтрации
    condition = location_order_counts >= orders_limit_min

    # Если верхняя граница указана — добавляем её
    if orders_limit_max is not None:
        condition &= location_order_counts <= orders_limit_max

    # Получаем подходящие location_id
    valid_locations = location_order_counts[condition].index

    # Фильтруем DataFrame
    filtered_df = data[data["location_id"].isin(valid_locations)].copy()

    # Количество подходящих локаций
    locations_count = len(valid_locations)

    return filtered_df, locations_count


def classify_location(order_count, p50, p90, p99):
    if order_count <= p50:
        return "low"

    elif order_count <= p90:
        return "medium"

    elif order_count <= p99:
        return "high"

    return "mega"

def run_analysis(limit=None):
    engine = build_engine()
    print("\n=== Loading data ===")
    df = load_orders(engine, limit=limit)
    df = prepare_data(df)
    print(len(df))
    location_stats = (
        df.groupby("location_id")
        .size()
        .reset_index(name="orders_count")
    )
    print()
    print(
        location_stats["orders_count"].describe(
            percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )

    location_stats = (
        df.groupby("location_id")
        .size()
        .reset_index(name="orders_count")
    )

    p50 = location_stats["orders_count"].quantile(0.50)
    p90 = location_stats["orders_count"].quantile(0.85)
    p99 = location_stats["orders_count"].quantile(0.95)

    print()
    location_stats["segment"] = (
        location_stats["orders_count"]
        .apply(lambda x: classify_location(x, p50, p90, p99))
    )
    print(
        location_stats.groupby("segment")["orders_count"]
        .agg(
            locations="count",
            total_orders="sum",
            avg_orders="mean",
            median_orders="median",
            max_orders="max"
        )
    )
    df = df.merge(
        location_stats[["location_id", "segment"]],
        on="location_id",
        how="left"
    )
    ## анализируем самые популярные локации
    mega_df = df[df["segment"] == "mega"]
    work_hours_df = load_work_hours_df(engine)
    # analyze_mega_locations(mega_df, work_hours_df)
    # high_df = df[df["segment"] == "high"]
    # analyze_mega_locations(high_df, work_hours_df)

    # medium_df =df[df["segment"] == "medium"]
    # analyze_mega_locations(medium_df, work_hours_df)

    low_df = df[df["segment"] == "low"]
    analyze_mega_locations(low_df, work_hours_df)


def analyze_mega_locations(df, work_hours_df):
    print("\n=== MEGA LOCATIONS ANALYSIS ===")

    # Получаем уникальные mega location_id
    mega_locations = df["location_id"].unique()

    print(f"Mega locations count: {len(mega_locations)}")
    print("Mega locations:", mega_locations)

    # Фильтруем рабочие часы только для mega локаций
    mega_work_hours = work_hours_df[
        work_hours_df["location_id"].isin(mega_locations)
    ].copy()

    # Сортировка для удобного вывода
    mega_work_hours = mega_work_hours.sort_values(
        by=["location_id", "weekday"]
    )

    print("\n=== WORK HOURS ===")
    print(mega_work_hours)
    hourly_orders = (
        df.groupby("hour")
        .size()
    )

    print(hourly_orders)

    print()
    df["time_segment"] = df.apply(
        lambda row: classify_hour_low(
            row["location_id"],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    print(
        df.groupby("time_segment")
        .size()
    )
    print("\n===  ===")
    print(
        df.groupby(["segment", "time_segment"])
        .size()
    )

    # interval_hours = {
    #     "deep_night": 6,
    #     "early_morning": 3,
    #     "morning": 2,
    #     "lunch": 4,
    #     "afternoon": 3,
    #     "dinner": 3,
    #     "late_evening": 2,
    # }
    interval_hours = {
        "first_half": 6,
        "second_half": 6,
        "third_half": 6,
        "fourth_half": 6,
    }

    stats = (
        df.groupby("time_segment")
        .size()
        .reset_index(name="orders")
    )

    stats["hours_count"] = (
        stats["time_segment"]
        .map(interval_hours)
    )

    stats["orders_per_hour"] = (
            stats["orders"]
            / stats["hours_count"]
    )

    print(stats)
    print()
    print(f'var {stats["orders_per_hour"].var()}')
    print(f'std {stats["orders_per_hour"].std()}')
    cv = (
            stats["orders_per_hour"].std()
            / stats["orders_per_hour"].mean()
    )

    print(f'cv {cv}')

    hourly_inside = (
        df.groupby(["time_segment", "hour"])
        .size()
        .reset_index(name="orders")
    )

    inside_stats = (
        hourly_inside.groupby("time_segment")["orders"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max"
        )
    )

    inside_stats["cv"] = (
            inside_stats["std"]
            / inside_stats["mean"]
    )

    print(inside_stats)

    print("\n=== INTERVAL STATS ===")
    print(stats.to_string(index=False))

    print("\n=== ORDERS PER HOUR STATS ===")
    print(f"mean: {stats['orders_per_hour'].mean():.2f}")
    print(f"std: {stats['orders_per_hour'].std():.2f}")

    cv = (
            stats["orders_per_hour"].std()
            / stats["orders_per_hour"].mean()
    )

    print(f"cv: {cv:.4f}")
    hourly = (
        df.groupby("hour")
        .size()
        .reset_index(name="orders")
    )

    hourly["pct_change"] = (
        hourly["orders"]
        .pct_change()
    )

    print("\n=== HOURLY DEMAND ===")
    print(hourly.to_string(index=False))

    print("\n=== ahui stats ===")
    df["time_segment"] = df.apply(
        lambda row: classify_hour_low(
            row["location_id"],
            row["weekday"],
            row["hour"],
            work_hours_df
        ),
        axis=1
    )
    daily_segment_stats = (
        df.groupby(["date", "time_segment"])
        .size()
        .reset_index(name="orders")
    )
    segment_stability = (
        daily_segment_stats
        .groupby("time_segment")["orders"]
        .agg(
            mean="mean",
            std="std",
            min="min",
            max="max"
        )
    )

    segment_stability["cv"] = (
            segment_stability["std"]
            / segment_stability["mean"]
    )

    print(segment_stability)

def classify_hour(location_id, weekday, hour, work_hours_df):

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id) &
        (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return "closed"

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    # -------------------------
    # Проверка открыт ли ресторан
    # -------------------------

    # Обычный случай:
    # 07 -> 22
    if start_hour < finish_hour:
        is_open = start_hour <= hour < finish_hour

    # Overnight case:
    # 20 -> 04
    else:
        is_open = (
            hour >= start_hour
            or hour < finish_hour
        )

    if not is_open:
        return "closed"

    # -------------------------
    # Time segmentation
    # -------------------------

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


def classify_hour_low(location_id, weekday, hour, work_hours_df):

    wh = work_hours_df[
        (work_hours_df["location_id"] == location_id) &
        (work_hours_df["weekday"] == weekday)
    ]

    if wh.empty:
        return "closed"

    start_hour = wh.iloc[0]["start_hour"]
    finish_hour = wh.iloc[0]["finish_hour"]

    # обычный график
    if start_hour < finish_hour:
        is_open = start_hour <= hour < finish_hour

    # overnight
    else:
        is_open = (
            hour >= start_hour
            or hour < finish_hour
        )

    if not is_open:
        return "closed"

    # -------------------------
    # BIG BUCKETS
    # -------------------------

    if 0 <= hour < 6:
        return "first_half"
    elif 6 <= hour < 12:
        return "second_half"
    elif 12 <= hour < 18:
        return "third_half"
    else:
        return "fourth_half"



if __name__ == "__main__":
    args = parse_args()
    run_analysis(limit=args.limit)

