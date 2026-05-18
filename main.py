import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, inspect, text


DB_URL = os.getenv("DB_URL", "postgresql://courier:1337@localhost:1338/coffee")
TABLE_NAME = os.getenv("ORDERS_TABLE", "orders")
TIMESTAMP_COLUMN = os.getenv("ORDERS_TIMESTAMP_COLUMN", "add_timestamp")
RESTAURANT_COLUMN = os.getenv("ORDERS_RESTAURANT_COLUMN", "location_id")

OUTPUT_DIR = Path(os.getenv("ANALYSIS_OUTPUT_DIR", "analysis_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = os.getenv("SHOW_PLOTS", "0") == "1"


def build_engine():
    return create_engine(DB_URL)


def show_schema(engine):
    inspector = inspect(engine)
    print("\n=== Tables ===")
    for table_name in inspector.get_table_names():
        print(f"- {table_name}")
        columns = inspector.get_columns(table_name)
        for column in columns:
            print(f"    - {column['name']} ({column['type']})")


def load_orders(engine, limit=None, where_clause=None):
    query = [
        f"SELECT {RESTAURANT_COLUMN}, {TIMESTAMP_COLUMN} FROM {TABLE_NAME}",
    ]
    if where_clause:
        query.append(f"WHERE {where_clause}")
    query.append(f"ORDER BY {TIMESTAMP_COLUMN}")
    if limit:
        query.append(f"LIMIT {int(limit)}")

    sql = " ".join(query)
    return pd.read_sql_query(text(sql), engine)


def prepare_data(df):
    df = df.copy()
    # Convert from milliseconds to datetime
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], unit="ms", errors="coerce")
    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT
    df = df.dropna(subset=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN])
    df[RESTAURANT_COLUMN] = df[RESTAURANT_COLUMN].astype(str)

    df["date"] = df[TIMESTAMP_COLUMN].dt.date
    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["weekday"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    df["weekday_name"] = df[TIMESTAMP_COLUMN].dt.day_name()
    df["month"] = df[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)
    return df


def filter_recent_data(df, lookback_days=365):
    latest_ts = df[TIMESTAMP_COLUMN].max()
    if pd.isna(latest_ts):
        return df

    cutoff = latest_ts - pd.Timedelta(days=lookback_days)
    filtered = df[df[TIMESTAMP_COLUMN] >= cutoff].copy()
    print(f"\n=== Time filter ===")
    print(f"Latest timestamp: {latest_ts}")
    print(f"Cutoff ({lookback_days} days back): {cutoff}")
    print(f"Rows after filter: {len(filtered):,} / {len(df):,}")
    return filtered


def basic_summary(df):
    print("\n=== Basic summary ===")
    print(f"Rows: {len(df):,}")
    print(f"Restaurants: {df[RESTAURANT_COLUMN].nunique():,}")
    print(f"Date range: {df[TIMESTAMP_COLUMN].min()} -> {df[TIMESTAMP_COLUMN].max()}")
    print("\nMissing values:")
    print(df[[RESTAURANT_COLUMN, TIMESTAMP_COLUMN]].isna().sum())


def plot_and_save(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")


def plot_overall_patterns(df):
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = df.groupby("weekday_name").size().reindex(weekday_order, fill_value=0)
    monthly = df.groupby("month").size()

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].plot(hourly.index, hourly.values, marker="o")
    axes[0].set_title("Orders by hour of day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Orders")
    axes[0].grid(alpha=0.3)

    axes[1].bar(weekday.index, weekday.values)
    axes[1].set_title("Orders by weekday")
    axes[1].set_xlabel("Weekday")
    axes[1].set_ylabel("Orders")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].plot(monthly.index.astype(str), monthly.values, marker="o")
    axes[2].set_title("Orders by month")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("Orders")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(alpha=0.3)

    plot_and_save(fig, "overall_patterns.png")


def plot_weekday_hour_heatmap(df):
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = (
        df.groupby(["weekday_name", "hour"]).size().reset_index(name="orders")
        .pivot(index="weekday_name", columns="hour", values="orders")
        .reindex(weekday_order)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_title("Heatmap: weekday x hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Weekday")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, label="Orders")

    plot_and_save(fig, "weekday_hour_heatmap.png")


def plot_restaurant_distribution(df, top_n=20):
    counts = df.groupby(RESTAURANT_COLUMN).size().sort_values(ascending=False)
    top = counts.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(top.index[::-1], top.values[::-1])
    axes[0].set_title(f"Top {top_n} restaurants by orders")
    axes[0].set_xlabel("Orders")

    axes[1].hist(counts.values, bins=30, color="steelblue", edgecolor="white")
    axes[1].set_title("Distribution of orders per restaurant")
    axes[1].set_xlabel("Orders per restaurant")
    axes[1].set_ylabel("Restaurants")

    plot_and_save(fig, "restaurant_distribution.png")


def plot_top_restaurant_profiles(df, top_n=8):
    top_restaurants = df.groupby(RESTAURANT_COLUMN).size().sort_values(ascending=False).head(top_n).index
    subset = df[df[RESTAURANT_COLUMN].isin(top_restaurants)].copy()

    profiles = (
        subset.groupby([RESTAURANT_COLUMN, "hour"]).size().reset_index(name="orders")
    )

    fig, axes = plt.subplots(top_n, 1, figsize=(14, 3 * top_n), sharex=True)
    if top_n == 1:
        axes = [axes]

    for ax, restaurant in zip(axes, top_restaurants):
        rest_data = profiles[profiles[RESTAURANT_COLUMN] == restaurant][["hour", "orders"]].set_index("hour")
        temp = rest_data.reindex(range(24), fill_value=0)
        ax.plot(temp.index, temp["orders"].values, marker="o")
        ax.set_title(f"Restaurant {restaurant}: hourly profile")
        ax.set_ylabel("Orders")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Hour")
    plot_and_save(fig, "top_restaurant_profiles.png")


def calculate_granularity_metrics(df, intervals=("15min", "30min", "1h")):
    results = []
    restaurant_ids = df[RESTAURANT_COLUMN].dropna().astype(str).unique()

    for freq in intervals:
        start = df[TIMESTAMP_COLUMN].min().floor(freq)
        end = df[TIMESTAMP_COLUMN].max().ceil(freq)
        periods = pd.date_range(start=start, end=end, freq=freq)

        grouped = (
            df.groupby([RESTAURANT_COLUMN, pd.Grouper(key=TIMESTAMP_COLUMN, freq=freq)])
            .size()
            .rename("orders")
            .reset_index()
        )

        full_index = pd.MultiIndex.from_product(
            [restaurant_ids, periods], names=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN]
        )
        completed = (
            grouped.set_index([RESTAURANT_COLUMN, TIMESTAMP_COLUMN])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )

        slot_counts = completed["orders"]
        mean_orders = slot_counts.mean()
        std_orders = slot_counts.std(ddof=0)

        results.append(
            {
                "interval": freq,
                "slots": len(slot_counts),
                "mean_orders_per_slot": mean_orders,
                "std_orders_per_slot": std_orders,
                "var_orders_per_slot": slot_counts.var(ddof=0),
                "cv": std_orders / mean_orders if mean_orders else 0,
                "zero_share": (slot_counts == 0).mean(),
                "p50": slot_counts.median(),
                "p90": slot_counts.quantile(0.9),
                "p99": slot_counts.quantile(0.99),
            }
        )

    return pd.DataFrame(results)


def plot_granularity_metrics(metrics_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(metrics_df["interval"], metrics_df["mean_orders_per_slot"], color="steelblue")
    axes[0, 0].set_title("Mean orders per slot")

    axes[0, 1].bar(metrics_df["interval"], metrics_df["std_orders_per_slot"], color="darkorange")
    axes[0, 1].set_title("Std dev per slot")

    axes[1, 0].bar(metrics_df["interval"], metrics_df["cv"], color="seagreen")
    axes[1, 0].set_title("Coefficient of variation")

    axes[1, 1].bar(metrics_df["interval"], metrics_df["zero_share"], color="crimson")
    axes[1, 1].set_title("Zero-share")

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.3)

    plot_and_save(fig, "granularity_metrics.png")


def daily_trend(df):
    daily = df.groupby(df[TIMESTAMP_COLUMN].dt.date).size().reset_index(name="orders")
    daily["date"] = pd.to_datetime(daily[TIMESTAMP_COLUMN])
    daily = daily.sort_values("date")
    daily["rolling_7d"] = daily["orders"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["orders"], alpha=0.35, label="Daily orders")
    ax.plot(daily["date"], daily["rolling_7d"], linewidth=2.5, label="7-day rolling mean")
    ax.set_title("Daily trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders")
    ax.legend()
    ax.grid(alpha=0.3)

    plot_and_save(fig, "daily_trend.png")




def export_metrics(metrics_df):
    path = OUTPUT_DIR / "granularity_metrics.csv"
    metrics_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_analysis(limit=None, where_clause=None, lookback_days=365):
    engine = build_engine()

    print("\n=== Schema preview ===")
    show_schema(engine)

    print("\n=== Loading data ===")
    df = load_orders(engine, limit=limit, where_clause=where_clause)
    df = prepare_data(df)
    df = filter_recent_data(df, lookback_days=lookback_days)

    basic_summary(df)

    metrics_df = calculate_granularity_metrics(df)
    print("\n=== Granularity comparison ===")
    print(metrics_df.to_string(index=False))
    export_metrics(metrics_df)

    plot_overall_patterns(df)
    plot_weekday_hour_heatmap(df)
    plot_restaurant_distribution(df)
    plot_top_restaurant_profiles(df)
    plot_granularity_metrics(metrics_df)
    daily_trend(df)


def parse_args():
    parser = argparse.ArgumentParser(description="Order analytics dashboard for restaurant demand analysis")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows loaded from the database")
    parser.add_argument("--where", type=str, default=None, help="Optional SQL WHERE clause")
    parser.add_argument("--lookback-days", type=int, default=365, help="Analyze only the most recent N days")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(limit=args.limit, where_clause=args.where, lookback_days=args.lookback_days)

