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
    query.append(f"ORDER BY {TIMESTAMP_COLUMN} DESC")
    if limit:
        query.append(f"LIMIT {int(limit)}")

    sql = " ".join(query)
    return pd.read_sql_query(text(sql), engine)

def prepare_data(df):
    df = df.copy()
    #
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], unit="ms", errors="coerce")
    ## ставим пусто где таймстам нет
    df.loc[df[TIMESTAMP_COLUMN].isna(), TIMESTAMP_COLUMN] = pd.NaT
    ## удаляет все строки, где в одной из этих колонок есть пустое значение NaN / NaT
    df = df.dropna(subset=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN])
    ##переводим в строку
    df[RESTAURANT_COLUMN] = df[RESTAURANT_COLUMN].astype(str)

    ## добавляем признаки в df, добавляем дату час день недели день месяц
    df["date"] = df[TIMESTAMP_COLUMN].dt.date
    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["weekday"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    df["weekday_name"] = df[TIMESTAMP_COLUMN].dt.day_name()
    df["month"] = df[TIMESTAMP_COLUMN].dt.to_period("M").astype(str)
    return df

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

def calculate_granularity_metrics(df, intervals=("15min", "30min", "1h", "2h", "3h")):
    results = []
    restaurant_ids = df[RESTAURANT_COLUMN].dropna().astype(str).unique()

    for freq in intervals:
        start = df[TIMESTAMP_COLUMN].min().floor(freq)
        end = df[TIMESTAMP_COLUMN].max().ceil(freq)
        periods = pd.date_range(start=start, end=end, freq=freq)

        ## групируем по типо локация 1 - 10:00-11.00 - 10 заказов
        grouped = (
            df.groupby([RESTAURANT_COLUMN, pd.Grouper(key=TIMESTAMP_COLUMN, freq=freq)])
            .size()
            .rename("orders")
            .reset_index()
        )
        ## строим все пары локация - время, это нужно так как в grouped например в 11.00 могло быть 0 заказов и там этого времени нет
        full_index = pd.MultiIndex.from_product(
            [restaurant_ids, periods], names=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN]
        )
        ## как раз заполняем тут grouped пустыми нулевыми значеями
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


def calculate_granularity_metrics2(df, intervals=("15min", "30min", "1h", "2h", "3h"), min_orders=None):
    """
    Считает те же метрики, что и `calculate_granularity_metrics`, но учитывает
    индивидуальный рабочий диапазон часов для каждого ресторана.

    Параметры:
    - df: dataframe с заказами
    - intervals: кортеж интервалов для агрегации
    - min_orders: если задано, фильтруют локации с минимум min_orders заказов

    Рабочие часы ресторана определяются по данным заказов:
    - min_hour: минимальный час, когда был заказ
    - max_hour: максимальный час, когда был заказ

    В расчёт попадают только слоты, которые попадают в [min_hour, max_hour]
    для конкретного ресторана.
    """

    data = df.copy()

    # Фильтруем по минимальному числу заказов, если задано
    if min_orders is not None:
        orders_per_rest = data.groupby(RESTAURANT_COLUMN).size()
        valid_rests = orders_per_rest[orders_per_rest >= min_orders].index
        data = data[data[RESTAURANT_COLUMN].isin(valid_rests)]
        if data.empty:
            return pd.DataFrame(
                {
                    "interval": list(intervals),
                    "slots": [0] * len(intervals),
                    "mean_orders_per_slot": [0.0] * len(intervals),
                    "std_orders_per_slot": [0.0] * len(intervals),
                    "var_orders_per_slot": [0.0] * len(intervals),
                    "cv": [0.0] * len(intervals),
                    "zero_share": [0.0] * len(intervals),
                    "p50": [0.0] * len(intervals),
                    "p90": [0.0] * len(intervals),
                    "p99": [0.0] * len(intervals),
                }
            )

    # список локаций (строки) — нужен всегда
    restaurant_ids = data[RESTAURANT_COLUMN].dropna().astype(str).unique().tolist()

    # Попытаемся прочитать расписания из таблицы work_hours в базе
    use_db_work_hours = True
    try:
        engine = build_engine()
        wh_sql = "SELECT location_id, weekday, working, start_hour, start_minutes, finish_hour, finish_minutes FROM work_hours where working = true"
        work_hours_df = pd.read_sql_query(text(wh_sql), engine)
        work_hours_df["location_id"] = work_hours_df["location_id"].astype(str)
    except Exception:
        use_db_work_hours = False
        work_hours_df = None

    if use_db_work_hours and work_hours_df is not None and not work_hours_df.empty:
        # Печатаем расписания для топ-10 локаций по числу заказов для валидации
        orders_counts = data.groupby(RESTAURANT_COLUMN).size().rename("orders").reset_index()
        top_locs = orders_counts.sort_values("orders", ascending=False).head(10)[RESTAURANT_COLUMN].tolist()
        bottom_locs = orders_counts.sort_values("orders", ascending=True).head(10)[RESTAURANT_COLUMN].tolist()
        print("\n=== work_hours from DB (top 10 locations) ===")
        for loc in top_locs:
            rows = work_hours_df[work_hours_df["location_id"] == str(loc)]
            if rows.empty:
                print(f"location={loc} | no schedule in work_hours table")
                continue
            sched = []
            for wd in range(7):
                r = rows[rows["weekday"] == wd]
                if r.empty or not bool(r.iloc[0]["working"]):
                    sched.append("closed")
                else:
                    r0 = r.iloc[0]
                    sh = int(r0["start_hour"]) if not pd.isna(r0["start_hour"]) else 0
                    sm = int(r0["start_minutes"]) if not pd.isna(r0["start_minutes"]) else 0
                    fh = int(r0["finish_hour"]) if not pd.isna(r0["finish_hour"]) else 0
                    fm = int(r0["finish_minutes"]) if not pd.isna(r0["finish_minutes"]) else 0
                    sched.append(f"{sh:02d}:{sm:02d}-{fh:02d}:{fm:02d}")
            print(f"location={loc} | " + ", ".join(sched) + f" | orders={int(orders_counts[orders_counts[RESTAURANT_COLUMN]==loc]['orders'].iloc[0])}")
        print("\n=== work_hours from DB (bottom 10 locations) ===")
        for loc in bottom_locs:
            rows = work_hours_df[work_hours_df["location_id"] == str(loc)]
            if rows.empty:
                print(f"location={loc} | no schedule in work_hours table")
                continue
            sched = []
            for wd in range(7):
                r = rows[rows["weekday"] == wd]
                if r.empty or not bool(r.iloc[0]["working"]):
                    sched.append("closed")
                else:
                    r0 = r.iloc[0]
                    sh = int(r0["start_hour"]) if not pd.isna(r0["start_hour"]) else 0
                    sm = int(r0["start_minutes"]) if not pd.isna(r0["start_minutes"]) else 0
                    fh = int(r0["finish_hour"]) if not pd.isna(r0["finish_hour"]) else 0
                    fm = int(r0["finish_minutes"]) if not pd.isna(r0["finish_minutes"]) else 0
                    sched.append(f"{sh:02d}:{sm:02d}-{fh:02d}:{fm:02d}")
            print(f"location={loc} | " + ", ".join(
                sched) + f" | orders={int(orders_counts[orders_counts[RESTAURANT_COLUMN] == loc]['orders'].iloc[0])}")

        # restaurant_ids уже определён выше
    restaurant_ids_df = pd.DataFrame({RESTAURANT_COLUMN: restaurant_ids})

    results = []
    for freq in intervals:
        start = data[TIMESTAMP_COLUMN].min().floor(freq)
        end = data[TIMESTAMP_COLUMN].max().ceil(freq)
        periods = pd.date_range(start=start, end=end, freq=freq)

        grouped = (
            data.groupby([RESTAURANT_COLUMN, pd.Grouper(key=TIMESTAMP_COLUMN, freq=freq)])
            .size()
            .rename("orders")
            .reset_index()
        )

        periods_df = pd.DataFrame({TIMESTAMP_COLUMN: periods})
        periods_df["slot_hour"] = periods_df[TIMESTAMP_COLUMN].dt.hour
        periods_df["slot_minute"] = (
            periods_df[TIMESTAMP_COLUMN].dt.hour * 60 + periods_df[TIMESTAMP_COLUMN].dt.minute
        )
        periods_df["slot_weekday"] = periods_df[TIMESTAMP_COLUMN].dt.dayofweek

        candidate_slots = restaurant_ids_df.merge(periods_df, how="cross")

        if use_db_work_hours and work_hours_df is not None and not work_hours_df.empty:
            # соединяем с расписаниями по location_id и weekday
            cs = candidate_slots.merge(
                work_hours_df,
                left_on=[RESTAURANT_COLUMN, "slot_weekday"],
                right_on=["location_id", "weekday"],
                how="left",
            )

            # фильтр: рабочий день и слот попадает в интервал (учитываем ночные смены)
            def slot_in_range(row):
                if pd.isna(row.get("working")) or not bool(row.get("working")):
                    return False
                start_min = int(row.get("start_hour", 0)) * 60 + int(row.get("start_minutes", 0))
                finish_min = int(row.get("finish_hour", 0)) * 60 + int(row.get("finish_minutes", 0))
                sm = int(row.get("slot_minute", 0))
                if start_min <= finish_min:
                    return (sm >= start_min) and (sm <= finish_min)
                else:
                    # overnight: e.g., 20:00 - 03:00
                    return (sm >= start_min) or (sm <= finish_min)

            cs["in_work"] = cs.apply(slot_in_range, axis=1)
            working_slots = cs[cs["in_work"]][[RESTAURANT_COLUMN, TIMESTAMP_COLUMN]]
        else:
            # fallback: используем минимальный/максимальный час заказов как ранее
            restaurant_hours = (
                data.groupby(RESTAURANT_COLUMN)["hour"]
                .agg(min_hour="min", max_hour="max")
                .reset_index()
            )
            cs = candidate_slots.merge(restaurant_hours, on=RESTAURANT_COLUMN, how="left")
            working_slots = cs[(cs["slot_hour"] >= cs["min_hour"]) & (cs["slot_hour"] <= cs["max_hour"])][[RESTAURANT_COLUMN, TIMESTAMP_COLUMN]]

        completed = (
            working_slots
            .merge(grouped, on=[RESTAURANT_COLUMN, TIMESTAMP_COLUMN], how="left")
            .fillna({"orders": 0})
        )
        completed["orders"] = completed["orders"].astype(int)

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

def export_metrics(metrics_df, suffix=""):
    if suffix:
        path = OUTPUT_DIR / f"granularity_metrics_{suffix}.csv"
    else:
        path = OUTPUT_DIR / "granularity_metrics.csv"
    metrics_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def run_analysis(limit=None, where_clause=None):
    engine = build_engine()

    print("\n=== Schema preview ===")
    show_schema(engine)

    print("\n=== Loading data ===")
    df = load_orders(engine, limit=limit, where_clause=where_clause)
    df = prepare_data(df)
    print(len(df))


    ## считаем базовую инфу
    basic_summary(df)

    ## подготовливаем метрики для всех локаций
    metrics_df = calculate_granularity_metrics2(df)

    print("\n=== Granularity comparison (all locations) ===")
    print(metrics_df.to_string(index=False))
    export_metrics(metrics_df)

    ## подготовливаем метрики только для локаций с >= 5000 заказов
    metrics_df_filtered = calculate_granularity_metrics2(df, min_orders=5000)
    print("\n=== Granularity comparison (locations with >= 5000 orders) ===")
    print(metrics_df_filtered.to_string(index=False))
    export_metrics(metrics_df_filtered, suffix="5k_plus")

    plot_overall_patterns(df)
    plot_weekday_hour_heatmap(df)
    plot_restaurant_distribution(df)
    plot_top_restaurant_profiles(df)
    plot_granularity_metrics(metrics_df)
    daily_trend(df)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Order analytics dashboard for restaurant demand analysis")
    parser.add_argument("--limit", type=int, default=1000000, help="Limit rows loaded from the database")
    parser.add_argument("--where", type=str, default=None, help="Optional SQL WHERE clause")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(limit=args.limit, where_clause=args.where)