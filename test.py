from datetime import datetime

week_ends = [
    "2025-09-14",
    "2025-09-21",
    "2025-09-28",
    "2025-10-05",
    "2025-10-12",
    "2025-10-19",
    "2025-10-26",
    "2025-11-02",
    "2026-05-17",
    "2026-05-24",
    "2026-05-31",
    "2026-06-07",
    "2026-06-14",
    "2026-06-21",
]

for date_str in week_ends:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    print(f"{date_str} -> {date_obj.strftime('%A')}")