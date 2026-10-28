"""Generate a synthetic food-delivery dataset.

Writes three parquet files into feature_repo/data/:
  restaurant_stats.parquet  hourly snapshots per restaurant (kitchen load)
  courier_stats.parquet     hourly snapshots per courier (speed, acceptance)
  orders.parquet            historical orders with the label `is_late`

The world has two things that change over time, which is what makes
point-in-time correctness matter:
  * restaurants get slammed at lunch (12-14) and dinner (19-21)
  * couriers slowly get faster or slower over the two weeks
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "feature_repo" / "data"
N_RESTAURANTS = 25
N_COURIERS = 40
N_DAYS = 14
N_ORDERS = 8000
PROMISED_MINUTES = 35
SEED = 42

rng = np.random.default_rng(SEED)

# The data ends at the current hour so the "latest" values are fresh when we
# materialize them into the online store later.
end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
start = end - timedelta(days=N_DAYS)
hours = pd.date_range(start, end, freq="h", inclusive="left")


def rush_factor(hour_of_day: np.ndarray) -> np.ndarray:
    lunch = np.exp(-((hour_of_day - 13) ** 2) / 2.0)
    dinner = np.exp(-((hour_of_day - 20) ** 2) / 2.0)
    night = (hour_of_day < 10) | (hour_of_day > 23)
    return np.where(night, 0.1, 0.4 + 1.6 * lunch + 2.0 * dinner)


# --- Restaurants: hourly kitchen load ------------------------------------------
restaurant_ids = np.arange(1001, 1001 + N_RESTAURANTS)
base_prep = rng.uniform(8, 16, N_RESTAURANTS)
base_load = rng.uniform(2, 6, N_RESTAURANTS)

rows = []
for i, rid in enumerate(restaurant_ids):
    rush = rush_factor(hours.hour.to_numpy())
    open_orders = rng.poisson(base_load[i] * rush)
    avg_prep = base_prep[i] + 2.0 * open_orders + rng.normal(0, 1.0, len(hours))
    rows.append(
        pd.DataFrame(
            {
                "restaurant_id": rid,
                "event_timestamp": hours,
                "open_orders": open_orders.astype("int64"),
                "avg_prep_min_1h": np.clip(avg_prep, 3, None).astype("float32"),
            }
        )
    )
restaurant_stats = pd.concat(rows, ignore_index=True)

# --- Couriers: hourly speed estimate that drifts over the two weeks -------------
courier_ids = np.arange(1, 1 + N_COURIERS)
speed_start = rng.uniform(2.0, 4.0, N_COURIERS)  # minutes per km
speed_drift = rng.uniform(-2.0, 2.0, N_COURIERS)  # change over the whole period
acceptance = rng.uniform(0.6, 0.99, N_COURIERS)
progress = (hours - start) / (end - start)

rows = []
for i, cid in enumerate(courier_ids):
    true_speed = speed_start[i] + speed_drift[i] * progress.to_numpy()
    rows.append(
        pd.DataFrame(
            {
                "courier_id": cid,
                "event_timestamp": hours,
                "min_per_km_7d": (true_speed + rng.normal(0, 0.1, len(hours))).astype("float32"),
                "acceptance_rate": np.clip(acceptance[i] + rng.normal(0, 0.02, len(hours)), 0, 1).astype("float32"),
            }
        )
    )
courier_stats = pd.concat(rows, ignore_index=True)

# --- Orders: sampled more often during rush hours --------------------------------
weights = rush_factor(hours.hour.to_numpy())
order_hours = rng.choice(len(hours), size=N_ORDERS, p=weights / weights.sum())
order_ts = hours[order_hours] + pd.to_timedelta(rng.uniform(0, 3600, N_ORDERS), unit="s")

orders = pd.DataFrame(
    {
        "order_id": np.arange(N_ORDERS),
        "event_timestamp": order_ts,
        "courier_id": rng.choice(courier_ids, N_ORDERS),
        "restaurant_id": rng.choice(restaurant_ids, N_ORDERS),
        "distance_km": rng.uniform(1.0, 5.0, N_ORDERS).round(2),
    }
).sort_values("event_timestamp", ignore_index=True)

# The label is decided by what the world looked like *at order time*.
r_now = pd.merge_asof(orders, restaurant_stats.sort_values("event_timestamp"), on="event_timestamp", by="restaurant_id")
c_now = pd.merge_asof(orders, courier_stats.sort_values("event_timestamp"), on="event_timestamp", by="courier_id")
delivery_min = (
    r_now["avg_prep_min_1h"]
    + orders["distance_km"] * c_now["min_per_km_7d"]
    + rng.normal(0, 4, N_ORDERS)
)
orders["is_late"] = (delivery_min > PROMISED_MINUTES).astype("int64")

DATA_DIR.mkdir(parents=True, exist_ok=True)
restaurant_stats.to_parquet(DATA_DIR / "restaurant_stats.parquet", index=False)
courier_stats.to_parquet(DATA_DIR / "courier_stats.parquet", index=False)
orders.to_parquet(DATA_DIR / "orders.parquet", index=False)

print(f"Time range       : {start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} UTC")
print(f"restaurant_stats : {len(restaurant_stats):>6} rows ({N_RESTAURANTS} restaurants x {len(hours)} hours)")
print(f"courier_stats    : {len(courier_stats):>6} rows ({N_COURIERS} couriers x {len(hours)} hours)")
print(f"orders           : {len(orders):>6} rows, {orders['is_late'].mean():.0%} late")
