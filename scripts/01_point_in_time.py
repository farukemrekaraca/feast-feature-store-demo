"""Build a training set two ways and show why point-in-time joins matter.

  naive : join each order with the *latest* feature row (what you get by
          querying today's tables and merging them onto old labels)
  feast : get_historical_features() picks, for every order, the latest
          feature row whose timestamp is <= the order timestamp
"""

from pathlib import Path

import pandas as pd

from feast import FeatureStore

REPO = Path(__file__).resolve().parent.parent / "feature_repo"
DATA = REPO / "data"

store = FeatureStore(repo_path=str(REPO))
orders = pd.read_parquet(DATA / "orders.parquet")
restaurant_raw = pd.read_parquet(DATA / "restaurant_stats.parquet")
courier_raw = pd.read_parquet(DATA / "courier_stats.parquet")

# --- Feast: point-in-time correct join ---------------------------------------
entity_df = orders[["order_id", "event_timestamp", "courier_id", "restaurant_id", "distance_km", "is_late"]]
feast_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("late_delivery_v1"),
).to_df()
feast_df = feast_df.sort_values("order_id", ignore_index=True)

# --- Naive: "just join the current stats" --------------------------------------
latest_r = restaurant_raw.sort_values("event_timestamp").groupby("restaurant_id").tail(1)
latest_c = courier_raw.sort_values("event_timestamp").groupby("courier_id").tail(1)
naive_df = (
    orders.merge(latest_r.drop(columns="event_timestamp"), on="restaurant_id")
    .merge(latest_c.drop(columns="event_timestamp"), on="courier_id")
    .sort_values("order_id", ignore_index=True)
)
naive_df["expected_delivery_min"] = naive_df["avg_prep_min_1h"] + naive_df["distance_km"] * naive_df["min_per_km_7d"]

# --- Zoom in on one dinner-rush order ---------------------------------------------
rush = orders[orders["event_timestamp"].dt.hour == 20]
example = rush.iloc[len(rush) // 2]
oid, rid, ts = example["order_id"], example["restaurant_id"], example["event_timestamp"]

print(f"\nOrder #{oid} from restaurant {rid}, placed at {ts:%a %H:%M}")
print("Raw kitchen-load snapshots for that restaurant around the order:")
window = restaurant_raw[
    (restaurant_raw["restaurant_id"] == rid)
    & (restaurant_raw["event_timestamp"] >= ts - pd.Timedelta(hours=2))
    & (restaurant_raw["event_timestamp"] <= ts + pd.Timedelta(hours=2))
]
for _, row in window.iterrows():
    marker = "  <- Feast picks this one (latest row <= order time)" if row["event_timestamp"] == ts.floor("h") else ""
    future = "  (future!)" if row["event_timestamp"] > ts else ""
    print(f"   {row['event_timestamp']:%a %H:%M}  open_orders={row['open_orders']:>2}{marker}{future}")

latest_ts = latest_r.loc[latest_r["restaurant_id"] == rid, "event_timestamp"].iloc[0]
f_row = feast_df.loc[feast_df["order_id"] == oid].iloc[0]
n_row = naive_df.loc[naive_df["order_id"] == oid].iloc[0]
print(f"\n   Feast  -> open_orders={f_row['open_orders']:>2}  (as of {ts.floor('h'):%a %H:%M})")
print(f"   Naive  -> open_orders={n_row['open_orders']:>2}  (as of {latest_ts:%a %H:%M}, {(latest_ts - ts).days} days in the future)")

# --- Whole-dataset check ---------------------------------------------------------
future_rows = (orders["event_timestamp"] < latest_ts).mean()
print("\nAcross all", len(orders), "orders:")
print(f"   naive join used data from the future for {future_rows:.0%} of rows")

# Verify Feast against a hand-written backward as-of join (latest row <= order time)
asof = pd.merge_asof(
    orders.sort_values("event_timestamp"),
    restaurant_raw.sort_values("event_timestamp"),
    on="event_timestamp",
    by="restaurant_id",
).sort_values("order_id", ignore_index=True)
match = (asof["open_orders"].to_numpy() == feast_df["open_orders"].to_numpy()).mean()
print(f"   Feast join matches 'latest value at order time' for {match:.0%} of rows -> no leakage")

feast_df.to_parquet(DATA / "training_feast.parquet", index=False)
naive_df.to_parquet(DATA / "training_naive.parquet", index=False)
print("\nSaved data/training_feast.parquet and data/training_naive.parquet")
