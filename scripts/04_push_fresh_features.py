"""Simulate a streaming update: a quiet restaurant's kitchen suddenly gets slammed.

A stream processor (Kafka/Flink/Spark) would call store.push(); no
materialization job, no redeploy. The very next request sees the new value.
"""

from datetime import datetime, timezone

import pandas as pd

from serving import score, show, store

# Find the calmest kitchen right now (one online lookup for all restaurants)
restaurant_ids = list(range(1001, 1026))
load = store.get_online_features(
    features=["restaurant_stats:avg_prep_min_1h"],
    entity_rows=[{"restaurant_id": rid} for rid in restaurant_ids],
).to_df()
calmest = int(load.sort_values("avg_prep_min_1h").iloc[0]["restaurant_id"])

order = {"courier_id": 7, "restaurant_id": calmest, "distance_km": 2.5}

print(f"Before the rush (restaurant {calmest} is the quietest kitchen right now):")
show([score(order)])

store.push(
    "restaurant_stats_push",
    pd.DataFrame(
        {
            "restaurant_id": [calmest],
            "event_timestamp": [datetime.now(timezone.utc)],
            "open_orders": [15],
            "avg_prep_min_1h": [40.0],
        }
    ),
)

print(f"\n>>> pushed: restaurant {calmest} now has 15 open orders, 40 min prep time\n")
print("After the push (same order, next request):")
show([score(order)])
