"""Load the latest feature values into the online store, then score live orders."""

from datetime import datetime, timezone

from serving import score, show, store

# Offline store (parquet) -> online store (SQLite): only the latest value per entity
store.materialize_incremental(end_date=datetime.now(timezone.utc))

incoming_orders = [
    {"courier_id": 7, "restaurant_id": 1005, "distance_km": 2.5},
    {"courier_id": 12, "restaurant_id": 1013, "distance_km": 4.5},
    {"courier_id": 30, "restaurant_id": 1021, "distance_km": 1.5},
]
print("\nIncoming orders scored with online features:")
show([score(order) for order in incoming_orders])
