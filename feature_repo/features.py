"""Feature definitions for the food-delivery project.

`feast apply` reads this file and registers everything in it.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd

from feast import Entity, FeatureService, FeatureView, Field, FileSource, PushSource, RequestSource, ValueType
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

DATA = Path(__file__).resolve().parent / "data"

# 1. Entities: the "things" features are about (the join keys)
courier = Entity(name="courier", join_keys=["courier_id"], value_type=ValueType.INT64)
restaurant = Entity(name="restaurant", join_keys=["restaurant_id"], value_type=ValueType.INT64)

# 2. Data sources: where the raw feature values already live
courier_source = FileSource(
    name="courier_stats_source",
    path=str(DATA / "courier_stats.parquet"),
    timestamp_field="event_timestamp",
)
restaurant_source = FileSource(
    name="restaurant_stats_source",
    path=str(DATA / "restaurant_stats.parquet"),
    timestamp_field="event_timestamp",
)
# A push source lets a streaming job write fresh values straight to the online store
restaurant_push = PushSource(name="restaurant_stats_push", batch_source=restaurant_source)

# 3. Feature views: a named, typed group of features from one source
courier_stats = FeatureView(
    name="courier_stats",
    entities=[courier],
    ttl=timedelta(days=1),
    schema=[
        Field(name="min_per_km_7d", dtype=Float32, description="Avg minutes per km, last 7 days"),
        Field(name="acceptance_rate", dtype=Float32, description="Share of offered orders accepted"),
    ],
    source=courier_source,
    online=True,
    tags={"team": "logistics"},
)

restaurant_stats = FeatureView(
    name="restaurant_stats",
    entities=[restaurant],
    ttl=timedelta(hours=2),  # kitchen load older than 2h is useless -> treated as missing
    schema=[
        Field(name="open_orders", dtype=Int64, description="Orders currently in the kitchen"),
        Field(name="avg_prep_min_1h", dtype=Float32, description="Avg prep time, last hour"),
    ],
    source=restaurant_push,
    online=True,
    tags={"team": "restaurants"},
)

# 4. On-demand feature view: computed at request time, identically offline and online
order_request = RequestSource(name="order_request", schema=[Field(name="distance_km", dtype=Float64)])


@on_demand_feature_view(
    sources=[courier_stats, restaurant_stats, order_request],
    schema=[Field(name="expected_delivery_min", dtype=Float64)],
)
def order_eta(inputs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["expected_delivery_min"] = inputs["avg_prep_min_1h"] + inputs["distance_km"] * inputs["min_per_km_7d"]
    return out


# 5. Feature service: the exact feature list one model version consumes
late_delivery_v1 = FeatureService(
    name="late_delivery_v1",
    features=[courier_stats, restaurant_stats, order_eta],
    description="Features for the late-delivery classifier",
)
