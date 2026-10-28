"""Shared helper: score an order using features from the online store."""

import time
from pathlib import Path

import joblib
import pandas as pd

from feast import FeatureStore

ROOT = Path(__file__).resolve().parent.parent
store = FeatureStore(repo_path=str(ROOT / "feature_repo"))
bundle = joblib.load(ROOT / "models" / "late_delivery_v1.joblib")


def score(order: dict) -> dict:
    """order = {"courier_id": .., "restaurant_id": .., "distance_km": ..}"""
    start = time.perf_counter()
    features = store.get_online_features(
        features=store.get_feature_service("late_delivery_v1"),  # same list the model was trained on
        entity_rows=[order],
    ).to_dict()
    latency_ms = (time.perf_counter() - start) * 1000

    # Request-time inputs (distance_km) come from the request itself, the rest from Feast
    values = {**{k: [v] for k, v in order.items()}, **features}
    row = pd.DataFrame({name: values[name] for name in bundle["features"]})
    p_late = bundle["model"].predict_proba(row)[0, 1]
    return {
        **order,
        "open_orders": features["open_orders"][0],
        "avg_prep_min_1h": round(features["avg_prep_min_1h"][0], 1),
        "expected_delivery_min": round(features["expected_delivery_min"][0], 1),
        "p_late": round(p_late, 2),
        "lookup_ms": round(latency_ms, 1),
    }


def show(rows: list[dict]) -> None:
    print(pd.DataFrame(rows).to_string(index=False))
