"""Train the late-delivery classifier on both training sets.

Both models are evaluated on the *last 4 days* with point-in-time features,
because that is what the model will actually see in production.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "feature_repo" / "data"
MODEL_PATH = ROOT / "models" / "late_delivery_v1.joblib"
FEATURES = ["min_per_km_7d", "acceptance_rate", "open_orders", "avg_prep_min_1h", "distance_km", "expected_delivery_min"]

feast_df = pd.read_parquet(DATA / "training_feast.parquet").sort_values("order_id")
naive_df = pd.read_parquet(DATA / "training_naive.parquet").sort_values("order_id")

cutoff = feast_df["event_timestamp"].max() - pd.Timedelta(days=4)
train_mask = (feast_df["event_timestamp"] < cutoff).to_numpy()
# Rows outside the restaurant TTL come back as NaN; drop them from the test set
test = feast_df[~train_mask].dropna(subset=FEATURES)


def fit(train: pd.DataFrame) -> GradientBoostingClassifier:
    train = train.dropna(subset=FEATURES)
    return GradientBoostingClassifier(random_state=0).fit(train[FEATURES], train["is_late"])


def auc(model, df: pd.DataFrame) -> float:
    return roc_auc_score(df["is_late"], model.predict_proba(df[FEATURES])[:, 1])


naive_model = fit(naive_df[train_mask])
feast_model = fit(feast_df[train_mask])

print("Same algorithm, same labels, same test set. Only the join differs:")
print(f"   trained on naive join     -> test AUC {auc(naive_model, test):.3f}")
print(f"   trained on Feast PIT join -> test AUC {auc(feast_model, test):.3f}")

model = feast_model
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} (trained on Feast features)")
