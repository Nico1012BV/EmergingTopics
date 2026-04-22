"""
05_test_predictions.py
-----------------------
Loads test.csv, applies identical feature engineering to 01_feature_engineering.py
(minus Burnoff-dependent features), loads the best trained model by Val-RMSE,
generates predictions on the full test set, and saves a Kaggle-ready submission.

Outputs:
  - data/test_engineered.csv   (feature-engineered test set)
  - data/submission.csv        (FlightID, Burnoff — Kaggle submission format)

Run: python 05_test_predictions.py
(Must run after 02_model_training.py)
"""

import math
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from airport_coords import get_coords

# ── helpers (mirrors 01_feature_engineering.py exactly) ──────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def route_distance_km(route: str):
    try:
        org, dst = route.split("-")
        c1, c2 = get_coords(org), get_coords(dst)
        if c1 and c2:
            return haversine_km(*c1, *c2)
    except Exception:
        pass
    return None


def month_to_season(m):
    if m in [12, 1, 2]:  return "Winter"
    elif m in [3, 4, 5]: return "Spring"
    elif m in [6, 7, 8]: return "Summer"
    else:                return "Fall"


# ── 1. Locate test.csv ────────────────────────────────────────────────────────
candidates = [
    BASE_DIR / "test" / "test.csv",
    BASE_DIR / "test.csv",
    DATA_DIR / "test.csv",
]
test_path = next((p for p in candidates if p.exists()), None)
if test_path is None:
    raise FileNotFoundError(
        "test.csv not found. Checked:\n" + "\n".join(f"  {p}" for p in candidates)
    )
print(f"Loading {test_path} …")
df = pd.read_csv(test_path)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ── 2. Feature engineering (mirrors 01, skipping Burnoff-derived features) ───
df["DepartureScheduled"] = pd.to_datetime(df["DepartureScheduled"])
df["ArrivalScheduled"]   = pd.to_datetime(df["ArrivalScheduled"])

df["DepartureHour"]      = df["DepartureScheduled"].dt.hour
df["DepartureMonth"]     = df["DepartureScheduled"].dt.month
df["DepartureYear"]      = df["DepartureScheduled"].dt.year
df["DepartureDayOfWeek"] = df["DepartureScheduled"].dt.dayofweek
df["Season"]             = df["DepartureMonth"].map(month_to_season)

df["Origin"]      = df["ScheduledRoute"].str.split("-").str[0]
df["Destination"] = df["ScheduledRoute"].str.split("-").str[1]

print("  Computing great-circle distances …")
df["RouteDistanceKm"] = df["ScheduledRoute"].apply(route_distance_km)
SPEED_KMH = 750
missing_mask = df["RouteDistanceKm"].isna()
df.loc[missing_mask, "RouteDistanceKm"] = (
    df.loc[missing_mask, "BlockTimeScheduled"] / 60 * SPEED_KMH
)
print(f"  Distance imputed for {missing_mask.sum()} routes via BlockTime fallback")

df["TotalPassengers"] = df["Adults"] + df["Children"] + df["Infants"]
df["LoadFactor"]      = (df["TotalPassengers"] / df["AircraftCapacity"]).clip(0, 1)
df["FreightPerKg"]    = df["Freight"].fillna(0)

df["WeightDelta"] = df["PlannedTOW"] - df["PlannedZeroFuelWeight"]

type_map = {"NG": 1, "Max": 2, "Airbus": 1.5}
df["TypeWeightInteraction"]   = (df["AircraftTypeGroup"].map(type_map).fillna(1)
                                  * df["PlannedTOW"])
df["TypeTripTimeInteraction"] = (df["AircraftTypeGroup"].map(type_map).fillna(1)
                                  * df["PlannedTripTime"])
df["BaselineDelta"] = df["TeledyneRampWeight"] - df["PlannedTOW"]

df.to_csv(DATA_DIR / "test_engineered.csv", index=False)
print(f"Saved data/test_engineered.csv  ({len(df)} rows, {len(df.columns)} cols)")

# ── 3. Build feature matrix (must match 02_model_training.py exactly) ─────────
NUMERIC_FEATURES = [
    "PlannedTripTime", "PlannedTOW", "PlannedZeroFuelWeight",
    "TeledyneRampWeight", "BlockTimeScheduled", "RouteDistanceKm",
    "LoadFactor", "TotalPassengers", "WeightDelta",
    "TypeWeightInteraction", "TypeTripTimeInteraction",
    "DepartureHour", "DepartureMonth", "DepartureDayOfWeek", "Freight",
]
CATEGORICAL_FEATURES = ["AircraftTypeGroup", "Carrier", "AOCDescription", "Season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X_test = df[ALL_FEATURES].copy()

# Fill missing categoricals with the column mode (prevents OHE errors)
for col in CATEGORICAL_FEATURES:
    n_missing = X_test[col].isna().sum()
    if n_missing > 0:
        mode_val = X_test[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
        X_test[col] = X_test[col].fillna(fill_val)
        print(f"  Filled {n_missing} missing in '{col}' with mode='{fill_val}'")

# Fill missing numerics with column median
for col in NUMERIC_FEATURES:
    n_missing = X_test[col].isna().sum()
    if n_missing > 0:
        med = X_test[col].median()
        X_test[col] = X_test[col].fillna(med)
        print(f"  Filled {n_missing} missing in '{col}' with median={med:.2f}")

# ── 4. Select best model and generate predictions ─────────────────────────────
model_results_path = DATA_DIR / "model_results.csv"
if not model_results_path.exists():
    raise FileNotFoundError(
        "data/model_results.csv not found. Run 02_model_training.py first."
    )

model_results = pd.read_csv(model_results_path)
ranked = model_results.sort_values("RMSE")
print(f"\nModel ranking by Val-RMSE:")
print(ranked[["Model", "RMSE", "MAPE", "R2"]].to_string(index=False))

MODEL_FILES = {
    "LightGBM":         MODEL_DIR / "lgbm_model.pkl",
    "XGBoost":          MODEL_DIR / "xgboost.pkl",
    "Random Forest":    MODEL_DIR / "rf_model.pkl",
    "Ridge Regression": MODEL_DIR / "ridge_model.pkl",
}

predictions = None
used_model  = None

for _, row in ranked.iterrows():
    model_name = row["Model"]
    pkl_path   = MODEL_FILES.get(model_name)
    if pkl_path is None or not pkl_path.exists():
        print(f"  {model_name}: pkl not found at {pkl_path}, skipping.")
        continue
    try:
        with open(pkl_path, "rb") as f:
            model_obj = pickle.load(f)

        if isinstance(model_obj, tuple):
            # XGBoost or LightGBM: serialised as (preprocessor, model)
            pre, model = model_obj
            X_arr = pre.transform(X_test)
            predictions = model.predict(X_arr)
        else:
            # sklearn Pipeline (Ridge, Random Forest)
            predictions = model_obj.predict(X_test)

        used_model = model_name
        print(f"\n  Loaded and applied: {model_name}  (Val-RMSE={row['RMSE']:.1f} kg)")
        break

    except Exception as e:
        print(f"  {model_name}: failed to predict — {e}")
        continue

if predictions is None:
    raise RuntimeError(
        "Could not load any trained model. Ensure 02_model_training.py completed successfully."
    )

# ── 5. Sanity check predictions ───────────────────────────────────────────────
n_negative = (predictions < 0).sum()
if n_negative > 0:
    print(f"\n  WARNING: {n_negative} negative predictions clipped to 0 kg.")
    predictions = np.clip(predictions, 0, None)

print(f"\nPrediction summary ({used_model} on {len(predictions)} test flights):")
print(f"  Mean:   {predictions.mean():.1f} kg")
print(f"  Std:    {predictions.std():.1f} kg")
print(f"  Min:    {predictions.min():.1f} kg")
print(f"  Max:    {predictions.max():.1f} kg")
print(f"  P25/P75:{np.percentile(predictions, 25):.1f} / {np.percentile(predictions, 75):.1f} kg")

# Cross-check against training distribution (warn if mean drifts > 20%)
train_mean = 4619.0  # known from dataset documentation
drift_pct  = abs(predictions.mean() - train_mean) / train_mean * 100
if drift_pct > 20:
    print(f"\n  WARNING: Test mean ({predictions.mean():.0f} kg) deviates "
          f"{drift_pct:.1f}% from training mean ({train_mean:.0f} kg). "
          f"Check for data distribution shift.")

# ── 6. Save Kaggle submission ──────────────────────────────────────────────────
submission = pd.DataFrame({
    "FlightID": df["FlightID"],
    "Burnoff":  np.round(predictions, 2),
})
submission_path = DATA_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSaved data/submission.csv  ({len(submission)} rows)")
print(f"\n✅ Test predictions complete. Model used: {used_model}")
