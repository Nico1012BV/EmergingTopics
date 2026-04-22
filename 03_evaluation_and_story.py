"""
03_evaluation_and_story.py
---------------------------
Deep-dive evaluation and narrative analysis. Goes beyond raw metrics to tell
the story the professor asked for:
  - Where does the model outperform the airline baseline?
  - Why might the data be biased?
  - Is error systematic across routes, aircraft types, seasons, load?
  - How do we interpret feature interactions?

Outputs:
  - data/route_error_analysis.csv   (per-route error, for the map dashboard)
  - figures/story_*.png
  - data/bias_analysis.csv

Run: python 03_evaluation_and_story.py
(Must run after 02_model_training.py)
"""

import os
import pickle
import sys
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
MPLCONFIG_DIR = CACHE_DIR / "matplotlib"
CACHE_DIR.mkdir(exist_ok=True)
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(BASE_DIR)  # ensure all relative paths resolve to the script's directory
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130})


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# ── 1. Load predictions and engineered data ───────────────────────────────────
print("Loading data …")
val_preds  = pd.read_csv("data/val_predictions.csv", index_col=0)
model_res  = pd.read_csv("data/model_results.csv")
df_eng     = pd.read_csv("data/train_engineered.csv")

available_models = [m for m in model_res["Model"] if m in val_preds.columns]
primary_model = "LightGBM" if "LightGBM" in available_models else model_res[
    model_res["Model"].isin(available_models)
].sort_values("RMSE").iloc[0]["Model"]

y_true    = val_preds["y_true"].values
y_primary = val_preds[primary_model].values

# Merge aircraft metadata back into val
df_val = df_eng.loc[val_preds.index].copy()
df_val["y_pred_primary"] = y_primary
df_val["Residual"]       = y_primary - y_true
df_val["AbsError"]     = np.abs(df_val["Residual"])
df_val["RelError"]     = df_val["AbsError"] / df_val["Burnoff"].clip(lower=1)
df_val["ExcessBurn_kg"] = -df_val["Residual"]
df_val["ExcessBurnPct"] = df_val["ExcessBurn_kg"] / df_val["y_pred_primary"].clip(lower=1) * 100

# ── 2. Baseline comparison ────────────────────────────────────────────────────
# Load the airline's own predictions from predicted_fuel_consumption.csv.
# This file ships with the Kaggle competition and contains the Teledyne/ops
# system's pre-flight fuel forecasts for every flight in train.csv.
# Column expected: a fuel prediction column (float, kg) aligned to train.csv rows.

def load_airline_baseline(df_eng, val_index):
    """
    Try to load predicted_fuel_consumption.csv and align to validation rows.
    Returns a Series of baseline predictions indexed like val_index, or None.
    """
    candidates = [
        "predicted_fuel_consumption.csv",
        "data/predicted_fuel_consumption.csv",
        "../data/predicted_fuel_consumption.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            bl = pd.read_csv(path)
            print(f"  Loaded airline baseline from: {path}  shape={bl.shape}")
            print(f"  Columns: {bl.columns.tolist()}")

            # Identify the prediction column (float column that isn't an id/index)
            float_cols = bl.select_dtypes(include="number").columns.tolist()
            # Drop obvious id columns
            pred_col = next(
                (c for c in float_cols if "burnoff" in c.lower() or "fuel" in c.lower()
                 or "pred" in c.lower() or "consumption" in c.lower()),
                float_cols[0] if float_cols else None
            )
            if pred_col is None:
                print("  WARNING: could not identify prediction column in baseline CSV")
                return None

            print(f"  Using column '{pred_col}' as airline baseline prediction")

            # Align: the CSV rows should correspond 1-to-1 with train.csv rows
            # (same row order, same length). We index into it using df_eng's
            # positional index relative to the original file.
            if len(bl) == len(df_eng):
                baseline_series = bl[pred_col].values  # positional
                # val_index contains the integer labels from df_eng
                # We need the positional location of those labels
                positional = df_eng.index.get_indexer(val_index)
                if (positional == -1).any():
                    print("  WARNING: some val indices not found in df_eng index")
                    return None
                return pd.Series(baseline_series[positional], index=val_index)
            else:
                print(f"  WARNING: baseline length {len(bl)} != df_eng length {len(df_eng)}; skipping")
                return None
    print("  predicted_fuel_consumption.csv not found — falling back to naive baseline")
    return None

baseline_preds = load_airline_baseline(df_eng, val_preds.index)

if baseline_preds is not None:
    df_val["Baseline_pred"]  = baseline_preds.values
    baseline_label = "Airline baseline (Teledyne/ops)"
else:
    # Fallback: mean burnoff per aircraft type
    baseline_means = df_eng.groupby("AircraftTypeGroup")["Burnoff"].mean()
    df_val["Baseline_pred"] = df_val["AircraftTypeGroup"].map(baseline_means)
    baseline_label = "Naive baseline (mean/type)"
    print(f"  Using fallback: {baseline_label}")

df_val["Baseline_resid"] = df_val["Baseline_pred"] - y_true

print(f"\n--- Baseline vs {primary_model} ---")
for label, pred in [(baseline_label, df_val["Baseline_pred"].values),
                    (primary_model, y_primary)]:
    print(f"  {label:40s}  RMSE={rmse(y_true, pred):8.1f}  "
          f"MAPE={mean_absolute_percentage_error(y_true, pred)*100:.3f}%  "
          f"R²={r2_score(y_true, pred):.4f}")

# ── 3. Per-route error analysis ───────────────────────────────────────────────
from airport_coords import get_info

route_err = (
    df_val.groupby("ScheduledRoute")
    .agg(
        FlightCount=("Burnoff", "count"),
        MeanBurnoff=("Burnoff", "mean"),
        MeanPred=("y_pred_primary", "mean"),
        RMSE=("Residual", lambda x: np.sqrt((x**2).mean())),
        MAPE=("RelError", "mean"),
        MeanAbsError=("AbsError", "mean"),
        MeanTripTime=("PlannedTripTime", "mean"),
        MeanLoadFactor=("LoadFactor", "mean"),
        MeanDistKm=("RouteDistanceKm", "mean"),
    )
    .reset_index()
)
route_err["Origin"]      = route_err["ScheduledRoute"].str.split("-").str[0]
route_err["Destination"] = route_err["ScheduledRoute"].str.split("-").str[1]

def _lat(c): i = get_info(c); return i["lat"] if i else None
def _lon(c): i = get_info(c); return i["lon"] if i else None
def _city(c): i = get_info(c); return i["city"] if i else c

for prefix, col in [("Org", "Origin"), ("Dst", "Destination")]:
    route_err[f"{prefix}Lat"]  = route_err[col].map(_lat)
    route_err[f"{prefix}Lon"]  = route_err[col].map(_lon)
    route_err[f"{prefix}City"] = route_err[col].map(_city)

route_err = route_err.dropna(subset=["OrgLat", "OrgLon"])
route_err.to_csv("data/route_error_analysis.csv", index=False)
print(f"\nSaved data/route_error_analysis.csv ({len(route_err)} routes)")

# Top worst routes
worst = route_err.nlargest(10, "RMSE")[["ScheduledRoute","FlightCount","RMSE","MAPE","MeanBurnoff"]]
print("\n--- Top 10 routes by RMSE ---")
print(worst.to_string(index=False))

# ── 4. Story figures ──────────────────────────────────────────────────────────

# 4a. Error by aircraft type – H4 test
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df_val, x="AircraftTypeGroup", y="AbsError",
            estimator=np.mean, ci=95, palette="Set2", ax=ax)
ax.set(title="Mean Absolute Error by Aircraft Type\n(Testing H4: MAX vs NG variability)",
       xlabel="Aircraft Type", ylabel="MAE (kg)")
plt.tight_layout()
plt.savefig("figures/story_error_by_aircraft.png")
plt.close()

# 4b. Error by season
fig, ax = plt.subplots(figsize=(8, 5))
season_order = ["Winter", "Spring", "Summer", "Fall"]
sns.barplot(data=df_val, x="Season", y="AbsError",
            order=season_order, estimator=np.mean, ci=95, palette="coolwarm", ax=ax)
ax.set(title="Mean Absolute Error by Season", xlabel="Season", ylabel="MAE (kg)")
plt.tight_layout()
plt.savefig("figures/story_error_by_season.png")
plt.close()

# 4c. Error vs load factor (bias analysis)
fig, ax = plt.subplots(figsize=(8, 5))
bins = pd.cut(df_val["LoadFactor"], bins=10)
lf_err = df_val.groupby(bins, observed=True)["Residual"].mean()
ax.bar(range(len(lf_err)), lf_err.values, color=["#DC2626" if v > 0 else "#2563EB" for v in lf_err.values])
ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(range(len(lf_err)))
ax.set_xticklabels([str(i.mid.round(2)) for i in lf_err.index], rotation=45)
ax.set(title="Mean Residual by Load Factor Bin\n(Bias check: over/under-prediction per load)",
       xlabel="Load Factor (midpoint)", ylabel="Mean Residual (kg)")
plt.tight_layout()
plt.savefig("figures/story_bias_by_loadfactor.png")
plt.close()

# 4d. Error vs route distance
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_val["RouteDistanceKm"], df_val["AbsError"],
           alpha=0.15, s=5, c="#7C3AED")
z = np.polyfit(df_val["RouteDistanceKm"].fillna(0), df_val["AbsError"], 1)
p = np.poly1d(z)
xs = np.linspace(df_val["RouteDistanceKm"].min(), df_val["RouteDistanceKm"].max(), 200)
ax.plot(xs, p(xs), "r-", linewidth=2, label="Trend")
ax.set(title="Absolute Error vs Route Distance",
       xlabel="Route Distance (km)", ylabel="Absolute Error (kg)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/story_error_vs_distance.png")
plt.close()

# 4e. Residual over time (temporal bias / data drift)
df_val_time = df_val.copy()
df_val_time["DepartureScheduled"] = pd.to_datetime(df_val_time["DepartureScheduled"])
monthly = df_val_time.groupby(
    df_val_time["DepartureScheduled"].dt.to_period("M")
)["Residual"].mean()
monthly.index = monthly.index.astype(str)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly.index, monthly.values, marker="o", markersize=4, color="#2563EB")
ax.axhline(0, color="red", linestyle="--")
ax.set(title="Mean Residual by Month (Temporal Bias / Drift Check)",
       xlabel="Month", ylabel="Mean Residual (kg)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("figures/story_temporal_bias.png")
plt.close()

# 4f. Model comparison: all 4 models on the same axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
model_preds = {
    model_name: val_preds[model_name].values
    for model_name in available_models
}
colors = ["#6B7280", "#16A34A", "#DC2626", "#2563EB"]
for ax, (name, preds), col in zip(axes.flat, model_preds.items(), colors):
    sample_idx = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
    ax.scatter(y_true[sample_idx], preds[sample_idx], alpha=0.2, s=6, c=col)
    lims = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    r2 = r2_score(y_true, preds)
    mpe = mean_absolute_percentage_error(y_true, preds) * 100
    ax.set_title(f"{name}\nRMSE={rmse(y_true,preds):.0f}  R²={r2:.4f}  MAPE={mpe:.2f}%")
    ax.set_xlabel("Actual (kg)")
    ax.set_ylabel("Predicted (kg)")
for ax in axes.flat[len(model_preds):]:
    ax.axis("off")
plt.suptitle("All Models: Predicted vs Actual Burnoff", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/story_all_models_comparison.png", bbox_inches="tight")
plt.close()

# ── 5. Bias analysis summary ─────────────────────────────────────────────────
bias = {
    "aircraft_type": df_val.groupby("AircraftTypeGroup")["Residual"].agg(["mean","std"]).round(2),
    "season":        df_val.groupby("Season")["Residual"].agg(["mean","std"]).round(2),
    "year":          df_val.groupby("DepartureYear")["Residual"].agg(["mean","std"]).round(2),
}
print("\n--- Residual bias by aircraft type ---")
print(bias["aircraft_type"])
print("\n--- Residual bias by season ---")
print(bias["season"])

bias_df = pd.concat([b.assign(Group=k) for k, b in bias.items()])
bias_df.to_csv("data/bias_analysis.csv")
print("\nSaved data/bias_analysis.csv")

# Save baseline vs model summary for dashboard / report reference
baseline_comparison = pd.DataFrame([
    {
        "Label": baseline_label,
        "RMSE":  round(rmse(y_true, df_val["Baseline_pred"].values), 2),
        "MAPE":  round(mean_absolute_percentage_error(y_true, df_val["Baseline_pred"].values) * 100, 3),
        "R2":    round(r2_score(y_true, df_val["Baseline_pred"].values), 4),
    },
    {
        "Label": primary_model,
        "RMSE":  round(rmse(y_true, y_primary), 2),
        "MAPE":  round(mean_absolute_percentage_error(y_true, y_primary) * 100, 3),
        "R2":    round(r2_score(y_true, y_primary), 4),
    },
])
baseline_comparison.to_csv("data/baseline_comparison.csv", index=False)
print("Saved data/baseline_comparison.csv")

validation_export = df_val[[
    "ScheduledRoute", "Carrier", "AOCDescription", "AircraftTypeGroup",
    "Origin", "Destination", "Season", "DepartureYear", "DepartureMonth",
    "DepartureHour", "Burnoff", "y_pred_primary", "Baseline_pred",
    "Residual", "AbsError", "RelError", "ExcessBurn_kg", "ExcessBurnPct",
    "RouteDistanceKm", "LoadFactor", "TotalPassengers", "PlannedTripTime",
    "PlannedTOW", "PlannedZeroFuelWeight", "TeledyneRampWeight",
]].copy()
validation_export.to_csv("data/validation_diagnostics.csv", index=False)
print("Saved data/validation_diagnostics.csv")

print("\n✅ Evaluation and story analysis complete.")
