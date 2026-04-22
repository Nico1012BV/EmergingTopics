"""
01_feature_engineering.py
--------------------------
Loads train.csv, builds all engineered features, performs EDA and outputs:
  - data/train_engineered.csv   (feature-rich dataset for modelling)
  - data/eda_summary.csv        (per-route aggregate stats for the dashboard)
  - figures/eda_*.png           (saved EDA plots)

Run: python 01_feature_engineering.py
"""

from __future__ import annotations

import math
import os
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

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Ensure sibling modules (airport_coords) are importable regardless of cwd
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ── airport helpers ─────────────────────────────────────────────────────────
from airport_coords import get_coords


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def route_distance_km(route: str) -> float | None:
    """Parse 'ORG-DST' and return great-circle km, or None."""
    try:
        org, dst = route.split("-")
        c1, c2 = get_coords(org), get_coords(dst)
        if c1 and c2:
            return haversine_km(*c1, *c2)
    except Exception:
        pass
    return None


# ── 1. Load data ─────────────────────────────────────────────────────────────
def resolve_train_csv() -> Path:
    env_path = os.environ.get("DATA_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([
        BASE_DIR / "train" / "train.csv",
        BASE_DIR / "data" / "train.csv",
        BASE_DIR / "train.csv",
        Path.cwd() / "train" / "train.csv",
        Path.cwd() / "data" / "train.csv",
        Path.cwd() / "train.csv",
    ])

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find train.csv. Checked:\n"
        f"{searched}\n"
        "Set DATA_PATH=/full/path/to/train.csv to override."
    )


print("Loading train.csv …")
data_path = resolve_train_csv()
df = pd.read_csv(data_path)
print(f"  Source: {data_path}")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ── 2. Basic cleaning ────────────────────────────────────────────────────────
# Remove clearly impossible burnoff values (negative = bad data)
before = len(df)
df = df[df["Burnoff"] > 0].copy()
print(f"  Removed {before - len(df)} rows with non-positive Burnoff")

# Parse timestamps
df["DepartureScheduled"] = pd.to_datetime(df["DepartureScheduled"])
df["ArrivalScheduled"]   = pd.to_datetime(df["ArrivalScheduled"])

# ── 3. Time features ─────────────────────────────────────────────────────────
df["DepartureHour"]  = df["DepartureScheduled"].dt.hour
df["DepartureMonth"] = df["DepartureScheduled"].dt.month
df["DepartureYear"]  = df["DepartureScheduled"].dt.year
df["DepartureDayOfWeek"] = df["DepartureScheduled"].dt.dayofweek  # 0=Mon

# Season
def month_to_season(m):
    if m in [12, 1, 2]:  return "Winter"
    elif m in [3, 4, 5]: return "Spring"
    elif m in [6, 7, 8]: return "Summer"
    else:                return "Fall"

df["Season"] = df["DepartureMonth"].map(month_to_season)

# ── 4. Route features ────────────────────────────────────────────────────────
print("  Computing great-circle distances …")
df["Origin"]      = df["ScheduledRoute"].str.split("-").str[0]
df["Destination"] = df["ScheduledRoute"].str.split("-").str[1]

df["RouteDistanceKm"] = df["ScheduledRoute"].apply(route_distance_km)

# Fallback: estimate from BlockTimeScheduled (typical cruise ~750 km/h)
SPEED_KMH = 750
missing_mask = df["RouteDistanceKm"].isna()
df.loc[missing_mask, "RouteDistanceKm"] = df.loc[missing_mask, "BlockTimeScheduled"] / 60 * SPEED_KMH
print(f"  Distance imputed for {missing_mask.sum()} routes via BlockTime fallback")

# ── 5. Load & passenger features ─────────────────────────────────────────────
df["TotalPassengers"] = df["Adults"] + df["Children"] + df["Infants"]
df["LoadFactor"]      = (df["TotalPassengers"] / df["AircraftCapacity"]).clip(0, 1)
df["FreightPerKg"]    = df["Freight"].fillna(0)

# ── 6. Weight features ───────────────────────────────────────────────────────
df["WeightDelta"] = df["PlannedTOW"] - df["PlannedZeroFuelWeight"]  # fuel at takeoff

# ── 7. Interaction features (domain-motivated) ───────────────────────────────
# Aircraft-type × weight: captures that Max burns fuel differently than NG per kg
df["TypeWeightInteraction"]    = df["AircraftTypeGroup"].map({"NG": 1, "Max": 2, "Airbus": 1.5}).fillna(1) * df["PlannedTOW"]
df["TypeTripTimeInteraction"]  = df["AircraftTypeGroup"].map({"NG": 1, "Max": 2, "Airbus": 1.5}).fillna(1) * df["PlannedTripTime"]

# Fuel efficiency proxy: burnoff per minute of planned trip
df["BurnoffPerMinute"] = df["Burnoff"] / (df["PlannedTripTime"] / 60).clip(lower=0.1)

# ── 8. Airline baseline error ────────────────────────────────────────────────
# TeledyneRampWeight includes planned fuel; difference reveals over/under-fuelling
df["BaselineDelta"] = df["TeledyneRampWeight"] - df["PlannedTOW"]

# ── 8b. VIF analysis — multicollinearity diagnostic ──────────────────────────
# Proposed in §4.5 as mitigation for multicollinearity among weight features.
# VIF > 10 → high collinearity; flagged red in figure.
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    VIF_COLS = [
        "PlannedTripTime", "PlannedTOW", "PlannedZeroFuelWeight",
        "TeledyneRampWeight", "BlockTimeScheduled", "RouteDistanceKm",
        "LoadFactor", "TotalPassengers", "WeightDelta",
        "TypeWeightInteraction", "TypeTripTimeInteraction",
    ]
    _vif_mat = df[VIF_COLS].dropna().values.astype(float)
    vif_df = pd.DataFrame({
        "Feature": VIF_COLS,
        "VIF": [
            variance_inflation_factor(_vif_mat, i)
            for i in range(len(VIF_COLS))
        ],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    vif_df.to_csv(DATA_DIR / "vif_analysis.csv", index=False)
    print(f"\nVIF analysis ({len(VIF_COLS)} features):")
    print(vif_df.to_string(index=False))
    print("  → Features with VIF > 10 are multicollinear; Ridge L2 regularisation mitigates this.")

    bar_colors = [
        "#DC2626" if v > 10 else "#F59E0B" if v > 5 else "#16A34A"
        for v in vif_df["VIF"]
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vif_df["Feature"][::-1], vif_df["VIF"][::-1],
            color=bar_colors[::-1], edgecolor="white", linewidth=0.4)
    ax.axvline(5,  color="#F59E0B", linestyle="--", linewidth=1.4,
               label="VIF = 5  (moderate multicollinearity)")
    ax.axvline(10, color="#DC2626", linestyle="--", linewidth=1.4,
               label="VIF = 10 (high multicollinearity)")
    ax.set(title="Variance Inflation Factor — Multicollinearity Diagnostic",
           xlabel="VIF", ylabel="Feature")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_vif_analysis.png")
    plt.close()
    print("Saved figures/eda_vif_analysis.png")

except ImportError:
    print("  statsmodels not installed — skipping VIF. Run: pip install statsmodels")

# ── 9. Save engineered dataset ───────────────────────────────────────────────
df.to_csv(DATA_DIR / "train_engineered.csv", index=False)
print(f"\nSaved data/train_engineered.csv  ({len(df)} rows, {len(df.columns)} cols)")

# ── 10. EDA summary for dashboard ────────────────────────────────────────────
from airport_coords import get_info

agg = (
    df.groupby("ScheduledRoute")
    .agg(
        FlightCount=("Burnoff", "count"),
        MeanBurnoff=("Burnoff", "mean"),
        MedianBurnoff=("Burnoff", "median"),
        StdBurnoff=("Burnoff", "std"),
        MeanTOW=("PlannedTOW", "mean"),
        MeanTripTime=("PlannedTripTime", "mean"),
        MeanLoadFactor=("LoadFactor", "mean"),
        MeanDistanceKm=("RouteDistanceKm", "mean"),
    )
    .reset_index()
)
agg["Origin"]      = agg["ScheduledRoute"].str.split("-").str[0]
agg["Destination"] = agg["ScheduledRoute"].str.split("-").str[1]

def _lat(code): c = get_coords(code); return c[0] if c else None
def _lon(code): c = get_coords(code); return c[1] if c else None
def _city(code): i = get_info(code); return i["city"] if i else code
def _country(code): i = get_info(code); return i["country"] if i else ""

for prefix, col in [("Org", "Origin"), ("Dst", "Destination")]:
    agg[f"{prefix}Lat"]     = agg[col].map(_lat)
    agg[f"{prefix}Lon"]     = agg[col].map(_lon)
    agg[f"{prefix}City"]    = agg[col].map(_city)
    agg[f"{prefix}Country"] = agg[col].map(_country)

agg.to_csv(DATA_DIR / "eda_summary.csv", index=False)
print(f"Saved data/eda_summary.csv  ({len(agg)} routes)")

# ── 11. EDA plots ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

# 11a. Burnoff distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Burnoff"], bins=80, color="#2563EB", edgecolor="white", linewidth=0.4)
axes[0].set(title="Burnoff Distribution", xlabel="Burnoff (kg)", ylabel="Count")
axes[1].hist(np.log1p(df["Burnoff"]), bins=80, color="#7C3AED", edgecolor="white", linewidth=0.4)
axes[1].set(title="log(Burnoff+1) Distribution", xlabel="log(Burnoff)", ylabel="Count")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_burnoff_distribution.png")
plt.close()

# 11b. Burnoff by aircraft type
fig, ax = plt.subplots(figsize=(8, 5))
order = df.groupby("AircraftTypeGroup")["Burnoff"].median().sort_values().index
sns.boxplot(data=df, x="AircraftTypeGroup", y="Burnoff", order=order, palette="Set2", ax=ax, width=0.5)
ax.set(title="Burnoff by Aircraft Type", xlabel="Aircraft Type", ylabel="Burnoff (kg)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_burnoff_by_aircraft.png")
plt.close()

# 11c. Correlation heatmap (numeric features)
num_cols = [
    "Burnoff", "PlannedTripTime", "PlannedTOW", "PlannedZeroFuelWeight",
    "TeledyneRampWeight", "RouteDistanceKm", "LoadFactor", "TotalPassengers",
    "BlockTimeScheduled", "WeightDelta"
]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_correlation_heatmap.png")
plt.close()

# 11d. Burnoff vs PlannedTripTime
fig, ax = plt.subplots(figsize=(8, 5))
sample = df.sample(min(5000, len(df)), random_state=42)
colors = {"NG": "#2563EB", "Max": "#DC2626", "Airbus": "#16A34A"}
for atype, grp in sample.groupby("AircraftTypeGroup"):
    ax.scatter(grp["PlannedTripTime"] / 60, grp["Burnoff"],
               c=colors.get(atype, "grey"), label=atype, alpha=0.35, s=8)
ax.set(title="Burnoff vs Planned Trip Time", xlabel="Trip Time (hours)", ylabel="Burnoff (kg)")
ax.legend(title="Aircraft")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_burnoff_vs_triptime.png")
plt.close()

# 11e. Burnoff by Season
fig, ax = plt.subplots(figsize=(8, 5))
order_s = ["Winter", "Spring", "Summer", "Fall"]
sns.violinplot(data=df, x="Season", y="Burnoff", order=order_s,
               palette="pastel", inner="quartile", ax=ax)
ax.set(title="Burnoff Distribution by Season", xlabel="Season", ylabel="Burnoff (kg)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_burnoff_by_season.png")
plt.close()

# 11f. Load factor vs Burnoff
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample["LoadFactor"], sample["Burnoff"], alpha=0.3, s=8, c="#F59E0B")
ax.set(title="Load Factor vs Burnoff", xlabel="Load Factor", ylabel="Burnoff (kg)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "eda_loadfactor_vs_burnoff.png")
plt.close()

print("\n✅ EDA complete. Figures saved to figures/")
