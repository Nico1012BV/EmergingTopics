"""
06_applications.py
-------------------
Builds model-derived operational analytics layers on top of the validation set.

Outputs:
  - data/flight_analytics.csv
  - data/co2_analysis.csv
  - data/cost_savings.csv
  - data/anomalies.csv
  - data/route_opportunity.csv
  - data/segment_opportunity.csv
  - data/anomaly_monitor.csv
  - data/intervention_scenarios.csv

Run:
    python 06_applications.py

This script intentionally treats the model prediction as an operational
benchmark, not a causal optimum. Positive excess burn highlights flights or
segments that consumed more fuel than the model would structurally expect.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
os.makedirs("data", exist_ok=True)

# Constants
CO2_PER_KG_FUEL = 3.16
JET_A1_PRICE_PER_KG = 0.80
FUEL_CARRY_PENALTY = 0.035
EU_ETS_PRICE_PER_TONNE = 70
RYANAIR_ANNUAL_FLIGHTS = 550_000
MIN_SEGMENT_SIZE = 8
ANOMALY_Z_THRESHOLD = 2.5


def safe_div(num, den, default=np.nan):
    den_arr = np.asarray(den)
    num_arr = np.asarray(num)
    out = np.full(np.broadcast_shapes(num_arr.shape, den_arr.shape), default, dtype=float)
    mask = den_arr != 0
    np.divide(num_arr, den_arr, out=out, where=mask)
    return out


def rmse(values):
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else np.nan


def load_primary_model(model_res, val_preds):
    available = [m for m in model_res["Model"] if m in val_preds.columns]
    if not available:
        raise ValueError("No model predictions found in data/val_predictions.csv")
    if "LightGBM" in available:
        return "LightGBM"
    return model_res[model_res["Model"].isin(available)].sort_values("RMSE").iloc[0]["Model"]


def assign_hour_band(hour):
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if 5 <= hour <= 8:
        return "Morning Peak"
    if 9 <= hour <= 12:
        return "Late Morning"
    if 13 <= hour <= 16:
        return "Afternoon"
    if 17 <= hour <= 20:
        return "Evening Peak"
    return "Night / Early"


def load_band(load_factor):
    if pd.isna(load_factor):
        return "Unknown"
    if load_factor < 0.60:
        return "<60%"
    if load_factor < 0.75:
        return "60-75%"
    if load_factor < 0.85:
        return "75-85%"
    if load_factor < 0.95:
        return "85-95%"
    return "95-100%"


def distance_band(distance):
    if pd.isna(distance):
        return "Unknown"
    if distance < 800:
        return "Short (<800 km)"
    if distance < 1500:
        return "Medium (800-1500 km)"
    if distance < 2500:
        return "Long (1500-2500 km)"
    return "Ultra-long (>2500 km)"


def compute_segment_zscores(df):
    global_mean = df["Residual_kg"].mean()
    global_std = df["Residual_kg"].std(ddof=0) or 1.0
    df["ResidualMean_segment"] = global_mean
    df["ResidualStd_segment"] = global_std
    df["SegmentBasis"] = "Global"

    specs = [
        (["ScheduledRoute", "AircraftTypeGroup"], "Route × Aircraft"),
        (["ScheduledRoute"], "Route"),
        (["AircraftTypeGroup"], "Aircraft"),
        (["Origin", "DepartureHourBand"], "Airport × Hour"),
        (["Season"], "Season"),
    ]

    for cols, label in specs:
        stats = (
            df.groupby(cols)
            .agg(_count=("Residual_kg", "size"),
                 _mean=("Residual_kg", "mean"),
                 _std=("Residual_kg", lambda x: x.std(ddof=0)))
            .reset_index()
        )
        stats["_std"] = stats["_std"].replace(0, np.nan)
        stats = stats[stats["_count"] >= MIN_SEGMENT_SIZE]
        if stats.empty:
            continue

        merged = df[cols].merge(stats, on=cols, how="left")
        mask = merged["_count"].notna() & (df["SegmentBasis"] == "Global")
        df.loc[mask, "ResidualMean_segment"] = merged.loc[mask, "_mean"].values
        df.loc[mask, "ResidualStd_segment"] = merged.loc[mask, "_std"].fillna(global_std).values
        df.loc[mask, "SegmentBasis"] = label

    df["ResidualZ_global"] = (df["Residual_kg"] - global_mean) / global_std
    df["ResidualZ_segment"] = (
        (df["Residual_kg"] - df["ResidualMean_segment"])
        / df["ResidualStd_segment"].replace(0, global_std).fillna(global_std)
    )
    return df, global_mean, global_std


def opportunity_score(df, annual_cost_col, positive_rate_col, volume_col, std_col):
    volume_scale = np.log1p(df[volume_col].clip(lower=0))
    stability = 1 - safe_div(df[std_col], df[std_col].median() + 1e-9, default=1)
    stability = np.clip(np.nan_to_num(stability, nan=0.0), 0, 1)
    score = (
        np.nan_to_num(df[annual_cost_col], nan=0.0)
        * np.nan_to_num(df[positive_rate_col], nan=0.0)
        * np.maximum(volume_scale, 1.0)
        * (0.5 + 0.5 * stability)
    )
    return score


print("Loading data ...")
df_eng = pd.read_csv("data/train_engineered.csv")
val_preds = pd.read_csv("data/val_predictions.csv", index_col=0)
model_res = pd.read_csv("data/model_results.csv")
route_err = pd.read_csv("data/route_error_analysis.csv") if os.path.exists("data/route_error_analysis.csv") else None

primary_model = load_primary_model(model_res, val_preds)
df_val = df_eng.loc[val_preds.index].copy()
df_val["PredictedBurn_kg"] = val_preds[primary_model].values
df_val["ActualBurn_kg"] = df_val["Burnoff"]
df_val["Residual_kg"] = df_val["PredictedBurn_kg"] - df_val["ActualBurn_kg"]
df_val["AbsError_kg"] = df_val["Residual_kg"].abs()
df_val["ExcessBurn_kg"] = -df_val["Residual_kg"]

print(f"  Primary model: {primary_model}")
print(f"  Validation flights: {len(df_val):,}")

# Baseline planning comparator
baseline_means = df_eng.groupby("AircraftTypeGroup")["Burnoff"].mean()
df_val["BaselinePred_kg"] = df_val["AircraftTypeGroup"].map(baseline_means)
baseline_rmse = rmse(df_val["BaselinePred_kg"] - df_val["ActualBurn_kg"])
model_rmse = rmse(df_val["PredictedBurn_kg"] - df_val["ActualBurn_kg"])

# Derived operational variables
df_val["ExcessBurnPct_pred"] = safe_div(df_val["ExcessBurn_kg"], df_val["PredictedBurn_kg"]) * 100
df_val["ExcessBurnPct_actual"] = safe_div(df_val["ExcessBurn_kg"], df_val["ActualBurn_kg"]) * 100
df_val["ActualBurnPerKm"] = safe_div(df_val["ActualBurn_kg"], df_val["RouteDistanceKm"])
df_val["PredictedBurnPerKm"] = safe_div(df_val["PredictedBurn_kg"], df_val["RouteDistanceKm"])
df_val["IntensityGapPerKm"] = df_val["ActualBurnPerKm"] - df_val["PredictedBurnPerKm"]

has_pax = df_val["TotalPassengers"] > 0
seat_distance = df_val["TotalPassengers"] * df_val["RouteDistanceKm"]
df_val["BurnPerPaxKm_actual"] = np.where(
    has_pax & (seat_distance > 0),
    safe_div(df_val["ActualBurn_kg"] * 1000, seat_distance),
    np.nan,
)
df_val["BurnPerPaxKm_pred"] = np.where(
    has_pax & (seat_distance > 0),
    safe_div(df_val["PredictedBurn_kg"] * 1000, seat_distance),
    np.nan,
)
df_val["StructuralCO2_kg"] = df_val["PredictedBurn_kg"] * CO2_PER_KG_FUEL
df_val["ActualCO2_kg"] = df_val["ActualBurn_kg"] * CO2_PER_KG_FUEL
df_val["AvoidableCO2_kg"] = df_val["ExcessBurn_kg"].clip(lower=0) * CO2_PER_KG_FUEL
df_val["AvoidableFuelCost_eur"] = df_val["ExcessBurn_kg"].clip(lower=0) * JET_A1_PRICE_PER_KG
df_val["PlannedFuelProxy_kg"] = df_val["TeledyneRampWeight"] - df_val["PlannedZeroFuelWeight"]
df_val["OpsPlanningGap_kg"] = df_val["PlannedFuelProxy_kg"] - df_val["PredictedBurn_kg"]
df_val["CarryPenaltyCost_eur"] = df_val["OpsPlanningGap_kg"].clip(lower=0) * FUEL_CARRY_PENALTY * JET_A1_PRICE_PER_KG
df_val["DepartureHourBand"] = df_val["DepartureHour"].apply(assign_hour_band)
df_val["LoadFactorBand"] = df_val["LoadFactor"].apply(load_band)
df_val["DistanceBand"] = df_val["RouteDistanceKm"].apply(distance_band)
df_val["RouteAircraft"] = df_val["ScheduledRoute"] + " | " + df_val["AircraftTypeGroup"].astype(str)
df_val["AirportHour"] = df_val["Origin"].astype(str) + " | " + df_val["DepartureHourBand"].astype(str)

df_val, residual_mean, residual_std = compute_segment_zscores(df_val)
df_val["IsOverburnAnomaly"] = df_val["ResidualZ_segment"] < -ANOMALY_Z_THRESHOLD
df_val["IsUnderburnAnomaly"] = df_val["ResidualZ_segment"] > ANOMALY_Z_THRESHOLD
df_val["IsAnomaly"] = df_val["IsOverburnAnomaly"] | df_val["IsUnderburnAnomaly"]
df_val["AnomalyType"] = np.where(
    df_val["IsOverburnAnomaly"], "Over-consumption",
    np.where(df_val["IsUnderburnAnomaly"], "Under-consumption", "Normal")
)
df_val["PossibleDataIssue"] = (
    (df_val["AbsError_kg"] > df_val["AbsError_kg"].quantile(0.995))
    | (df_val["ActualBurn_kg"] < 500)
    | (df_val["RouteDistanceKm"] < 100)
)
df_val["LikelyIssueClass"] = np.where(
    df_val["PossibleDataIssue"], "Possible data-quality issue",
    np.where(df_val["OpsPlanningGap_kg"] > 500, "Planning conservatism / fuel policy",
             np.where(df_val["IsOverburnAnomaly"], "Operational over-burn",
                      np.where(df_val["IsUnderburnAnomaly"], "Favourable conditions / under-burn", "Normal")))
)

scale_to_annual = RYANAIR_ANNUAL_FLIGHTS / len(df_val)
df_val["AnnualisedAvoidableCO2_t"] = df_val["AvoidableCO2_kg"] * scale_to_annual / 1000
df_val["AnnualisedAvoidableCost_eur"] = df_val["AvoidableFuelCost_eur"] * scale_to_annual

# Route opportunity
route_opportunity = (
    df_val.groupby("ScheduledRoute")
    .agg(
        FlightCount=("ActualBurn_kg", "size"),
        MeanActualBurn=("ActualBurn_kg", "mean"),
        MeanPredictedBurn=("PredictedBurn_kg", "mean"),
        MeanExcessBurn=("ExcessBurn_kg", "mean"),
        MedianExcessBurn=("ExcessBurn_kg", "median"),
        P90ExcessBurn=("ExcessBurn_kg", lambda x: np.quantile(x, 0.90)),
        PositiveExcessRate=("ExcessBurn_kg", lambda x: (x > 0).mean()),
        TotalAvoidableCO2_kg=("AvoidableCO2_kg", "sum"),
        TotalAvoidableCost_eur=("AvoidableFuelCost_eur", "sum"),
        MeanAvoidableCost_eur=("AvoidableFuelCost_eur", "mean"),
        MeanStructuralCO2_kg=("StructuralCO2_kg", "mean"),
        MeanAvoidableCO2_kg=("AvoidableCO2_kg", "mean"),
        AnomalyRate=("IsAnomaly", "mean"),
        OverburnAnomalyRate=("IsOverburnAnomaly", "mean"),
        ResidualStd=("Residual_kg", lambda x: x.std(ddof=0)),
        RouteRMSE=("ExcessBurn_kg", rmse),
        MeanOpsPlanningGap_kg=("OpsPlanningGap_kg", "mean"),
        MeanLoadFactor=("LoadFactor", "mean"),
        MeanDistanceKm=("RouteDistanceKm", "mean"),
        AircraftTypeMode=("AircraftTypeGroup", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    )
    .reset_index()
)
route_opportunity["AnnualAvoidableCO2_t"] = route_opportunity["TotalAvoidableCO2_kg"] * scale_to_annual / 1000
route_opportunity["AnnualAvoidableCost_eur"] = route_opportunity["TotalAvoidableCost_eur"] * scale_to_annual
route_opportunity["StructuralVsAvoidableRatio"] = safe_div(
    route_opportunity["MeanAvoidableCO2_kg"], route_opportunity["MeanStructuralCO2_kg"]
)
route_opportunity["ConfidenceScore"] = np.clip(route_opportunity["FlightCount"] / 25, 0, 1)
route_opportunity["OpportunityScore"] = opportunity_score(
    route_opportunity,
    annual_cost_col="AnnualAvoidableCost_eur",
    positive_rate_col="PositiveExcessRate",
    volume_col="FlightCount",
    std_col="ResidualStd",
)
route_opportunity["VolumePriority"] = (
    route_opportunity["FlightCount"] * route_opportunity["MeanExcessBurn"].clip(lower=0)
)
route_opportunity["PersistentOverburnFlag"] = (
    (route_opportunity["FlightCount"] >= MIN_SEGMENT_SIZE)
    & (route_opportunity["PositiveExcessRate"] >= 0.60)
    & (route_opportunity["MeanExcessBurn"] > 0)
)

if route_err is not None:
    geo_cols = [c for c in [
        "ScheduledRoute", "Origin", "Destination",
        "OrgLat", "OrgLon", "DstLat", "DstLon", "OrgCity", "DstCity"
    ] if c in route_err.columns]
    route_opportunity = route_opportunity.merge(
        route_err[geo_cols].drop_duplicates("ScheduledRoute"),
        on="ScheduledRoute",
        how="left",
    )

# Sustainability route summary
co2_analysis = (
    route_opportunity[[
        "ScheduledRoute", "FlightCount", "MeanDistanceKm", "MeanLoadFactor",
        "MeanStructuralCO2_kg", "MeanAvoidableCO2_kg",
        "AnnualAvoidableCO2_t", "StructuralVsAvoidableRatio", "OpportunityScore"
    ]]
    .rename(columns={
        "MeanStructuralCO2_kg": "MeanStructuralCO2_kg",
        "MeanAvoidableCO2_kg": "MeanAvoidableCO2_kg",
    })
    .sort_values("AnnualAvoidableCO2_t", ascending=False)
)

# Segment opportunity table
segment_frames = []
segment_specs = {
    "Route": ["ScheduledRoute"],
    "Aircraft": ["AircraftTypeGroup"],
    "Carrier": ["Carrier"],
    "Origin": ["Origin"],
    "Destination": ["Destination"],
    "Season": ["Season"],
    "Departure Hour": ["DepartureHourBand"],
    "Load Factor Band": ["LoadFactorBand"],
    "Distance Band": ["DistanceBand"],
    "Route × Aircraft": ["ScheduledRoute", "AircraftTypeGroup"],
    "Airport × Hour": ["Origin", "DepartureHourBand"],
}

for segment_type, cols in segment_specs.items():
    grouped = (
        df_val.groupby(cols)
        .agg(
            FlightCount=("ActualBurn_kg", "size"),
            MeanExcessBurn=("ExcessBurn_kg", "mean"),
            MedianExcessBurn=("ExcessBurn_kg", "median"),
            PositiveExcessRate=("ExcessBurn_kg", lambda x: (x > 0).mean()),
            AnomalyRate=("IsAnomaly", "mean"),
            OverburnAnomalyRate=("IsOverburnAnomaly", "mean"),
            MeanAvoidableCost_eur=("AvoidableFuelCost_eur", "mean"),
            TotalAvoidableCost_eur=("AvoidableFuelCost_eur", "sum"),
            TotalAvoidableCO2_kg=("AvoidableCO2_kg", "sum"),
            ResidualStd=("Residual_kg", lambda x: x.std(ddof=0)),
            MeanStructuralCO2_kg=("StructuralCO2_kg", "mean"),
            MeanOpsPlanningGap_kg=("OpsPlanningGap_kg", "mean"),
        )
        .reset_index()
    )
    grouped["SegmentType"] = segment_type
    grouped["SegmentName"] = grouped[cols].astype(str).agg(" | ".join, axis=1)
    grouped["AnnualAvoidableCost_eur"] = grouped["TotalAvoidableCost_eur"] * scale_to_annual
    grouped["AnnualAvoidableCO2_t"] = grouped["TotalAvoidableCO2_kg"] * scale_to_annual / 1000
    grouped["ConfidenceScore"] = np.clip(grouped["FlightCount"] / 25, 0, 1)
    grouped["OpportunityScore"] = opportunity_score(
        grouped,
        annual_cost_col="AnnualAvoidableCost_eur",
        positive_rate_col="PositiveExcessRate",
        volume_col="FlightCount",
        std_col="ResidualStd",
    )
    grouped["PersistentOverburnFlag"] = (
        (grouped["FlightCount"] >= MIN_SEGMENT_SIZE)
        & (grouped["PositiveExcessRate"] >= 0.60)
        & (grouped["MeanExcessBurn"] > 0)
    )
    segment_frames.append(grouped)

segment_opportunity = pd.concat(segment_frames, ignore_index=True)
segment_opportunity = segment_opportunity.sort_values("OpportunityScore", ascending=False)

# Persistent anomaly monitor
monitor_specs = {
    "Route": ["ScheduledRoute"],
    "Aircraft": ["AircraftTypeGroup"],
    "Airport × Hour": ["Origin", "DepartureHourBand"],
    "Route × Aircraft": ["ScheduledRoute", "AircraftTypeGroup"],
    "Season": ["Season"],
}

monitor_frames = []
for segment_type, cols in monitor_specs.items():
    mon = (
        df_val.groupby(cols)
        .agg(
            FlightCount=("ActualBurn_kg", "size"),
            AnomalyCount=("IsAnomaly", "sum"),
            OverburnCount=("IsOverburnAnomaly", "sum"),
            UnderburnCount=("IsUnderburnAnomaly", "sum"),
            AnomalyRate=("IsAnomaly", "mean"),
            MeanExcessBurn=("ExcessBurn_kg", "mean"),
            ResidualStd=("Residual_kg", lambda x: x.std(ddof=0)),
            MeanOpsPlanningGap_kg=("OpsPlanningGap_kg", "mean"),
        )
        .reset_index()
    )
    mon["SegmentType"] = segment_type
    mon["SegmentName"] = mon[cols].astype(str).agg(" | ".join, axis=1)
    mon["PersistenceFlag"] = (
        (mon["FlightCount"] >= MIN_SEGMENT_SIZE)
        & (mon["AnomalyRate"] >= 0.12)
    )
    mon["LikelyIssueClass"] = np.where(
        mon["MeanOpsPlanningGap_kg"] > 500,
        "Planning bias / conservative fuel loading",
        np.where(mon["MeanExcessBurn"] > 0, "Recurring operational over-burn", "Mixed / low-signal")
    )
    monitor_frames.append(mon)

anomaly_monitor = pd.concat(monitor_frames, ignore_index=True)
anomaly_monitor = anomaly_monitor.sort_values(
    ["PersistenceFlag", "AnomalyRate", "FlightCount"], ascending=[False, False, False]
)

# Intervention scenarios
scenario_base = segment_opportunity[
    segment_opportunity["SegmentType"].isin(["Route", "Aircraft", "Airport × Hour", "Route × Aircraft"])
].copy()
scenario_base["CurrentAnnualValue_eur"] = scenario_base["AnnualAvoidableCost_eur"]
scenario_rows = []
for scenario_name, factor in [
    ("Reduce excess burn by 25%", 0.25),
    ("Reduce excess burn by 50%", 0.50),
]:
    temp = scenario_base.copy()
    temp["Scenario"] = scenario_name
    temp["SavingsPotential_eur"] = temp["CurrentAnnualValue_eur"] * factor
    temp["SavingsPotential_CO2_t"] = temp["AnnualAvoidableCO2_t"] * factor
    scenario_rows.append(temp)

peer = scenario_base.copy()
peer["Scenario"] = "Move to peer median"
peer["SavingsPotential_eur"] = peer["CurrentAnnualValue_eur"] * np.clip(peer["PositiveExcessRate"] - 0.5, 0, 1)
peer["SavingsPotential_CO2_t"] = peer["AnnualAvoidableCO2_t"] * np.clip(peer["PositiveExcessRate"] - 0.5, 0, 1)
scenario_rows.append(peer)

intervention_scenarios = pd.concat(scenario_rows, ignore_index=True).sort_values(
    "SavingsPotential_eur", ascending=False
)

# Legacy-compatible outputs
anomalies = df_val[df_val["IsAnomaly"]].copy()
anomalies["AnomalyScore"] = df_val.loc[anomalies.index, "ResidualZ_segment"].abs()
anomaly_output = anomalies[[
    "ScheduledRoute", "AircraftTypeGroup", "Carrier", "Origin", "Destination",
    "Season", "DepartureHourBand", "LoadFactor", "RouteDistanceKm",
    "ActualBurn_kg", "PredictedBurn_kg", "ExcessBurn_kg", "Residual_kg",
    "AnomalyType", "AnomalyScore", "LikelyIssueClass", "OpsPlanningGap_kg"
]].copy()

buffer_reduction_kg = 1.65 * max(baseline_rmse - model_rmse, 0)
fuel_carry_saving_per_flight_kg = buffer_reduction_kg * FUEL_CARRY_PENALTY
annual_buffer_fuel_savings_t = fuel_carry_saving_per_flight_kg * RYANAIR_ANNUAL_FLIGHTS / 1000
annual_buffer_cost_savings = annual_buffer_fuel_savings_t * 1000 * JET_A1_PRICE_PER_KG
annual_buffer_co2_t = annual_buffer_fuel_savings_t * CO2_PER_KG_FUEL

annual_avoidable_cost = df_val["AvoidableFuelCost_eur"].sum() * scale_to_annual
annual_avoidable_co2_t = df_val["AvoidableCO2_kg"].sum() * scale_to_annual / 1000
annual_carry_penalty = df_val["CarryPenaltyCost_eur"].sum() * scale_to_annual

cost_savings = pd.DataFrame([
    {"Metric": "Primary model", "Value": primary_model},
    {"Metric": "Baseline RMSE (kg)", "Value": round(baseline_rmse, 1)},
    {"Metric": "Model RMSE (kg)", "Value": round(model_rmse, 1)},
    {"Metric": "Buffer reduction per flight (kg)", "Value": round(buffer_reduction_kg, 1)},
    {"Metric": "Annual buffer fuel saved (tonnes)", "Value": round(annual_buffer_fuel_savings_t, 0)},
    {"Metric": "Annual buffer cost savings (EUR)", "Value": round(annual_buffer_cost_savings, 0)},
    {"Metric": "Annual buffer CO2 avoided (tonnes)", "Value": round(annual_buffer_co2_t, 0)},
    {"Metric": "Annual avoidable operational cost (EUR)", "Value": round(annual_avoidable_cost, 0)},
    {"Metric": "Annual avoidable operational CO2 (tonnes)", "Value": round(annual_avoidable_co2_t, 0)},
    {"Metric": "Annual planning carry penalty (EUR)", "Value": round(annual_carry_penalty, 0)},
    {"Metric": "Annual ETS savings on avoidable CO2 (EUR)", "Value": round(annual_avoidable_co2_t * EU_ETS_PRICE_PER_TONNE, 0)},
    {"Metric": "Total annual value (EUR)", "Value": round(annual_avoidable_cost + annual_carry_penalty + annual_avoidable_co2_t * EU_ETS_PRICE_PER_TONNE, 0)},
])

# Persist outputs
flight_cols = [
    "ScheduledRoute", "Carrier", "AOCDescription", "AircraftTypeGroup",
    "Origin", "Destination", "Season", "DepartureYear", "DepartureMonth",
    "DepartureHour", "DepartureHourBand", "LoadFactorBand", "DistanceBand",
    "ActualBurn_kg", "PredictedBurn_kg", "BaselinePred_kg",
    "Residual_kg", "AbsError_kg", "ExcessBurn_kg",
    "ExcessBurnPct_pred", "ExcessBurnPct_actual",
    "ActualBurnPerKm", "PredictedBurnPerKm", "IntensityGapPerKm",
    "BurnPerPaxKm_actual", "BurnPerPaxKm_pred",
    "ActualCO2_kg", "StructuralCO2_kg", "AvoidableCO2_kg",
    "AvoidableFuelCost_eur", "CarryPenaltyCost_eur",
    "PlannedFuelProxy_kg", "OpsPlanningGap_kg",
    "ResidualZ_global", "ResidualZ_segment", "SegmentBasis",
    "IsAnomaly", "IsOverburnAnomaly", "IsUnderburnAnomaly",
    "AnomalyType", "LikelyIssueClass", "PossibleDataIssue",
    "AnnualisedAvoidableCO2_t", "AnnualisedAvoidableCost_eur",
]
df_val[flight_cols].to_csv("data/flight_analytics.csv", index=False)
co2_analysis.to_csv("data/co2_analysis.csv", index=False)
cost_savings.to_csv("data/cost_savings.csv", index=False)
anomaly_output.to_csv("data/anomalies.csv", index=False)
route_opportunity.to_csv("data/route_opportunity.csv", index=False)
segment_opportunity.to_csv("data/segment_opportunity.csv", index=False)
anomaly_monitor.to_csv("data/anomaly_monitor.csv", index=False)
intervention_scenarios.to_csv("data/intervention_scenarios.csv", index=False)

print("\nSaved outputs:")
for path in [
    "data/flight_analytics.csv",
    "data/co2_analysis.csv",
    "data/cost_savings.csv",
    "data/anomalies.csv",
    "data/route_opportunity.csv",
    "data/segment_opportunity.csv",
    "data/anomaly_monitor.csv",
    "data/intervention_scenarios.csv",
]:
    print(f"  - {path}")

print("\nHeadline metrics:")
print(f"  Baseline RMSE:                  {baseline_rmse:,.1f} kg")
print(f"  {primary_model} RMSE:           {model_rmse:,.1f} kg")
print(f"  Annual avoidable operational CO2: {annual_avoidable_co2_t:,.0f} tonnes")
print(f"  Annual avoidable operational cost: EUR {annual_avoidable_cost:,.0f}")
print(f"  Annual planning carry penalty:    EUR {annual_carry_penalty:,.0f}")
print(f"  Persistent risk segments:         {int(route_opportunity['PersistentOverburnFlag'].sum())} routes")
print(f"  Anomalies flagged:                {int(df_val['IsAnomaly'].sum())} flights")
