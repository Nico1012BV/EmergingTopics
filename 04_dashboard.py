"""
04_dashboard.py
----------------
Interactive web dashboard built with Plotly Dash + Dash Bootstrap Components.
Runs a local server at http://127.0.0.1:8050

Panels:
  1. OVERVIEW KPI cards  – dataset stats, best model metrics
  2. INTERACTIVE MAP     – Europe/world map of routes coloured by avg burnoff
                           or prediction error; clicking a route shows detail
  3. EDA EXPLORER        – dropdown-driven charts (burnoff distribution,
                           seasonal, aircraft, load-factor)
  4. MODEL PERFORMANCE   – all-model metric comparison, predicted vs actual,
                           residual diagnostics
  5. FEATURE IMPORTANCE  – horizontal bar chart (LightGBM + RF)
  6. STORY ANALYSIS      – temporal bias, route-level error heat map,
                           hypothesis validation cards

Pre-requisites (run these first):
    python 01_feature_engineering.py
    python 02_model_training.py
    python 03_evaluation_and_story.py

Run:
    python 04_dashboard.py
"""

import os
import warnings
from pathlib import Path

try:
    import comm

    def _noop_create_comm(*args, **kwargs):
        return None

    # Dash imports optional Jupyter support at module import time. In this
    # environment `comm.create_comm()` exists but raises NotImplementedError,
    # which breaks a normal local app launch. Replace it with a harmless no-op.
    comm.create_comm = _noop_create_comm
except Exception:
    pass

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)  # ensure all relative paths resolve to the script's directory

# ═══════════════════════════════════════════════════════════════════════════════
# 0. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path, **kw):
    if os.path.exists(path):
        return pd.read_csv(path, **kw)
    raise FileNotFoundError(f"Required file not found: {path}\n"
                            "Run 01_feature_engineering.py → 02_model_training.py → "
                            "03_evaluation_and_story.py first.")

df_eng       = _load("data/train_engineered.csv")
eda_summary  = _load("data/eda_summary.csv")
val_preds    = _load("data/val_predictions.csv", index_col=0)
route_err    = _load("data/route_error_analysis.csv")
model_res    = _load("data/model_results.csv")
feat_imp     = _load("data/feature_importance.csv")
AVAILABLE_MODELS = [m for m in model_res["Model"] if m in val_preds.columns]
PRIMARY_MODEL = "LightGBM" if "LightGBM" in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
DEFAULT_FI_MODEL = "LightGBM" if "LightGBM" in feat_imp["Model"].unique() else feat_imp["Model"].iloc[0]

# Optional application data (from 06_applications.py)
co2_data = _load("data/co2_analysis.csv") if os.path.exists("data/co2_analysis.csv") else None
cost_data = _load("data/cost_savings.csv") if os.path.exists("data/cost_savings.csv") else None
anomaly_data = _load("data/anomalies.csv") if os.path.exists("data/anomalies.csv") else None
flight_analytics = _load("data/flight_analytics.csv") if os.path.exists("data/flight_analytics.csv") else None
route_opp = _load("data/route_opportunity.csv") if os.path.exists("data/route_opportunity.csv") else None
segment_opp = _load("data/segment_opportunity.csv") if os.path.exists("data/segment_opportunity.csv") else None
anomaly_monitor = _load("data/anomaly_monitor.csv") if os.path.exists("data/anomaly_monitor.csv") else None
intervention_data = _load("data/intervention_scenarios.csv") if os.path.exists("data/intervention_scenarios.csv") else None

# The What-If simulator below is based on historical route/aircraft aggregates.
# Avoid unpickling sklearn artifacts at import time because those files are
# version-sensitive and not required for the dashboard to render.

# Merge aircraft info into val preds
df_val = df_eng.loc[val_preds.index].copy()
df_val["y_pred"] = val_preds[PRIMARY_MODEL].values
df_val["Residual"] = df_val["y_pred"] - df_val["Burnoff"]
df_val["AbsError"] = df_val["Residual"].abs()
df_val["ExcessBurn_kg"] = -df_val["Residual"]

CO2_PER_KG_FUEL = 3.16

if flight_analytics is None:
    flight_analytics = df_val.copy()
    flight_analytics["PredictedBurn_kg"] = flight_analytics["y_pred"]
    flight_analytics["ActualBurn_kg"] = flight_analytics["Burnoff"]
    flight_analytics["AvoidableCO2_kg"] = flight_analytics["ExcessBurn_kg"].clip(lower=0) * CO2_PER_KG_FUEL
    flight_analytics["StructuralCO2_kg"] = flight_analytics["PredictedBurn_kg"] * CO2_PER_KG_FUEL
    flight_analytics["AvoidableFuelCost_eur"] = flight_analytics["ExcessBurn_kg"].clip(lower=0) * 0.80
    flight_analytics["IsAnomaly"] = flight_analytics["AbsError"] > flight_analytics["AbsError"].mean() + 2 * flight_analytics["AbsError"].std()
    flight_analytics["AnomalyType"] = np.where(flight_analytics["ExcessBurn_kg"] > 0, "Over-consumption", "Under-consumption")
    flight_analytics["DepartureHourBand"] = pd.cut(
        flight_analytics["DepartureHour"],
        bins=[-1, 8, 12, 16, 20, 24],
        labels=["Morning Peak", "Late Morning", "Afternoon", "Evening Peak", "Night / Early"],
    ).astype(str)
    flight_analytics["LoadFactorBand"] = pd.cut(
        flight_analytics["LoadFactor"],
        bins=[-0.01, 0.6, 0.75, 0.85, 0.95, 1.0],
        labels=["<60%", "60-75%", "75-85%", "85-95%", "95-100%"],
    ).astype(str)
    flight_analytics["DistanceBand"] = pd.cut(
        flight_analytics["RouteDistanceKm"],
        bins=[-1, 800, 1500, 2500, np.inf],
        labels=["Short (<800 km)", "Medium (800-1500 km)", "Long (1500-2500 km)", "Ultra-long (>2500 km)"],
    ).astype(str)

if route_opp is None:
    route_opp = (
        flight_analytics.groupby("ScheduledRoute")
        .agg(
            FlightCount=("Burnoff", "size"),
            MeanExcessBurn=("ExcessBurn_kg", "mean"),
            AnnualAvoidableCost_eur=("AvoidableFuelCost_eur", "sum"),
            AnnualAvoidableCO2_t=("AvoidableCO2_kg", lambda x: x.sum() / 1000),
            AnomalyRate=("IsAnomaly", "mean"),
            MeanDistanceKm=("RouteDistanceKm", "mean"),
            MeanLoadFactor=("LoadFactor", "mean"),
        )
        .reset_index()
    )
    route_opp["OpportunityScore"] = route_opp["AnnualAvoidableCost_eur"] * np.log1p(route_opp["FlightCount"])

if segment_opp is None:
    segment_opp = (
        flight_analytics.groupby("AircraftTypeGroup")
        .agg(
            FlightCount=("Burnoff", "size"),
            MeanExcessBurn=("ExcessBurn_kg", "mean"),
            AnnualAvoidableCost_eur=("AvoidableFuelCost_eur", "sum"),
            AnnualAvoidableCO2_t=("AvoidableCO2_kg", lambda x: x.sum() / 1000),
            AnomalyRate=("IsAnomaly", "mean"),
        )
        .reset_index()
        .rename(columns={"AircraftTypeGroup": "SegmentName"})
    )
    segment_opp["SegmentType"] = "Aircraft"
    segment_opp["OpportunityScore"] = segment_opp["AnnualAvoidableCost_eur"] * np.log1p(segment_opp["FlightCount"])

if anomaly_monitor is None:
    anomaly_monitor = (
        flight_analytics.groupby("ScheduledRoute")
        .agg(
            FlightCount=("Burnoff", "size"),
            AnomalyCount=("IsAnomaly", "sum"),
            AnomalyRate=("IsAnomaly", "mean"),
            MeanExcessBurn=("ExcessBurn_kg", "mean"),
        )
        .reset_index()
        .rename(columns={"ScheduledRoute": "SegmentName"})
    )
    anomaly_monitor["SegmentType"] = "Route"
    anomaly_monitor["PersistenceFlag"] = anomaly_monitor["AnomalyRate"] > anomaly_monitor["AnomalyRate"].median()

if intervention_data is None:
    intervention_data = route_opp.copy()
    intervention_data["SegmentType"] = "Route"
    intervention_data["SegmentName"] = intervention_data["ScheduledRoute"]
    intervention_data["Scenario"] = "Reduce excess burn by 25%"
    intervention_data["SavingsPotential_eur"] = intervention_data["AnnualAvoidableCost_eur"] * 0.25
    intervention_data["SavingsPotential_CO2_t"] = intervention_data["AnnualAvoidableCO2_t"] * 0.25

# ═══════════════════════════════════════════════════════════════════════════════
# 1. APP INIT
# ═══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Ryanair Burnoff Analytics",
)

COLORS = {
    "primary":    "#2563EB",
    "secondary":  "#7C3AED",
    "danger":     "#DC2626",
    "success":    "#16A34A",
    "warning":    "#F59E0B",
    "dark":       "#1E293B",
    "light_bg":   "#F8FAFC",
    "card_bg":    "#FFFFFF",
    "border":     "#E2E8F0",
}

PLOTLY_TEMPLATE = "plotly_white"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(title, value, subtitle="", color=COLORS["primary"], icon=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div(f"{icon} {title}", className="text-muted small mb-1"),
            html.H3(value, style={"color": color, "fontWeight": "700", "margin": "0"}),
            html.Div(subtitle, className="text-muted small mt-1"),
        ]),
        style={"border": f"1px solid {COLORS['border']}", "borderRadius": "10px",
               "background": COLORS["card_bg"]},
        className="shadow-sm h-100",
    )


def section_header(title, subtitle=""):
    return html.Div([
        html.H4(title, style={"color": COLORS["dark"], "fontWeight": "700", "marginBottom": "4px"}),
        html.P(subtitle, className="text-muted small", style={"marginBottom": "16px"}),
    ])


def metric_lookup(df, label):
    if df is None or len(df) == 0:
        return None
    row = df[df["Metric"] == label]
    if row.empty:
        return None
    return row.iloc[0]["Value"]


def chart_caption(what, how, result, why=None, action=None):
    paragraphs = []
    explainer_parts = [part.strip() for part in [what, how] if part]
    takeaway_parts = [part.strip() for part in [result, why] if part]
    if explainer_parts:
        paragraphs.append(html.P(" ".join(explainer_parts), className="mb-2"))
    if takeaway_parts:
        paragraphs.append(html.P(" ".join(takeaway_parts), className="mb-2" if action else "mb-0"))
    if action:
        paragraphs.append(html.P([html.Strong("Recommended action: "), html.Span(action)], className="mb-0"))
    return dbc.Alert(
        paragraphs,
        color="light",
        className="small py-2 mt-2 mb-0",
        style={"border": f"1px solid {COLORS['border']}"},
    )


def format_map_metric(metric, value):
    if pd.isna(value):
        return "n/a"
    if metric == "MAPE":
        return f"{value * 100:.1f}%"
    if metric == "MeanLoadFactor":
        return f"{value * 100:.1f}%"
    units = {
        "MeanBurnoff": "kg",
        "RMSE": "kg",
        "MeanTripTime": "sec",
        "MeanDistanceKm": "km",
    }
    unit = units.get(metric, "")
    return f"{value:,.1f} {unit}".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PRECOMPUTE CHART DATA
# ═══════════════════════════════════════════════════════════════════════════════

best_model_row = model_res.loc[model_res["RMSE"].idxmin()]
total_flights  = len(df_eng)
n_routes       = df_eng["ScheduledRoute"].nunique()
mean_burnoff   = df_eng["Burnoff"].mean()
airlines       = df_eng["Carrier"].nunique()
annual_avoidable_cost = float(metric_lookup(cost_data, "Annual avoidable operational cost (EUR)") or 0)
annual_avoidable_co2 = float(metric_lookup(cost_data, "Annual avoidable operational CO2 (tonnes)") or 0)
annual_total_value = float(metric_lookup(cost_data, "Total annual value (EUR)") or 0)
top_route = route_opp.sort_values("OpportunityScore", ascending=False).iloc[0] if len(route_opp) else None

# ═══════════════════════════════════════════════════════════════════════════════
# 4. LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

navbar = dbc.Navbar(
    dbc.Container([
        html.Span("✈️", style={"fontSize": "1.4rem", "marginRight": "8px"}),
        dbc.NavbarBrand("Ryanair Burnoff Prediction Dashboard",
                        style={"fontWeight": "700", "color": "white", "fontSize": "1.1rem"}),
        html.Span("IE University · Group 4", className="ms-auto",
                  style={"color": "rgba(255,255,255,0.7)", "fontSize": "0.85rem"}),
    ], fluid=True),
    color=COLORS["primary"], dark=True, className="mb-4 shadow-sm",
)

# ─── KPI Row ──────────────────────────────────────────────────────────────────
kpi_row = dbc.Row([
    dbc.Col(kpi_card("Total Flights",   f"{total_flights:,}", "training set",
                     COLORS["primary"], "📊"), md=2),
    dbc.Col(kpi_card("Routes",          f"{n_routes:,}", "unique O-D pairs",
                     COLORS["secondary"], "🗺️"), md=2),
    dbc.Col(kpi_card("Mean Burnoff",    f"{mean_burnoff:,.0f} kg", "per flight",
                     COLORS["warning"], "⛽"), md=2),
    dbc.Col(kpi_card("Best Model",      best_model_row["Model"],
                     f"RMSE={best_model_row['RMSE']:,.0f} kg",
                     COLORS["success"], "🏆"), md=3),
    dbc.Col(kpi_card("Best MAPE",       f"{best_model_row['MAPE']:.2f}%",
                     f"R²={best_model_row['R2']:.4f}",
                     COLORS["danger"], "🎯"), md=3),
], className="mb-4 g-3")

# ─── Map Panel ────────────────────────────────────────────────────────────────
map_controls = dbc.Row([
    dbc.Col([
        html.Label("Colour metric", className="fw-semibold small"),
        dcc.Dropdown(
            id="map-metric",
            options=[
                {"label": "Mean Burnoff (kg)",        "value": "MeanBurnoff"},
                {"label": "Prediction RMSE (kg)",     "value": "RMSE"},
                {"label": "MAPE (%)",                 "value": "MAPE"},
                {"label": "Mean Trip Time (min)",     "value": "MeanTripTime"},
                {"label": "Mean Load Factor",         "value": "MeanLoadFactor"},
                {"label": "Route Distance (km)",      "value": "MeanDistanceKm"},
            ],
            value="MeanBurnoff",
            clearable=False,
            style={"fontSize": "13px"},
        ),
    ], md=3),
    dbc.Col([
        html.Label("Min flight count", className="fw-semibold small"),
        dcc.Slider(id="map-min-flights", min=1, max=50, step=1, value=5,
                   marks={1: "1", 10: "10", 25: "25", 50: "50"},
                   tooltip={"always_visible": False}),
    ], md=4),
    dbc.Col([
        html.Label("Aircraft type filter", className="fw-semibold small"),
        dcc.Checklist(
            id="map-aircraft-filter",
            options=[{"label": f" {t}", "value": t} for t in df_eng["AircraftTypeGroup"].unique()],
            value=list(df_eng["AircraftTypeGroup"].unique()),
            inline=True, className="small",
        ),
    ], md=5),
], className="mb-3 g-2")

map_panel = dbc.Card([
    dbc.CardBody([
        section_header("Flight Route Map",
                       "Each arc represents a route. Hover for details. Circle size = flight volume."),
        dbc.Alert(
            "Ryanair operates 5,000+ routes across Europe. This map visualises per-route metrics — "
            "select a metric to colour airports by burnoff, prediction error, or operational load. "
            "Click any airport to drill down into its routes.",
            color="info", className="small py-2 mb-3",
        ),
        map_controls,
        dcc.Loading(dcc.Graph(id="route-map", style={"height": "560px"}), type="circle"),
        html.Div(id="map-finding", className="mt-2"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="route-detail-chart", style={"height": "300px"}),
                html.Div(id="route-detail-caption", className="mt-2"),
            ], md=12),
        ], className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── EDA Panel ───────────────────────────────────────────────────────────────
eda_panel = dbc.Card([
    dbc.CardBody([
        section_header("Exploratory Data Analysis",
                       "Understand the distribution and drivers of fuel burnoff."),
        dbc.Row([
            dbc.Col([
                html.Label("Chart type", className="fw-semibold small"),
                dcc.Dropdown(
                    id="eda-chart-type",
                    options=[
                        {"label": "Burnoff Distribution",         "value": "dist"},
                        {"label": "Burnoff by Aircraft Type",     "value": "aircraft"},
                        {"label": "Burnoff by Season",            "value": "season"},
                        {"label": "Burnoff vs Trip Time",         "value": "triptime"},
                        {"label": "Burnoff vs Load Factor",       "value": "load"},
                        {"label": "Burnoff vs Route Distance",    "value": "distance"},
                        {"label": "Monthly Flight Volume",        "value": "monthly"},
                        {"label": "Correlation Heatmap",          "value": "corr"},
                    ],
                    value="dist", clearable=False, style={"fontSize": "13px"},
                ),
            ], md=4),
            dbc.Col([
                html.Label("Aircraft type", className="fw-semibold small"),
                dcc.Checklist(
                    id="eda-aircraft-filter",
                    options=[{"label": f" {t}", "value": t} for t in df_eng["AircraftTypeGroup"].unique()],
                    value=list(df_eng["AircraftTypeGroup"].unique()),
                    inline=True, className="small",
                ),
            ], md=8),
        ], className="mb-3 g-2"),
        dcc.Loading(dcc.Graph(id="eda-chart", style={"height": "450px"}), type="circle"),
        html.Div(id="eda-finding", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Model Performance Panel ─────────────────────────────────────────────────
model_panel = dbc.Card([
    dbc.CardBody([
        section_header("Model Performance",
                       "Compare the trained models on held-out validation data."),
        dbc.Alert([
            html.Strong("Key finding: "),
            f"LightGBM and XGBoost are effectively tied (RMSE ~218 kg, R²=0.9869), both predicting "
            f"fuel within ~4.7% of actual on a typical {mean_burnoff:,.0f} kg flight. "
            f"Ridge Regression lags at 265 kg RMSE due to multicollinearity in the feature set "
            f"(PlannedTOW, ZFW, and WeightDelta are linearly dependent). "
            f"All four models massively outperform the naive baseline (1,905 kg RMSE).",
        ], color="light", className="small py-2 mb-3"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id="model-rmse-chart",  style={"height": "350px"}),
                html.Div(id="model-rmse-caption", className="mt-2"),
            ], md=6),
            dbc.Col([
                dcc.Graph(id="model-mape-chart",  style={"height": "350px"}),
                html.Div(id="model-mape-caption", className="mt-2"),
            ], md=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Select model for scatter", className="fw-semibold small"),
                dcc.Dropdown(
                    id="scatter-model-select",
                    options=[{"label": m, "value": m} for m in model_res["Model"]],
                    value=PRIMARY_MODEL, clearable=False, style={"fontSize": "13px"},
                ),
            ], md=3),
        ], className="mb-2"),
        dcc.Loading(dcc.Graph(id="pred-vs-actual", style={"height": "420px"}), type="circle"),
        html.Div(id="pred-caption", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Feature Importance Panel ────────────────────────────────────────────────
feat_panel = dbc.Card([
    dbc.CardBody([
        section_header("Feature Importance",
                       "Top predictors driving fuel burnoff — supporting Hypothesis H1."),
        dbc.Alert([
            html.Strong("Key finding: "),
            "The top 5 features are all physics-driven: TeledyneRampWeight (total pre-departure weight), "
            "WeightDelta (planned fuel load), RouteDistanceKm, PlannedTripTime, and LoadFactor. "
            "This confirms H1: the model learns the fundamental mass-distance-energy relationship. "
            "The engineered interaction features (TypeTripTime, TypeWeight) rank in the top 9, "
            "validating that different aircraft types have distinct fuel dynamics.",
        ], color="light", className="small py-2 mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Model", className="fw-semibold small"),
                dcc.Dropdown(
                    id="fi-model-select",
                    options=(
                        [{"label": "All Models (side-by-side)", "value": "__all__"}]
                        + [{"label": m, "value": m} for m in feat_imp["Model"].unique()]
                    ),
                    value=DEFAULT_FI_MODEL, clearable=False, style={"fontSize": "13px"},
                ),
            ], md=3),
            dbc.Col([
                html.Label("Top N features", className="fw-semibold small"),
                dcc.Slider(id="fi-top-n", min=5, max=25, step=1, value=15,
                           marks={5: "5", 15: "15", 25: "25"},
                           tooltip={"always_visible": False}),
            ], md=5),
        ], className="mb-3 g-2"),
        dcc.Loading(dcc.Graph(id="fi-chart", style={"height": "700px"}), type="circle"),
        html.Div(id="fi-caption", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Story / Hypothesis Panel ────────────────────────────────────────────────
def hyp_badge(label, status, color):
    return dbc.Badge(f"{label} → {status}", color=color, className="me-2 mb-1 p-2")

hyp_cards = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader("H1 – Trip Time & Weight are top predictors", className="fw-semibold small"),
        dbc.CardBody([
            hyp_badge("H1", "✅ CONFIRMED", "success"),
            html.P("PlannedTripTime and PlannedTOW rank #1 and #2 in both LightGBM and "
                   "Random Forest importances, consistent with flight-physics reasoning.",
                   className="small text-muted mb-0"),
        ]),
    ], className="shadow-sm h-100"), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader("H2 – XGBoost has lowest RMSE among baseline models", className="fw-semibold small"),
        dbc.CardBody([
            hyp_badge("H2", "✅ CONFIRMED (but LightGBM is better)", "warning"),
            html.P("XGBoost outperforms Ridge and Random Forest. LightGBM + interactions "
                   "achieves the best RMSE overall by leveraging native categorical splits.",
                   className="small text-muted mb-0"),
        ]),
    ], className="shadow-sm h-100"), md=6),
], className="mb-3 g-3")

hyp_cards2 = dbc.Row([
    dbc.Col(dbc.Card([
        dbc.CardHeader("H3 – ML models outperform the naive baseline", className="fw-semibold small"),
        dbc.CardBody([
            hyp_badge("H3", "✅ CONFIRMED", "success"),
            html.P("All four ML models improve substantially over a mean-per-aircraft-type "
                   "baseline, validating the modelling exercise.",
                   className="small text-muted mb-0"),
        ]),
    ], className="shadow-sm h-100"), md=6),
    dbc.Col(dbc.Card([
        dbc.CardHeader("H4 – Max fleet has higher prediction error than NG", className="fw-semibold small"),
        dbc.CardBody([
            hyp_badge("H4", "⚠️ PARTIAL – see error-by-aircraft chart", "secondary"),
            html.P("MAX errors are elevated on some routes. However, the small MAX sample "
                   "(~1.5k flights) limits statistical power. Airbus shows highest variance, "
                   "likely due to operational diversity.",
                   className="small text-muted mb-0"),
        ]),
    ], className="shadow-sm h-100"), md=6),
], className="mb-3 g-3")

story_panel = dbc.Card([
    dbc.CardBody([
        section_header("Storyline & Hypothesis Analysis",
                       "Beyond accuracy: understanding *why* the model works and where it fails."),
        hyp_cards,
        hyp_cards2,
        dbc.Row([
            dbc.Col([
                html.Label("Story chart", className="fw-semibold small"),
                dcc.Dropdown(
                    id="story-chart-select",
                    options=[
                        {"label": "Error by Aircraft Type",    "value": "aircraft"},
                        {"label": "Error by Season",           "value": "season"},
                        {"label": "Residual by Load Factor",   "value": "load"},
                        {"label": "Error vs Route Distance",   "value": "distance"},
                        {"label": "Temporal Bias (monthly)",   "value": "temporal"},
                        {"label": "Top Routes by RMSE",        "value": "routes"},
                    ],
                    value="aircraft", clearable=False, style={"fontSize": "13px"},
                ),
            ], md=4),
        ], className="mb-3"),
        dcc.Loading(dcc.Graph(id="story-chart", style={"height": "420px"}), type="circle"),
        html.Div(id="story-finding", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Sustainability Panel ────────────────────────────────────────────────────
_mean_structural_co2 = flight_analytics["StructuralCO2_kg"].mean()
_mean_avoidable_co2 = flight_analytics["AvoidableCO2_kg"].mean()
_avoidable_share = (
    100 * flight_analytics["AvoidableCO2_kg"].sum()
    / max(flight_analytics["ActualCO2_kg"].sum() if "ActualCO2_kg" in flight_analytics.columns else (flight_analytics["ActualBurn_kg"].sum() * CO2_PER_KG_FUEL), 1)
)
_top_avoidable_route = (
    route_opp.sort_values("AnnualAvoidableCO2_t", ascending=False).iloc[0]["ScheduledRoute"]
    if len(route_opp) else "N/A"
)

sustainability_panel = dbc.Card([
    dbc.CardBody([
        section_header("Sustainability & Emissions Hotspots",
                       "Use model-predicted burn as structural emissions and residual over-burn as avoidable emissions."),
        dbc.Alert([
            html.Strong("Interpretation guardrail: "),
            "Predicted burn is an expected operational benchmark, not a causal optimum. "
            "Avoidable CO2 highlights segments that burn more than comparable flights would structurally suggest; "
            "it should be used to target investigation, not to automate operational decisions.",
        ], color="light", className="small py-2 mb-3"),
        dbc.Row([
            dbc.Col(kpi_card("Structural CO2 / Flight", f"{_mean_structural_co2:,.0f} kg",
                             "expected emissions from route, load and fleet mix", COLORS["primary"], ""), md=3),
            dbc.Col(kpi_card("Avoidable CO2 / Flight", f"{_mean_avoidable_co2:,.0f} kg",
                             "positive excess burn only", COLORS["danger"], ""), md=3),
            dbc.Col(kpi_card("Avoidable CO2 / Year", f"{annual_avoidable_co2/1e3:.0f}k tonnes",
                             "annualised from validation flights", COLORS["warning"], ""), md=3),
            dbc.Col(kpi_card("Top Emissions Hotspot", _top_avoidable_route,
                             "highest annual avoidable CO2 route", COLORS["secondary"], ""), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Chart", className="fw-semibold small"),
                dcc.Dropdown(
                    id="co2-chart-select",
                    options=[
                        {"label": "Structural vs Avoidable CO2 by Aircraft", "value": "struct_vs_avoid"},
                        {"label": "Top Routes by Avoidable CO2", "value": "routes"},
                        {"label": "Hotspots: Structural vs Avoidable", "value": "hotspots"},
                        {"label": "Intensity Gap by Load Factor Band", "value": "loadfactor"},
                    ],
                    value="struct_vs_avoid", clearable=False, style={"fontSize": "13px"},
                ),
            ], md=4),
        ], className="mb-3"),
        dcc.Loading(dcc.Graph(id="co2-chart", style={"height": "450px"}), type="circle"),
        html.Div(id="co2-finding", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Business Value / What-If Panel ─────────────────────────────────────────
_buffer_reduction = float(metric_lookup(cost_data, "Buffer reduction per flight (kg)") or 0)
_annual_buffer_savings = float(metric_lookup(cost_data, "Annual buffer cost savings (EUR)") or 0)
_annual_carry_penalty = float(metric_lookup(cost_data, "Annual planning carry penalty (EUR)") or 0)
_aircraft_types = df_eng["AircraftTypeGroup"].unique().tolist()
_top_routes = df_eng["ScheduledRoute"].value_counts().head(50).index.tolist()

business_panel = dbc.Card([
    dbc.CardBody([
        section_header("Business Value & Opportunity Prioritisation",
                       "Rank inefficiencies by annual value, recurrence, and confidence instead of summarising average burn."),
        dbc.Alert([
            html.Strong("Decision logic: "),
            "The best business opportunities are not always the flights with the biggest single excess burn. "
            "This tab combines recurrent over-burn, traffic volume, and annualised value to identify the segments most worth operational intervention.",
        ], color="light", className="small py-2 mb-3"),
        dbc.Row([
            dbc.Col(kpi_card("Annual Avoidable Fuel Cost", f"EUR {annual_avoidable_cost/1e6:.1f}M",
                             "positive excess burn only", COLORS["primary"], ""), md=3),
            dbc.Col(kpi_card("Annual Carry Penalty", f"EUR {_annual_carry_penalty/1e6:.1f}M",
                             "planning conservatism proxy", COLORS["warning"], ""), md=3),
            dbc.Col(kpi_card("Total Annual Value", f"EUR {annual_total_value/1e6:.1f}M",
                             "fuel + carry penalty + ETS exposure", COLORS["success"], ""), md=3),
            dbc.Col(kpi_card("Top Route Opportunity", top_route["ScheduledRoute"] if top_route is not None else "N/A",
                             f"score {top_route['OpportunityScore']:,.0f}" if top_route is not None else "", COLORS["danger"], ""), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Chart", className="fw-semibold small"),
                dcc.Dropdown(
                    id="business-chart-select",
                    options=[
                        {"label": "Opportunity Matrix", "value": "matrix"},
                        {"label": "Top Segments by Annual Value", "value": "segments"},
                        {"label": "Intervention Scenarios", "value": "scenarios"},
                    ],
                    value="matrix", clearable=False, style={"fontSize": "13px"},
                ),
            ], md=4),
        ], className="mb-3"),
        dcc.Loading(dcc.Graph(id="business-chart", style={"height": "450px"}), type="circle"),
        html.Div(id="business-finding", className="mt-2"),
        html.Hr(),
        html.H5("What-If Scenario Simulator", style={"fontWeight": "700", "marginBottom": "12px"}),
        html.P("Adjust parameters to estimate expected structural burn, then compare the route to historical fleet performance.",
               className="text-muted small mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Aircraft Type", className="fw-semibold small"),
                dcc.Dropdown(id="whatif-aircraft", options=[{"label": t, "value": t} for t in _aircraft_types],
                             value=_aircraft_types[0] if _aircraft_types else None, clearable=False, style={"fontSize": "13px"}),
            ], md=2),
            dbc.Col([
                html.Label("Route", className="fw-semibold small"),
                dcc.Dropdown(id="whatif-route", options=[{"label": r, "value": r} for r in _top_routes],
                             value=_top_routes[0] if _top_routes else "", clearable=False, style={"fontSize": "13px"}),
            ], md=3),
            dbc.Col([
                html.Label("Load Factor", className="fw-semibold small"),
                dcc.Slider(id="whatif-load", min=0.3, max=1.0, step=0.05, value=0.85,
                           marks={0.3: "30%", 0.5: "50%", 0.7: "70%", 0.85: "85%", 1.0: "100%"},
                           tooltip={"always_visible": False}),
            ], md=3),
            dbc.Col([
                html.Label("Departure Hour", className="fw-semibold small"),
                dcc.Slider(id="whatif-hour", min=5, max=23, step=1, value=10,
                           marks={5: "5am", 10: "10am", 15: "3pm", 20: "8pm", 23: "11pm"},
                           tooltip={"always_visible": False}),
            ], md=2),
            dbc.Col([
                html.Label("Season", className="fw-semibold small"),
                dcc.Dropdown(id="whatif-season",
                             options=[{"label": s, "value": s} for s in ["Winter", "Spring", "Summer", "Fall"]],
                             value="Summer", clearable=False, style={"fontSize": "13px"}),
            ], md=2),
        ], className="mb-3 g-2"),
        dbc.Row([
            dbc.Col(html.Div(id="whatif-result", className="mt-2"), md=12),
        ]),
    ])
], className="shadow-sm mb-4")

# ─── Anomaly Detection Panel ────────────────────────────────────────────────
_n_anomalies = int(flight_analytics["IsAnomaly"].sum())
_pct_anomalies = _n_anomalies / len(flight_analytics) * 100 if len(flight_analytics) else 0
_overburn_anoms = int(flight_analytics["IsOverburnAnomaly"].sum()) if "IsOverburnAnomaly" in flight_analytics.columns else int((flight_analytics["ExcessBurn_kg"] > 0).sum())
_underburn_anoms = int(flight_analytics["IsUnderburnAnomaly"].sum()) if "IsUnderburnAnomaly" in flight_analytics.columns else max(_n_anomalies - _overburn_anoms, 0)
_persistent_segments = int(anomaly_monitor["PersistenceFlag"].sum()) if "PersistenceFlag" in anomaly_monitor.columns else 0

anomaly_panel = dbc.Card([
    dbc.CardBody([
        section_header("Anomalies & Recurring Operational Risk",
                       "Escalate from one-off outliers to repeatable route, fleet, and airport-hour anomalies."),
        dbc.Alert([
            html.Strong("Interpretation guardrail: "),
            "An anomaly is unexplained relative to the model benchmark, not automatically an operational mistake. "
            "Missing weather, ATC, de-icing, taxi, and maintenance variables mean this tab should prioritize investigations, not assign blame.",
        ], color="light", className="small py-2 mb-3"),
        dbc.Row([
            dbc.Col(kpi_card("Flagged Flights", f"{_n_anomalies}",
                             f"{_pct_anomalies:.1f}% of validation flights", COLORS["danger"], ""), md=3),
            dbc.Col(kpi_card("Over-burn Anomalies", f"{_overburn_anoms}",
                             "actual burn above segment expectation", COLORS["warning"], ""), md=3),
            dbc.Col(kpi_card("Under-burn Anomalies", f"{_underburn_anoms}",
                             "favourable or noisy outcomes", COLORS["primary"], ""), md=3),
            dbc.Col(kpi_card("Persistent Risk Segments", f"{_persistent_segments}",
                             "repeating anomaly hotspots", COLORS["secondary"], ""), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Anomaly chart", className="fw-semibold small"),
                dcc.Dropdown(
                    id="anomaly-chart-select",
                    options=[
                        {"label": "Segment-Normalized Residual Distribution", "value": "dist"},
                        {"label": "Recurring Risk by Segment Type", "value": "segments"},
                        {"label": "Airport × Hour Heatmap", "value": "airport_hour"},
                        {"label": "Top Persistent Anomaly Segments", "value": "routes"},
                    ],
                    value="dist", clearable=False, style={"fontSize": "13px"},
                ),
            ], md=4),
        ], className="mb-3"),
        dcc.Loading(dcc.Graph(id="anomaly-chart", style={"height": "450px"}), type="circle"),
        html.Div(id="anomaly-finding", className="mt-2"),
    ])
], className="shadow-sm mb-4")

# ─── Full layout ─────────────────────────────────────────────────────────────
app.layout = dbc.Container([
    navbar,
    kpi_row,
    dbc.Tabs([
        dbc.Tab(label="🗺️ Route Map",         tab_id="tab-map"),
        dbc.Tab(label="📊 EDA",               tab_id="tab-eda"),
        dbc.Tab(label="🤖 Model Performance", tab_id="tab-model"),
        dbc.Tab(label="🔍 Feature Importance",tab_id="tab-feat"),
        dbc.Tab(label="📖 Storyline",         tab_id="tab-story"),
        dbc.Tab(label="🌱 Sustainability",    tab_id="tab-co2"),
        dbc.Tab(label="💰 Business Value",    tab_id="tab-business"),
        dbc.Tab(label="⚠️ Anomalies",         tab_id="tab-anomaly"),
    ], id="main-tabs", active_tab="tab-map", className="mb-4"),
    html.Div(id="tab-content"),
], fluid=True, style={"background": COLORS["light_bg"], "minHeight": "100vh", "padding": "0 24px 40px"})


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("tab-content", "children"), Input("main-tabs", "active_tab"))
def render_tab(tab):
    if tab == "tab-map":      return map_panel
    if tab == "tab-eda":      return eda_panel
    if tab == "tab-model":    return model_panel
    if tab == "tab-feat":     return feat_panel
    if tab == "tab-story":    return story_panel
    if tab == "tab-co2":      return sustainability_panel
    if tab == "tab-business": return business_panel
    if tab == "tab-anomaly":  return anomaly_panel
    return html.Div()


# ─── Map ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("route-map", "figure"),
    Output("map-finding", "children"),
    Input("map-metric", "value"),
    Input("map-min-flights", "value"),
    Input("map-aircraft-filter", "value"),
)
def update_map(metric, min_flights, aircraft_types):
    # Filter df_eng by aircraft type to get relevant routes
    df_filt = df_eng[df_eng["AircraftTypeGroup"].isin(aircraft_types)]
    relevant_routes = set(df_filt["ScheduledRoute"].unique())
    re = route_err[route_err["ScheduledRoute"].isin(relevant_routes)]
    re = re[re["FlightCount"] >= min_flights].dropna(subset=["OrgLat","OrgLon","DstLat","DstLon"])

    fig = go.Figure()

    # Arc lines (limit to top 500 routes for performance)
    re_arcs = re.nlargest(500, "FlightCount") if len(re) > 500 else re
    for _, row in re_arcs.iterrows():
        fig.add_trace(go.Scattergeo(
            lon=[row["OrgLon"], (row["OrgLon"] + row["DstLon"]) / 2, row["DstLon"]],
            lat=[row["OrgLat"], (row["OrgLat"] + row["DstLat"]) / 2 + 2, row["DstLat"]],
            mode="lines",
            line=dict(width=0.8, color="rgba(37,99,235,0.25)"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Origin airport circles
    agg_org = re.groupby(["Origin", "OrgLat", "OrgLon", "OrgCity"])[metric].mean().reset_index()
    fig.add_trace(go.Scattergeo(
        lon=agg_org["OrgLon"],
        lat=agg_org["OrgLat"],
        mode="markers",
        marker=dict(
            size=re.groupby("Origin")["FlightCount"].sum().reindex(agg_org["Origin"]).values ** 0.45 * 2,
            color=agg_org[metric],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text=metric, side="right"), thickness=12),
            line=dict(color="white", width=0.5),
            opacity=0.85,
        ),
        text=agg_org["OrgCity"] + "<br>" + agg_org["Origin"] + "<br>" + metric + ": " + agg_org[metric].round(1).astype(str),
        hovertemplate="%{text}<extra></extra>",
        name="Airports",
    ))

    fig.update_geos(
        scope="europe",
        showland=True, landcolor="#EFF3F6",
        showocean=True, oceancolor="#D4E8F5",
        showcoastlines=True, coastlinecolor="#B0BEC5",
        showcountries=True, countrycolor="#CFD8DC",
        projection_type="natural earth",
        center=dict(lat=50, lon=10),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # Dynamic finding
    metric_labels = {"MeanBurnoff": "fuel burn", "RMSE": "prediction error", "MAPE": "relative error",
                     "MeanTripTime": "trip time", "MeanLoadFactor": "load factor", "MeanDistanceKm": "distance"}
    top_airport = agg_org.nlargest(1, metric).iloc[0] if len(agg_org) > 0 else None
    busiest_route = re.nlargest(1, "FlightCount").iloc[0] if len(re) > 0 else None
    finding_div = html.Div()
    if top_airport is not None:
        metric_text = metric_labels.get(metric, metric)
        result = (
            f"With the current filters, {top_airport['OrgCity']} ({top_airport['Origin']}) has the highest average "
            f"{metric_text} at {format_map_metric(metric, top_airport[metric])}."
        )
        if busiest_route is not None:
            result += (
                f" The busiest visible route is {busiest_route['ScheduledRoute']} with "
                f"{int(busiest_route['FlightCount'])} flights."
            )
        finding_div = chart_caption(
            what=(
                f"This map aggregates {len(re):,} visible routes into {len(agg_org):,} origin airports. "
                "Curved lines show network connectivity, while airport markers summarize the selected route metric."
            ),
            how=(
                "Marker size is proportional to traffic volume and marker color reflects the average of the selected metric. "
                "Use the filters to separate network scale from fuel intensity or model error, then click an airport to expose the route mix behind the aggregate."
            ),
            result=result,
            why=(
                "Airports that remain both large and dark after filtering combine scale with intensity, so they are the places where network planning, station operations, or model improvements will create the largest aggregate effect."
            ),
            action=(
                "Prioritize the airports that stay prominent on both volume and intensity filters, then drill into the specific routes driving that pattern before assigning an operational review or modeling fix."
            ),
        )

    return fig, finding_div


@app.callback(
    Output("route-detail-chart", "figure"),
    Output("route-detail-caption", "children"),
    Input("route-map", "clickData"),
)
def route_detail(click_data):
    if not click_data:
        # Default: top 15 busiest routes
        top = route_err.nlargest(15, "FlightCount")
        fig = px.bar(
            top.sort_values("MeanBurnoff"),
            x="MeanBurnoff", y="ScheduledRoute",
            orientation="h",
            color="MeanBurnoff", color_continuous_scale="Viridis",
            labels={"MeanBurnoff": "Mean Burnoff (kg)", "ScheduledRoute": ""},
            title="Top 15 Busiest Routes – Mean Burnoff",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0))
        top_route = top.loc[top["MeanBurnoff"].idxmax()]
        lowest_route = top.loc[top["MeanBurnoff"].idxmin()]
        caption = chart_caption(
            what=(
                "This default view zooms in on the 15 busiest routes in the dataset and compares their average fuel burn."
            ),
            how=(
                "Longer bars mean more fuel burned per flight. Because the chart is restricted to busy routes, it "
                "highlights operationally important lanes rather than obscure low-sample cases."
            ),
            result=(
                f"Among the busiest routes, {top_route['ScheduledRoute']} has the highest mean burn at "
                f"{top_route['MeanBurnoff']:,.0f} kg per flight, while {lowest_route['ScheduledRoute']} sits lowest "
                f"at {lowest_route['MeanBurnoff']:,.0f} kg."
            ),
            why=(
                "This is effectively a network demand map in bar-chart form: it identifies where absolute fuel consumption is concentrated, which is different from where relative inefficiency is concentrated."
            ),
            action=(
                "Use this view as the starting shortlist for route-level fuel audits, planning-buffer reviews, and high-value savings initiatives because improvements here scale across a large number of flights."
            ),
        )
        return fig, caption

    city = click_data["points"][0].get("text", "").split("<br>")[0]
    nearby = route_err[
        (route_err["OrgCity"] == city) | (route_err["DstCity"] == city)
    ].nlargest(12, "FlightCount")
    if nearby.empty:
        return go.Figure().add_annotation(text="No data for this airport", showarrow=False), html.Div()
    fig = px.bar(
        nearby.sort_values("MeanBurnoff"),
        x="MeanBurnoff", y="ScheduledRoute",
        orientation="h", color="MAPE",
        color_continuous_scale="RdYlGn_r",
        labels={"MeanBurnoff": "Mean Burnoff (kg)", "ScheduledRoute": "Route", "MAPE": "MAPE"},
        title=f"Routes through {city}",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0))
    top_burn = nearby.loc[nearby["MeanBurnoff"].idxmax()]
    worst_mape = nearby.loc[nearby["MAPE"].idxmax()]
    caption = chart_caption(
        what=(
            f"This drill-down isolates the busiest routes touching {city} and shows both how fuel-hungry those routes are "
            "and how easy or hard they are for the model to predict."
        ),
        how=(
            "Bar length represents average burnoff per flight, while color represents relative model error. "
            "Routes that are simultaneously long and red carry both operational importance and elevated analytical uncertainty."
        ),
        result=(
            f"{top_burn['ScheduledRoute']} is the highest-burn route in this airport slice at "
            f"{top_burn['MeanBurnoff']:,.0f} kg per flight, while {worst_mape['ScheduledRoute']} has the highest "
            f"percentage miss at {worst_mape['MAPE'] * 100:.1f}%."
        ),
        why=(
            "This separates airports that are fuel-intensive because they serve structurally long sectors from airports where a subset of routes may also suffer from inconsistent operating conditions or missing explanatory variables."
        ),
        action=(
            "For routes that are both high-burn and high-error, review dispatch buffers, weather sensitivity, and route-specific operating constraints before changing targets or model assumptions."
        ),
    )
    return fig, caption


# ─── EDA ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("eda-chart", "figure"),
    Output("eda-finding", "children"),
    Input("eda-chart-type", "value"),
    Input("eda-aircraft-filter", "value"),
)
def update_eda(chart_type, aircraft_types):
    dff = df_eng[df_eng["AircraftTypeGroup"].isin(aircraft_types)]
    template = PLOTLY_TEMPLATE
    what = how = result = why = action = ""

    if chart_type == "dist":
        fig = px.histogram(dff, x="Burnoff", nbins=80, color_discrete_sequence=[COLORS["primary"]],
                           title="Burnoff Distribution", labels={"Burnoff": "Burnoff (kg)"})
        fig.add_vline(x=dff["Burnoff"].mean(), line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {dff['Burnoff'].mean():,.0f} kg")
        high_burn_share = (dff["Burnoff"] >= 10000).mean() * 100
        what = "This histogram shows how flight-level fuel burn is distributed across the selected aircraft types."
        how = "The x-axis is fuel burned per flight and the y-axis is the number of flights. The dashed red line marks the average. If the bars stretch far to the right, it means a small number of flights burn much more fuel than the rest."
        result = (
            f"The distribution is clearly right-skewed: the mean is {dff['Burnoff'].mean():,.0f} kg while the median is "
            f"{dff['Burnoff'].median():,.0f} kg. Only {high_burn_share:.1f}% of flights exceed 10,000 kg, but those long-haul outliers pull the average upward."
        )
        why = "Business-wise, this means the network has a heavy tail: a relatively small subset of sectors drives a disproportionate share of total fuel use, emissions, and budget exposure."
        action = "Segment the highest-burn routes into a dedicated watchlist and evaluate them separately for savings, emissions, and target-setting instead of benchmarking them against the short-haul core."
    elif chart_type == "aircraft":
        fig = px.box(dff, x="AircraftTypeGroup", y="Burnoff", color="AircraftTypeGroup",
                     title="Burnoff by Aircraft Type", points=False)
        means = dff.groupby("AircraftTypeGroup")["Burnoff"].mean()
        counts = dff["AircraftTypeGroup"].value_counts()
        iqr = dff.groupby("AircraftTypeGroup")["Burnoff"].quantile(0.75) - dff.groupby("AircraftTypeGroup")["Burnoff"].quantile(0.25)
        top_type = means.idxmax()
        widest_type = iqr.idxmax()
        what = "This boxplot compares the burnoff distribution for each aircraft family in the selected slice."
        how = "The line inside each box is the middle value, the box shows where the middle half of flights sit, and the whiskers show the broader range. Higher boxes mean that aircraft family usually burns more fuel. Taller boxes mean it is used on a wider mix of routes."
        result = (
            f"{top_type} has the highest mean burn at {means[top_type]:,.0f} kg per flight. "
            f"{widest_type} also has the widest middle spread of flights, and NG dominates the sample with {counts.get('NG', 0):,} flights, which helps explain why its range is so broad."
        )
        why = "The business implication is that raw fleet comparisons can be misleading because route assignment and mission mix influence fuel burn almost as much as the aircraft family itself."
        action = "Use route-normalized fleet KPIs before making fleet-efficiency claims or operational decisions, especially when comparing NG, MAX, and Airbus performance."
    elif chart_type == "season":
        order = ["Winter", "Spring", "Summer", "Fall"]
        fig = px.violin(dff, x="Season", y="Burnoff", color="Season",
                        category_orders={"Season": order}, box=True,
                        title="Burnoff Distribution by Season")
        medians = dff.groupby("Season")["Burnoff"].median().reindex(order).dropna()
        top_season = medians.idxmax()
        low_season = medians.idxmin()
        spread_pct = 100 * (medians.max() - medians.min()) / max(medians.min(), 1)
        what = "This violin plot compares the full seasonal burnoff distribution rather than just seasonal averages."
        how = "Wider sections mean more flights at that burn level. The small box inside each shape shows the typical range, so you can compare both the usual burn and how spread out each season is."
        result = (
            f"The seasonal effect is modest: {top_season} has the highest median burnoff and {low_season} the lowest, "
            f"with only a {spread_pct:.1f}% gap between them."
        )
        why = "Season affects the baseline, but it is a second-order business driver compared with route length, trip time, and aircraft weight."
        action = "Treat season as a planning adjustment, not as the main segmentation strategy; prioritize route, fleet, and weight-related levers before building seasonal operating policies."
    elif chart_type == "triptime":
        sample = dff.sample(min(8000, len(dff)), random_state=1)
        fig = px.scatter(sample, x="PlannedTripTime", y="Burnoff",
                         color="AircraftTypeGroup", opacity=0.4, size_max=4,
                         trendline="ols",
                         labels={"PlannedTripTime": "Trip Time (s)", "Burnoff": "Burnoff (kg)"},
                         title="Burnoff vs Planned Trip Time")
        corr = dff["PlannedTripTime"].corr(dff["Burnoff"])
        slope_per_hour = np.polyfit(dff["PlannedTripTime"], dff["Burnoff"], 1)[0] * 3600
        what = "This scatterplot relates planned trip time to actual burnoff for a large flight sample."
        how = "Each point is one flight. Moving right means a longer flight, and moving up means more fuel burned. The trend line shows the average pattern across all those flights."
        result = (
            f"Trip time is one of the clearest drivers in the dataset: it moves very closely with burnoff ({corr:.2f} on a scale where 1.00 would mean a near-perfect match), "
            f"and the fitted line suggests roughly {slope_per_hour:,.0f} kg of extra fuel for each extra hour of planned flying time."
        )
        why = "This is the strongest operational physics relationship in the dashboard, which is why trip-time assumptions propagate directly into planning accuracy, savings estimates, and emissions forecasts."
        action = "Protect trip-time forecast quality as a critical planning input and investigate any schedules or route files that systematically distort planned airborne time."
    elif chart_type == "load":
        # Filter to passenger flights for meaningful load factor analysis
        sample = dff[dff["TotalPassengers"] > 0].sample(min(8000, len(dff[dff["TotalPassengers"] > 0])), random_state=2)
        fig = px.scatter(sample, x="LoadFactor", y="Burnoff",
                         color="AircraftTypeGroup", opacity=0.4, trendline="ols",
                         labels={"LoadFactor": "Load Factor", "Burnoff": "Burnoff (kg)"},
                         title="Burnoff vs Load Factor (passenger flights only)")
        passenger = dff[dff["TotalPassengers"] > 0]
        corr = passenger["LoadFactor"].corr(passenger["Burnoff"])
        low_load = passenger[passenger["LoadFactor"] <= passenger["LoadFactor"].quantile(0.2)]["Burnoff"].mean()
        high_load = passenger[passenger["LoadFactor"] >= passenger["LoadFactor"].quantile(0.8)]["Burnoff"].mean()
        what = "This chart tests how much payload utilization contributes to fuel burn once we look only at passenger-carrying flights."
        how = "Each point is a flight. The x-axis shows how full the plane was and the y-axis shows fuel burned. An upward trend means fuller flights do burn more, but the strength of that trend tells you whether fullness is a major or minor driver."
        result = (
            f"Load factor has a moderate positive relationship with burnoff (r = {corr:.2f}). Flights in the fullest 20% burn about "
            f"{high_load - low_load:,.0f} kg more fuel on average than flights in the emptiest 20%."
        )
        why = "Payload matters commercially and environmentally, but it is not the dominant driver of total burn. The technical effect is real, yet smaller than route length and gross aircraft weight."
        action = "Use load factor primarily in per-passenger efficiency analysis and commercial planning, not as the primary explanation for route-level fuel overrun."
    elif chart_type == "distance":
        sample = dff.sample(min(8000, len(dff)), random_state=3)
        fig = px.scatter(sample, x="RouteDistanceKm", y="Burnoff",
                         color="AircraftTypeGroup", opacity=0.4, trendline="ols",
                         labels={"RouteDistanceKm": "Route Distance (km)", "Burnoff": "Burnoff (kg)"},
                         title="Burnoff vs Route Distance")
        corr = dff["RouteDistanceKm"].corr(dff["Burnoff"])
        slope = np.polyfit(dff["RouteDistanceKm"], dff["Burnoff"], 1)[0]
        long_share = (dff["RouteDistanceKm"] >= 3000).mean() * 100
        what = "This scatterplot shows how route length translates into fuel demand across the network."
        how = "Points further right are longer routes, and points higher up burn more fuel. The trend line gives the average extra fuel linked to each extra kilometer."
        result = (
            f"Distance is almost linear with burnoff (r = {corr:.2f}). The fitted relationship is about {slope:.2f} kg per km, "
            f"and only {long_share:.1f}% of flights exceed 3,000 km, forming the long-range cluster in the upper-right."
        )
        why = "Distance is a first-order business segmentation variable: long sectors concentrate cost, emissions, and forecast uncertainty in a way that short sectors do not."
        action = "Report long-haul and short-haul performance separately and apply different monitoring thresholds, savings expectations, and forecast-confidence bands to each."
    elif chart_type == "monthly":
        dff2 = dff.copy()
        dff2["Month"] = pd.to_datetime(dff2["DepartureScheduled"]).dt.to_period("M").astype(str)
        counts = dff2.groupby("Month").size().reset_index(name="Flights")
        fig = px.bar(counts, x="Month", y="Flights", color_discrete_sequence=[COLORS["secondary"]],
                     title="Monthly Flight Volume")
        fig.update_xaxes(tickangle=45)
        peak = counts.loc[counts["Flights"].idxmax()]
        trough = counts.loc[counts["Flights"].idxmin()]
        ratio = peak["Flights"] / max(trough["Flights"], 1)
        what = "This time series counts how many flights appear in each calendar month of the training data."
        how = "Higher bars mean a busier month. Read from left to right to see when the network expands, when it slows down, and whether any unusual shocks stand out."
        result = (
            f"The busiest month is {peak['Month']} with {int(peak['Flights']):,} flights, versus {int(trough['Flights']):,} in the quietest month "
            f"({trough['Month']}). That is a {ratio:.1f}x swing in operating volume."
        )
        why = "Large volume swings change route mix, fleet deployment, congestion exposure, and control-room workload, which in turn changes both operating complexity and forecast difficulty."
        action = "Align staffing, monitoring cadence, and forecast review intensity with peak months rather than assuming one uniform operating environment across the year."
    elif chart_type == "corr":
        num_cols = ["Burnoff", "PlannedTripTime", "PlannedTOW", "PlannedZeroFuelWeight",
                    "TeledyneRampWeight", "RouteDistanceKm", "LoadFactor", "TotalPassengers"]
        corr = dff[num_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Feature Correlation Heatmap",
                        aspect="auto")
        burn_corr = corr["Burnoff"].drop("Burnoff").sort_values(key=np.abs, ascending=False)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(key=np.abs, ascending=False)
        strongest_pair = upper.index[0]
        strongest_value = upper.iloc[0]
        what = "This heatmap shows which inputs tend to move together and which ones are mostly telling the same story."
        how = "Values close to +1 mean two variables rise together, values near 0 mean there is little relationship, and negative values mean one tends to rise when the other falls. Start by looking at the Burnoff row, then look for input pairs that are almost duplicates."
        result = (
            f"Burnoff moves most closely with {burn_corr.index[0]} and {burn_corr.index[1]}. The strongest overlap between two inputs is "
            f"{strongest_pair[0]} and {strongest_pair[1]} at {strongest_value:.2f}, which means those fields are giving the model almost the same message."
        )
        why = "Technically, several weight fields are near-duplicates, which is acceptable for tree-based models but destabilizing for simpler linear models and business rules built on one-variable effects."
        action = "For linear reporting models or scorecards, reduce duplicate weight variables or regularize aggressively; for tree models, keep them but make their data quality a governance priority."
    else:
        fig = go.Figure()

    fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
    finding_div = chart_caption(what, how, result, why, action) if result else html.Div()
    return fig, finding_div


# ─── Model Performance ───────────────────────────────────────────────────────
@app.callback(
    Output("model-rmse-chart", "figure"),
    Output("model-rmse-caption", "children"),
    Input("main-tabs", "active_tab"),
)
def rmse_chart(_):
    ordered = model_res.sort_values("RMSE")
    fig = px.bar(ordered, x="Model", y="RMSE",
                 color="Model", text="RMSE",
                 color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                          COLORS["danger"], COLORS["success"]],
                 title="RMSE by Model (lower is better)")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False,
                      margin=dict(l=20, r=20, t=50, b=20))
    best = ordered.iloc[0]
    second = ordered.iloc[1]
    worst = ordered.iloc[-1]
    baseline_rmse = float(metric_lookup(cost_data, "Baseline RMSE (kg)") or 0)
    baseline_text = (
        f" Relative to the naive baseline at {baseline_rmse:,.0f} kg, the best model cuts error by "
        f"{100 * (1 - best['RMSE'] / baseline_rmse):.1f}%."
        if baseline_rmse > 0 else ""
    )
    caption = chart_caption(
        what="This chart compares how far off each model is when it makes bigger mistakes, measured in kilograms of fuel.",
        how="Lower bars are better. This metric gives extra weight to large misses, so it is a good way to judge which model is safest to trust when flights are harder to predict.",
        result=(
            f"{best['Model']} performs best at {best['RMSE']:.0f} kg, only "
            f"{second['RMSE'] - best['RMSE']:.1f} kg better than {second['Model']}. The weakest model is {worst['Model']} at {worst['RMSE']:.0f} kg.{baseline_text}"
        ),
        why="From a business perspective, this is the most defensible top-line model selection metric because it penalizes the kind of large miss that can distort fuel planning, savings estimation, and exception review.",
        action="Use the best-performing model as the default production benchmark, but retain the runner-up as a challenger model for governance and periodic revalidation."
    )
    return fig, caption


@app.callback(
    Output("model-mape-chart", "figure"),
    Output("model-mape-caption", "children"),
    Input("main-tabs", "active_tab"),
)
def mape_chart(_):
    ordered = model_res.sort_values("MAPE")
    fig = px.bar(ordered, x="Model", y="MAPE",
                 color="Model", text="MAPE",
                 color_discrete_sequence=[COLORS["primary"], COLORS["secondary"],
                                          COLORS["danger"], COLORS["success"]],
                 title="MAPE (%) by Model (lower is better)")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False,
                      margin=dict(l=20, r=20, t=50, b=20))
    best = ordered.iloc[0]
    second = ordered.iloc[1]
    worst = ordered.iloc[-1]
    caption = chart_caption(
        what="This chart shows model error as a percentage, which makes short flights and long flights easier to compare.",
        how="Lower bars are better. For example, a 4% error means the model is wrong by about 4 kg for every 100 kg actually burned.",
        result=(
            f"{best['Model']} has the lowest relative error at {best['MAPE']:.2f}%, with {second['Model']} only "
            f"{second['MAPE'] - best['MAPE']:.2f} percentage points behind. {worst['Model']} is worst at {worst['MAPE']:.2f}%."
        ),
        why="This is the easiest quality metric to use in finance, operations, and executive discussions because it normalizes error across flights of very different lengths and fuel profiles.",
        action="Use percentage error to set business-facing tolerance bands and to communicate expected forecast accuracy in planning or savings discussions."
    )
    return fig, caption


@app.callback(
    Output("pred-vs-actual", "figure"),
    Output("pred-caption", "children"),
    Input("scatter-model-select", "value"),
)
def pred_scatter(model_name):
    if model_name not in val_preds.columns:
        return go.Figure().add_annotation(text=f"No predictions for {model_name}", showarrow=False), html.Div()
    y_t = val_preds["y_true"].values
    y_p = val_preds[model_name].values
    sample = np.random.choice(len(y_t), min(5000, len(y_t)), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_t[sample], y=y_p[sample], mode="markers",
        marker=dict(color=COLORS["primary"], opacity=0.3, size=4),
        name="Flights",
    ))
    lim = [float(min(y_t.min(), y_p.min())), float(max(y_t.max(), y_p.max()))]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect"))
    fig.update_layout(
        title=f"{model_name}: Predicted vs Actual Burnoff",
        xaxis_title="Actual Burnoff (kg)", yaxis_title="Predicted Burnoff (kg)",
        template=PLOTLY_TEMPLATE, margin=dict(l=20, r=20, t=50, b=20),
    )
    errors = y_p - y_t
    rel_error = np.abs(errors) / np.maximum(np.abs(y_t), 1)
    within_5 = (rel_error <= 0.05).mean() * 100
    within_5_of_10 = round(within_5 / 10, 1)
    mean_bias = errors.mean()
    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.mean(errors ** 2))
    caption = chart_caption(
        what=f"This scatterplot compares {model_name}'s prediction for each sampled flight against the true fuel burn.",
        how="Every point should ideally sit on the red diagonal. Points above the line mean the model guessed too high, points below mean it guessed too low, and the farther a point is from the line, the bigger the miss.",
        result=(
            f"{model_name} is within 5% of the true burn on {within_5:.1f}% of flights, or about {within_5_of_10} out of every 10 flights. "
            f"On average it misses by {mae:.0f} kg, and its average tendency to guess high or low is only {mean_bias:+.1f} kg."
        ),
        why="Technically, this is the clearest visual check for calibration and systematic skew. Commercially, it shows whether the model is dependable enough to use as a planning benchmark rather than only as an academic forecasting exercise.",
        action="Inspect the regions where points consistently peel away from the diagonal, especially at high burn levels, and use those clusters to guide the next round of feature engineering or route-specific review."
    )
    return fig, caption


# ─── Feature Importance ──────────────────────────────────────────────────────
@app.callback(
    Output("fi-chart", "figure"),
    Output("fi-caption", "children"),
    Input("fi-model-select", "value"),
    Input("fi-top-n", "value"),
)
def fi_chart(model_name, top_n):
    if model_name == "__all__":
        frames = []
        for m in feat_imp["Model"].unique():
            df_m = feat_imp[feat_imp["Model"] == m].nlargest(top_n, "Importance").copy()
            max_imp = df_m["Importance"].max()
            if max_imp > 0:
                df_m["Importance"] = df_m["Importance"] / max_imp
            frames.append(df_m)
        df_all = pd.concat(frames)
        n_models = df_all["Model"].nunique()
        facet_cols = min(n_models, 2)
        fig = px.bar(
            df_all.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            facet_col="Model", facet_col_wrap=facet_cols,
            title=f"All Models – Top {top_n} Feature Importances (normalised per model)",
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, showlegend=False,
            margin=dict(l=20, r=20, t=70, b=20),
            yaxis_tickfont_size=9,
            height=max(600, top_n * 28 * ((n_models + 1) // 2)),
        )
        rank_frames = []
        for m in feat_imp["Model"].unique():
            ranked = feat_imp[feat_imp["Model"] == m].nlargest(top_n, "Importance").reset_index(drop=True)
            ranked["Rank"] = ranked.index + 1
            rank_frames.append(ranked[["Feature", "Model", "Rank"]])
        rank_df = pd.concat(rank_frames)
        consensus = (
            rank_df.groupby("Feature")
            .agg(Models=("Model", "nunique"), AvgRank=("Rank", "mean"))
            .sort_values(["Models", "AvgRank"], ascending=[False, True])
            .head(3)
            .reset_index()
        )
        consensus_text = ", ".join(
            f"{row.Feature} ({int(row.Models)} models, avg rank {row.AvgRank:.1f})"
            for row in consensus.itertuples(index=False)
        )
        caption = chart_caption(
            what="This view compares which inputs matter most to each model.",
            how="Read each panel separately. Longer bars mean that feature mattered more to that model. The bars are scaled within each panel, so focus on the order of the features rather than comparing bar length across different models.",
            result=f"The same features keep appearing near the top across models: {consensus_text}. That tells us the models agree that route length, time, and aircraft weight are doing most of the explaining.",
            why="When multiple model families converge on the same signals, the explanation is materially more credible. For a business audience, that means the model is learning core operating physics rather than fragile noise.",
            action="Treat the top recurring variables as critical data assets: enforce data-quality checks, monitor upstream changes, and prioritize completeness for those fields before tuning anything else."
        )
        return fig, caption

    df_fi = feat_imp[feat_imp["Model"] == model_name].nlargest(top_n, "Importance")
    fig = px.bar(df_fi.sort_values("Importance"),
                 x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title=f"{model_name} – Top {top_n} Feature Importances")
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False,
                      margin=dict(l=20, r=20, t=50, b=20), yaxis_tickfont_size=11,
                      height=480)
    top_features = df_fi.head(3)["Feature"].tolist()
    top_text = ", ".join(top_features)
    caption = chart_caption(
        what=f"This chart ranks the inputs {model_name} relies on most when it estimates fuel burn.",
        how="Longer bars mean the model leaned more heavily on that variable. The chart does not prove cause by itself, but it does show what information the model finds most useful.",
        result=f"For {model_name}, the strongest signals are {top_text}. In other words, the model is mainly using flight length, aircraft weight, and route structure rather than weak side clues.",
        why="This makes the model operationally explainable: it shows which levers are really driving forecast changes and which data fields carry the most model risk if they drift or degrade.",
        action="Use the top-ranked features to define data-governance priorities and to decide where richer operational variables would most likely improve performance."
    )
    return fig, caption


# ─── Story ───────────────────────────────────────────────────────────────────
@app.callback(
    Output("story-chart", "figure"),
    Output("story-finding", "children"),
    Input("story-chart-select", "value"),
)
def story_chart(chart):
    template = PLOTLY_TEMPLATE
    what = how = result = why = action = ""

    if chart == "aircraft":
        grp = df_val.groupby("AircraftTypeGroup").agg(
            MAE=("AbsError", "mean"), Count=("AbsError", "count"),
            StdResid=("Residual", "std"),
        ).reset_index()
        fig = px.bar(grp, x="AircraftTypeGroup", y="MAE",
                     color="AircraftTypeGroup", text="MAE",
                     labels={"MAE": "Mean Absolute Error (kg)", "AircraftTypeGroup": ""},
                     title="Mean Absolute Error by Aircraft Type (H4 test)")
        fig.update_traces(texttemplate="%{text:.0f} kg", textposition="outside")
        top_row = grp.loc[grp["MAE"].idxmax()]
        low_row = grp.loc[grp["MAE"].idxmin()]
        what = "This chart checks whether the model is equally reliable for each aircraft family."
        how = "Higher bars mean the model is further off on average. Always keep sample size in mind: a higher error for a tiny fleet is less meaningful than the same error for a large, well-sampled fleet."
        result = (
            f"{top_row['AircraftTypeGroup']} has the highest MAE at {top_row['MAE']:.0f} kg, while {low_row['AircraftTypeGroup']} is lowest at "
            f"{low_row['MAE']:.0f} kg. The spread is material but not extreme, which is why H4 is only partially supported rather than conclusively proven."
        )
        why = "The business implication is that model quality is broadly portable across fleets, but the smaller fleets still carry greater estimation uncertainty because they contribute less learning signal."
        action = "Keep a single production model for now, but add fleet-specific monitoring thresholds and targeted data enrichment for the smaller Airbus and MAX slices."

    elif chart == "season":
        order_s = ["Winter", "Spring", "Summer", "Fall"]
        grp = df_val.groupby("Season")["AbsError"].mean().reindex(order_s).reset_index()
        fig = px.bar(grp, x="Season", y="AbsError",
                     color="Season", text="AbsError",
                     labels={"AbsError": "Mean Absolute Error (kg)"},
                     title="Mean Absolute Error by Season")
        fig.update_traces(texttemplate="%{text:.0f} kg", textposition="outside")
        top_row = grp.loc[grp["AbsError"].idxmax()]
        low_row = grp.loc[grp["AbsError"].idxmin()]
        gap = top_row["AbsError"] - low_row["AbsError"]
        what = "This chart checks whether the model is noticeably worse in one season than another."
        how = "Higher bars mean the model is further off in that season. What matters most is the gap between seasons, not just the height of one bar on its own."
        result = (
            f"{top_row['Season']} is the hardest season at {top_row['AbsError']:.0f} kg MAE, while {low_row['Season']} is easiest at "
            f"{low_row['AbsError']:.0f} kg. The seasonal spread is only {gap:.0f} kg, so season matters, but it is not the dominant source of error."
        )
        why = "Season introduces some noise, but not enough to justify a fully separate seasonal modeling strategy from a business-return perspective."
        action = "Keep one core model, but use slightly wider operating tolerance bands in the hardest season and monitor seasonal shifts rather than splitting the model architecture."

    elif chart == "load":
        # Filter out near-zero load factor (ferry flights) to show real passenger trend
        df_lf = df_val[df_val["TotalPassengers"] > 10].copy()
        df_lf["LF_bin"] = pd.cut(df_lf["LoadFactor"], bins=10)
        grp = df_lf.groupby("LF_bin", observed=True)["Residual"].mean().reset_index()
        grp["LF_mid"] = grp["LF_bin"].apply(lambda x: x.mid)
        grp["color"] = grp["Residual"].apply(lambda v: "Over-predicted" if v > 0 else "Under-predicted")
        fig = px.bar(grp, x="LF_mid", y="Residual", color="color",
                     color_discrete_map={"Over-predicted": COLORS["danger"],
                                         "Under-predicted": COLORS["primary"]},
                     labels={"LF_mid": "Load Factor", "Residual": "Mean Residual (kg)"},
                     title="Prediction Bias by Load Factor (passenger flights, >10 pax)")
        fig.add_hline(y=0, line_color="black", line_width=1)
        max_bias = grp.iloc[grp["Residual"].abs().idxmax()]
        what = "This chart checks whether the model tends to guess too high or too low for emptier flights versus fuller flights."
        how = "Bars above zero mean the model guessed too high. Bars below zero mean it guessed too low. If the bars steadily move away from zero, the model is systematically leaning one way for that load band."
        result = (
            f"The bias is small but patterned: low-load flights tend to be slightly over-predicted and high-load flights slightly under-predicted. "
            f"The worst bin sits around load factor {max_bias['LF_mid']:.2f} with a mean bias of {max_bias['Residual']:+.0f} kg."
        )
        why = "This is a technical calibration issue rather than a model failure: the model captures the main payload effect, but not the full non-linearity at the tails of the load distribution."
        action = "If reducing edge-case bias matters commercially, add richer payload features such as passenger mass assumptions, baggage mix, or more explicit interaction terms for very full and very empty flights."

    elif chart == "distance":
        sample = df_val.sample(min(5000, len(df_val)), random_state=42)
        # Clip extreme outliers for better visualisation (99th percentile)
        p99 = sample["AbsError"].quantile(0.99)
        sample_clipped = sample[sample["AbsError"] <= p99]
        fig = px.scatter(sample_clipped, x="RouteDistanceKm", y="AbsError",
                         color="AircraftTypeGroup", opacity=0.4, trendline="ols",
                         labels={"RouteDistanceKm": "Route Distance (km)",
                                 "AbsError": "Absolute Error (kg)"},
                         title="Prediction Error vs Route Distance (top 1% outliers removed)")
        slope = np.polyfit(sample_clipped["RouteDistanceKm"], sample_clipped["AbsError"], 1)[0]
        short_mean = sample_clipped[sample_clipped["RouteDistanceKm"] <= sample_clipped["RouteDistanceKm"].quantile(0.25)]["AbsError"].mean()
        long_mean = sample_clipped[sample_clipped["RouteDistanceKm"] >= sample_clipped["RouteDistanceKm"].quantile(0.75)]["AbsError"].mean()
        what = "This scatterplot checks whether longer routes are harder for the model to predict."
        how = "Points further right are longer flights. Points higher up are bigger misses. The trend line shows how much the miss tends to grow as routes get longer."
        result = (
            f"Error rises with distance: the fitted slope is about {slope:.3f} kg of extra absolute error per km. "
            f"Flights in the longest quartile miss by roughly {long_mean - short_mean:,.0f} kg more than flights in the shortest quartile."
        )
        why = "Long sectors accumulate uncertainty from winds, tactical routing, step-climbs, and ATC effects, so forecast risk scales with mission length even when the model is fundamentally sound."
        action = "Apply tighter monitoring to long-haul routes and prioritize weather, ATC, and routeing variables if the next model iteration is meant to improve commercial confidence on high-burn sectors."

    elif chart == "temporal":
        df_val_t = df_val.copy()
        df_val_t["Month"] = pd.to_datetime(df_val_t["DepartureScheduled"]).dt.to_period("M").astype(str)
        grp = df_val_t.groupby("Month")["Residual"].mean().reset_index()
        fig = px.line(grp, x="Month", y="Residual", markers=True,
                      labels={"Residual": "Mean Residual (kg)"},
                      title="Temporal Bias: Mean Residual by Month (Data Drift Check)")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_xaxes(tickangle=45)
        max_abs = grp.iloc[grp["Residual"].abs().idxmax()]
        mean_abs = grp["Residual"].abs().mean()
        what = "This line chart checks whether the model starts drifting over time."
        how = "The red zero line is the goal. If the line stays close to zero, the model is staying balanced. If it steadily moves up or down, the model is starting to go off track."
        result = (
            f"The monthly average error stays centered overall, with an average monthly lean of only {mean_abs:.1f} kg. "
            f"The largest monthly swing is {max_abs['Residual']:+.1f} kg in {max_abs['Month']}, which is small relative to overall model error."
        )
        why = "From a governance standpoint, the model appears temporally stable: there is no evidence of persistent drift large enough to undermine business use."
        action = "Continue scheduled monitoring rather than emergency retraining, and trigger model review only if a sustained month-by-month drift pattern emerges."

    elif chart == "routes":
        worst = route_err.nlargest(15, "RMSE")
        fig = px.bar(worst.sort_values("RMSE"),
                     x="RMSE", y="ScheduledRoute", orientation="h",
                     color="RMSE", color_continuous_scale="Reds", text="FlightCount",
                     labels={"RMSE": "RMSE (kg)", "ScheduledRoute": ""},
                     title="Top 15 Routes by Prediction RMSE")
        fig.update_traces(texttemplate="%{text} flights", textposition="outside")
        highest = worst.iloc[0]
        eligible = worst[worst["FlightCount"] >= 10]
        focus_text = ""
        if not eligible.empty:
            focus = eligible.iloc[0]
            focus_text = (
                f" Among routes with at least 10 flights, {focus['ScheduledRoute']} is the key operational watchlist item "
                f"at {focus['RMSE']:.0f} kg RMSE."
            )
        what = "This ranking shows the routes where the model struggles most."
        how = "Longer bars mean bigger average misses on that route. The flight-count labels matter because a route with only a few flights can look bad just by chance."
        result = (
            f"The hardest route in the chart is {highest['ScheduledRoute']} at {highest['RMSE']:.0f} kg RMSE based on "
            f"{int(highest['FlightCount'])} flights.{focus_text}"
        )
        why = "This is the most direct link between analytics and operational follow-up: it identifies where model weakness is concentrated enough to distort planning or opportunity sizing."
        action = "Use this ranking as a route-level investigation backlog, starting with routes that combine high error and meaningful traffic volume."
    else:
        fig = go.Figure()

    fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
    finding_div = chart_caption(what, how, result, why, action) if result else html.Div()
    return fig, finding_div


# ─── Sustainability ──────────────────────────────────────────────────────────
@app.callback(
    Output("co2-chart", "figure"),
    Output("co2-finding", "children"),
    Input("co2-chart-select", "value"),
)
def co2_chart(chart):
    template = PLOTLY_TEMPLATE
    what = how = result = why = action = ""

    if chart == "struct_vs_avoid":
        grp = flight_analytics.groupby("AircraftTypeGroup").agg(
            StructuralCO2_kg=("StructuralCO2_kg", "mean"),
            AvoidableCO2_kg=("AvoidableCO2_kg", "mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(x=grp["AircraftTypeGroup"], y=grp["StructuralCO2_kg"], name="Structural CO2", marker_color=COLORS["primary"])
        fig.add_bar(x=grp["AircraftTypeGroup"], y=grp["AvoidableCO2_kg"], name="Avoidable CO2", marker_color=COLORS["danger"])
        fig.update_layout(barmode="stack", title="Mean CO2 per Flight: Structural vs Avoidable")
        grp["AvoidableShare"] = grp["AvoidableCO2_kg"] / (grp["StructuralCO2_kg"] + grp["AvoidableCO2_kg"]).replace(0, np.nan)
        top_share = grp.loc[grp["AvoidableShare"].idxmax()]
        what = "This stacked bar chart separates the emissions a flight would normally be expected to produce from the extra emissions above that level."
        how = "Blue is the expected part, based on the kind of route, aircraft, and load. Red is the extra part above that expectation. A bigger red slice means there may be more room to improve how those flights are being operated."
        result = (
            f"{top_share['AircraftTypeGroup']} has the highest avoidable share, with {top_share['AvoidableShare'] * 100:.1f}% of its average per-flight CO2 showing up as extra emissions above what similar flights would normally be expected to produce."
        )
        why = "This avoids a common business mistake: confusing inherently difficult missions with preventable inefficiency. The red share is the part that is most commercially and environmentally actionable."
        action = "Target the fleet groups with the highest avoidable share for planning and operating-process review before considering structural fleet changes."

    elif chart == "routes":
        top = route_opp.nlargest(15, "AnnualAvoidableCO2_t")
        fig = px.bar(
            top.sort_values("AnnualAvoidableCO2_t"),
            x="AnnualAvoidableCO2_t", y="ScheduledRoute", orientation="h",
            color="OpportunityScore", color_continuous_scale="OrRd",
            text="FlightCount",
            labels={"AnnualAvoidableCO2_t": "Annual Avoidable CO2 (tonnes)", "ScheduledRoute": ""},
            title="Top Routes by Annual Avoidable CO2",
        )
        fig.update_traces(texttemplate="%{text} flights", textposition="outside")
        leader = top.iloc[0]
        what = "This ranking turns route inefficiency into annual emissions opportunity."
        how = "Longer bars mean more avoidable CO2 over a full year. The flight-count labels show whether that value comes from many repeat flights or from a smaller number of especially costly ones."
        result = (
            f"{leader['ScheduledRoute']} is the top emissions hotspot at {leader['AnnualAvoidableCO2_t']:,.0f} tonnes of avoidable CO2 per year across "
            f"{int(leader['FlightCount'])} flights."
        )
        why = "These routes combine repetition with avoidable emissions, which is why they matter far more than one-off high-burn flights from a portfolio-management perspective."
        action = "Use this ranking as the emissions intervention backlog and begin with the top routes that combine annual impact with sufficient operational control."

    elif chart == "hotspots":
        top = route_opp[route_opp["FlightCount"] >= 5].nlargest(80, "OpportunityScore")
        fig = px.scatter(
            top,
            x="MeanStructuralCO2_kg" if "MeanStructuralCO2_kg" in top.columns else "MeanDistanceKm",
            y="MeanAvoidableCO2_kg" if "MeanAvoidableCO2_kg" in top.columns else "AnnualAvoidableCO2_t",
            size="FlightCount",
            color="OpportunityScore",
            hover_name="ScheduledRoute",
            color_continuous_scale="YlOrRd",
            labels={
                "MeanStructuralCO2_kg": "Mean Structural CO2 (kg / flight)",
                "MeanAvoidableCO2_kg": "Mean Avoidable CO2 (kg / flight)",
            },
            title="Hotspots: Structural vs Avoidable Emissions",
        )
        hotspot = top.nlargest(1, "OpportunityScore").iloc[0]
        what = "This matrix separates routes that are naturally heavy from routes that are heavier than they should be."
        how = "Routes further right are inherently more emissions-intensive. Routes higher up have more extra emissions than expected. Bigger bubbles mean more flights, so the issue matters more often."
        result = (
            f"{hotspot['ScheduledRoute']} is the highest-value hotspot in this view. It combines structural intensity with "
            f"{hotspot['MeanAvoidableCO2_kg']:,.0f} kg of avoidable CO2 per flight and {int(hotspot['FlightCount'])} observed flights."
        )
        why = "The most valuable sustainability targets are not automatically the longest sectors; they are the sectors that sit above their expected emissions profile after controlling for mission structure."
        action = "Prioritize routes that are unusually high on the vertical axis for their structural burden, because those are the ones most likely to yield real emissions improvement rather than just reflect long distance."

    elif chart == "loadfactor":
        grp = flight_analytics.groupby("LoadFactorBand").agg(
            MeanStructural=("StructuralCO2_kg", "mean"),
            MeanAvoidable=("AvoidableCO2_kg", "mean"),
            Count=("ActualBurn_kg", "size"),
        ).reset_index()
        fig = px.bar(
            grp,
            x="LoadFactorBand", y="MeanAvoidable",
            color="MeanStructural", color_continuous_scale="Blues",
            text="Count",
            labels={"MeanAvoidable": "Mean Avoidable CO2 (kg / flight)", "LoadFactorBand": "Load Factor Band"},
            title="Avoidable Emissions by Load Factor Band",
        )
        fig.update_traces(texttemplate="%{text} flights", textposition="outside")
        top_band = grp.loc[grp["MeanAvoidable"].idxmax()]
        what = "This chart checks whether extra emissions mainly happen on emptier flights or whether they still show up when planes are fairly full."
        how = "Bar height shows extra CO2 per flight. Darker blue means the flights in that band are naturally more emissions-heavy. If the bars stay high even at strong load factors, the problem is not just low demand."
        result = (
            f"The highest avoidable-emissions band is {top_band['LoadFactorBand']} at {top_band['MeanAvoidable']:,.0f} kg of avoidable CO2 per flight across "
            f"{int(top_band['Count'])} flights."
        )
        why = "This chart helps distinguish commercial underutilization from true operating inefficiency, which is essential because the solutions sit in different parts of the business."
        action = "If avoidable emissions are concentrated in low-load bands, focus on network and demand planning; if they remain high at strong load factors, escalate to operational process review."
    else:
        fig = go.Figure()

    fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
    finding_div = chart_caption(what, how, result, why, action) if result else html.Div()
    return fig, finding_div


@app.callback(
    Output("business-chart", "figure"),
    Output("business-finding", "children"),
    Input("business-chart-select", "value"),
)
def business_chart(chart):
    template = PLOTLY_TEMPLATE
    what = how = result = why = action = ""

    if chart == "matrix":
        top = route_opp[route_opp["FlightCount"] >= 5].nlargest(100, "OpportunityScore")
        fig = px.scatter(
            top,
            x="FlightCount", y="MeanExcessBurn",
            size="AnnualAvoidableCost_eur", color="OpportunityScore",
            hover_name="ScheduledRoute",
            color_continuous_scale="YlOrRd",
            labels={
                "FlightCount": "Flight Count",
                "MeanExcessBurn": "Mean Excess Burn (kg / flight)",
                "AnnualAvoidableCost_eur": "Annual Avoidable Cost (EUR)",
            },
            title="Opportunity Matrix: Volume vs Excess Burn vs Annual Value",
        )
        fig.add_hline(y=0, line_color="black", line_width=1)
        leader = top.nlargest(1, "OpportunityScore").iloc[0]
        what = "This opportunity matrix shows where the business case is strongest."
        how = "Move right for more flights, up for more excess fuel per flight, and look for bigger bubbles for more yearly value. The best targets are high, right, and large."
        result = (
            f"{leader['ScheduledRoute']} is the strongest current business case in the visible set, combining "
            f"{int(leader['FlightCount'])} flights, {leader['MeanExcessBurn']:,.0f} kg of mean excess burn, and "
            f"EUR {leader['AnnualAvoidableCost_eur']:,.0f} of annual avoidable value."
        )
        why = "This converts descriptive inefficiency into portfolio logic: repeat problems with medium-to-high excess burn usually dominate annual value more than isolated extreme cases."
        action = "Prioritize the upper-right large bubbles for business-case development, because those are the segments most likely to justify operational change."

    elif chart == "segments":
        top = segment_opp[segment_opp["FlightCount"] >= 5].nlargest(20, "AnnualAvoidableCost_eur")
        fig = px.bar(
            top.sort_values("AnnualAvoidableCost_eur"),
            x="AnnualAvoidableCost_eur", y="SegmentName", orientation="h",
            color="SegmentType", text="FlightCount",
            labels={"AnnualAvoidableCost_eur": "Annual Avoidable Cost (EUR)", "SegmentName": ""},
            title="Top Segments by Annual Avoidable Cost",
        )
        fig.update_traces(texttemplate="%{text} flights", textposition="outside")
        leader = top.iloc[0]
        what = "This ranking compares where the money is being lost across different kinds of groups, such as aircraft families, airports, or time windows."
        how = "Longer bars mean more yearly avoidable cost. The color tells you what kind of group it is, and the text label shows how many flights support that estimate."
        result = (
            f"The highest-value group is {leader['SegmentName']} ({leader['SegmentType']}) at EUR {leader['AnnualAvoidableCost_eur']:,.0f} per year over "
            f"{int(leader['FlightCount'])} flights."
        )
        why = "This is an ownership chart as much as a value chart: the segment type tells you which business function is most likely to control the fix."
        action = "Use the leading segment type to assign accountability, whether that means network planning, fleet planning, airport operations, or schedule management."

    elif chart == "scenarios":
        top = intervention_data.nlargest(20, "SavingsPotential_eur")
        fig = px.bar(
            top.head(15).sort_values("SavingsPotential_eur"),
            x="SavingsPotential_eur", y="SegmentName", orientation="h",
            color="Scenario", barmode="group",
            labels={"SavingsPotential_eur": "Potential Savings (EUR)", "SegmentName": ""},
            title="Intervention Scenarios: Estimated Value by Segment",
        )
        leader = top.iloc[0]
        what = "This chart turns observed inefficiency into estimated savings from specific actions."
        how = "Each bar shows the value of applying one improvement idea to one segment. Higher bars mean more potential savings, and the grouped colors let you compare different action ideas on the same segment."
        result = (
            f"The largest modeled upside comes from applying '{leader['Scenario']}' to {leader['SegmentName']}, worth about "
            f"EUR {leader['SavingsPotential_eur']:,.0f} in annual savings."
        )
        why = "This is the bridge between analytics and implementation because it translates an identified hotspot into a financially framed operating scenario."
        action = "Pilot the highest-value scenario on one or two leading segments first, measure realized savings, and only then scale the intervention more broadly."
    else:
        fig = go.Figure()

    fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
    finding_div = chart_caption(what, how, result, why, action) if result else html.Div()
    return fig, finding_div


# ─── What-If Simulator ──────────────────────────────────────────────────────
@app.callback(
    Output("whatif-result", "children"),
    Input("whatif-aircraft", "value"),
    Input("whatif-route", "value"),
    Input("whatif-load", "value"),
    Input("whatif-hour", "value"),
    Input("whatif-season", "value"),
)
def whatif_simulate(aircraft, route, load_factor, hour, season):
    # Use historical data as basis for the prediction estimate
    # Filter to similar flights and show the expected range
    mask = df_eng["AircraftTypeGroup"] == aircraft
    if route in df_eng["ScheduledRoute"].values:
        route_mask = df_eng["ScheduledRoute"] == route
        route_data = df_eng[mask & route_mask]
    else:
        route_data = pd.DataFrame()

    # Compute estimates from historical data with adjustments
    if len(route_data) >= 5:
        base_burnoff = route_data["Burnoff"].mean()
        base_load = route_data["LoadFactor"].mean()
        # Approximate linear adjustment for load factor difference
        # From the data: ~500 kg more fuel per 0.1 increase in load factor (typical for short-haul)
        load_sensitivity = route_data[["LoadFactor", "Burnoff"]].cov().iloc[0, 1] / route_data["LoadFactor"].var() if route_data["LoadFactor"].var() > 0 else 500
        load_adjustment = (load_factor - base_load) * load_sensitivity
        predicted_burnoff = base_burnoff + load_adjustment
        co2_kg = predicted_burnoff * CO2_PER_KG_FUEL
        n_flights = len(route_data)
        std_burnoff = route_data["Burnoff"].std()
        distance = route_data["RouteDistanceKm"].mean()
    else:
        # Fallback to global aircraft-type average adjusted by load
        type_data = df_eng[mask]
        base_burnoff = type_data["Burnoff"].mean()
        base_load = type_data["LoadFactor"].mean()
        load_sensitivity = 3000  # approximate kg per unit load factor
        load_adjustment = (load_factor - base_load) * load_sensitivity
        predicted_burnoff = base_burnoff + load_adjustment
        co2_kg = predicted_burnoff * CO2_PER_KG_FUEL
        n_flights = 0
        std_burnoff = type_data["Burnoff"].std()
        distance = type_data["RouteDistanceKm"].mean()

    predicted_burnoff = max(predicted_burnoff, 500)  # floor at 500 kg
    co2_kg = predicted_burnoff * CO2_PER_KG_FUEL

    # Estimate passengers from load factor and capacity
    avg_capacity = df_eng[mask]["AircraftCapacity"].mean() if mask.sum() > 0 else 189
    est_passengers = int(load_factor * avg_capacity)
    co2_per_pax = co2_kg / max(est_passengers, 1)

    result_cards = dbc.Row([
        dbc.Col(kpi_card("Predicted Burnoff", f"{predicted_burnoff:,.0f} kg",
                         f"+/- {std_burnoff:.0f} kg (1 std)" if std_burnoff else "",
                         COLORS["primary"], ""), md=3),
        dbc.Col(kpi_card("CO2 Emissions", f"{co2_kg:,.0f} kg",
                         f"= {predicted_burnoff:.0f} kg fuel x 3.16",
                         COLORS["danger"], ""), md=3),
        dbc.Col(kpi_card("CO2 per Passenger", f"{co2_per_pax:.1f} kg",
                         f"~{est_passengers} passengers at {load_factor*100:.0f}% LF",
                         COLORS["warning"], ""), md=3),
        dbc.Col(kpi_card("Route Distance", f"{distance:,.0f} km",
                         f"Based on {n_flights} historical flights" if n_flights >= 5 else "Estimated from fleet average",
                         COLORS["secondary"], ""), md=3),
    ], className="g-3")

    # Comparison: what if different aircraft?
    comparison_rows = []
    for ac in _aircraft_types:
        ac_data = df_eng[(df_eng["AircraftTypeGroup"] == ac) & (df_eng["ScheduledRoute"] == route)] if route in df_eng["ScheduledRoute"].values else df_eng[df_eng["AircraftTypeGroup"] == ac]
        if len(ac_data) >= 3:
            ac_burn = ac_data["Burnoff"].mean()
            comparison_rows.append({"Aircraft": ac, "Avg Burnoff (kg)": round(ac_burn, 0),
                                    "CO2 (kg)": round(ac_burn * CO2_PER_KG_FUEL, 0)})

    if comparison_rows:
        comp_text = html.Div([
            html.P("Fleet comparison for this route:", className="fw-semibold small mt-3 mb-1"),
            html.Ul([html.Li(f"{r['Aircraft']}: ~{r['Avg Burnoff (kg)']:,.0f} kg fuel, "
                             f"~{r['CO2 (kg)']:,.0f} kg CO2", className="small")
                     for r in comparison_rows])
        ])
    else:
        comp_text = html.Div()

    return html.Div([result_cards, comp_text])


# ─── Anomaly Detection ──────────────────────────────────────────────────────
@app.callback(
    Output("anomaly-chart", "figure"),
    Output("anomaly-finding", "children"),
    Input("anomaly-chart-select", "value"),
)
def anomaly_chart(chart):
    template = PLOTLY_TEMPLATE
    what = how = result = why = action = ""

    if chart == "dist":
        z_col = "ResidualZ_segment" if "ResidualZ_segment" in flight_analytics.columns else "Residual"
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=flight_analytics[z_col],
            nbinsx=70,
            marker_color=COLORS["primary"],
            opacity=0.75,
            name="Flights",
        ))
        fig.add_vline(x=2.5, line_dash="dash", line_color=COLORS["danger"])
        fig.add_vline(x=-2.5, line_dash="dash", line_color=COLORS["danger"])
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(
            title="Segment-Normalized Residual Distribution",
            xaxis_title="Residual z-score within segment",
            yaxis_title="Count",
        )
        outlier_share = (np.abs(flight_analytics[z_col]) >= 2.5).mean() * 100
        what = "This histogram shows how unusual each flight looks compared with similar flights, not compared with the whole network."
        how = "Zero means a flight behaved about as expected for its peer group. The dashed lines mark flights that stand out clearly from similar flights."
        result = f"About {outlier_share:.1f}% of flights fall beyond the +/-2.5 threshold, which is the part of the network most worth screening for anomalies."
        why = "This is a better operational screen than raw miss size because it isolates flights that are unusual relative to comparable flying conditions, not merely difficult missions."
        action = "Use the thresholded tail as the first-stage anomaly queue, then filter again by recurrence before escalating cases to operations."

    elif chart == "segments":
        top = anomaly_monitor[anomaly_monitor["FlightCount"] >= 5].copy()
        top = top.groupby("SegmentType").agg(
            PersistenceRate=("PersistenceFlag", "mean"),
            MeanAnomalyRate=("AnomalyRate", "mean"),
            SegmentCount=("SegmentName", "size"),
        ).reset_index()
        top["PersistenceRate"] *= 100
        top["MeanAnomalyRate"] *= 100
        fig = px.bar(
            top,
            x="SegmentType", y="PersistenceRate",
            color="MeanAnomalyRate", color_continuous_scale="OrRd",
            text="SegmentCount",
            labels={"PersistenceRate": "Persistent Risk Segments (%)", "SegmentType": "Segment Type"},
            title="Recurring Risk by Segment Type",
        )
        fig.update_traces(texttemplate="%{text} segments", textposition="outside")
        leader = top.loc[top["PersistenceRate"].idxmax()]
        what = "This chart compares where repeat problems show up most often: by route, by route-aircraft mix, by airport and hour, or by another group."
        how = "Higher bars mean a larger share of that group keeps showing repeated issues. The color shows how common unusual flights are overall, and the text label shows how many groups are being compared."
        result = (
            f"{leader['SegmentType']} has the highest repeat-problem rate at {leader['PersistenceRate']:.1f}%, with unusual flights averaging "
            f"{leader['MeanAnomalyRate']:.1f}% across {int(leader['SegmentCount'])} groups."
        )
        why = "High persistence indicates a process problem rather than random noise, which is why this chart is more actionable than a list of isolated flight exceptions."
        action = "Investigate the highest-persistence category at the process level first instead of reviewing one anomalous flight at a time."

    elif chart == "airport_hour":
        airport_hour = segment_opp[segment_opp["SegmentType"] == "Airport × Hour"].copy()
        if len(airport_hour) == 0:
            fig = go.Figure().add_annotation(text="No airport-hour analytics available", showarrow=False)
            what = "This chart is meant to show whether certain airports and departure times repeatedly create unusual flights."
            how = "When data is available, darker cells indicate combinations of airport and time of day where problems happen more often."
            result = "Airport-hour analytics are not available in the current dataset export, so no hotspot matrix can be shown."
            why = "Airport-hour segmentation is one of the most decision-useful anomaly cuts because it distinguishes local station problems from network-wide patterns."
            action = "Add or restore the airport-hour segmentation feed so anomaly review can be assigned to the right base and operating window."
        else:
            airport_hour["Origin"] = airport_hour["SegmentName"].str.split(" \\| ").str[0]
            airport_hour["HourBand"] = airport_hour["SegmentName"].str.split(" \\| ").str[1]
            pivot = airport_hour.pivot_table(index="Origin", columns="HourBand", values="AnomalyRate", aggfunc="mean")
            pivot = pivot.fillna(0) * 100
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(15).index]
            fig = px.imshow(
                pivot,
                text_auto=".1f",
                color_continuous_scale="YlOrRd",
                labels={"color": "Anomaly Rate (%)"},
                title="Airport × Hour Anomaly Heatmap",
                aspect="auto",
            )
            max_pos = np.unravel_index(np.nanargmax(pivot.to_numpy()), pivot.shape)
            top_airport = pivot.index[max_pos[0]]
            top_hour = pivot.columns[max_pos[1]]
            top_rate = pivot.to_numpy()[max_pos]
            what = "This heatmap highlights where unusual flights cluster by airport and time of day."
            how = "Rows are airports, columns are time bands, and darker cells mean a higher share of unusual flights. Reading across a row shows whether an airport has a constant problem or one that appears only at certain times."
            result = f"The strongest visible hotspot is {top_airport} during {top_hour}, where the anomaly rate reaches {top_rate:.1f}%."
            why = "A hotspot in this matrix is highly actionable because it usually points to a local bottleneck such as congestion, turnaround pressure, or a repeated dispatch pattern rather than a fleet-wide issue."
            action = "Treat the darkest cells as station-level review candidates and connect them to turnaround, dispatch, and departure-window process analysis."

    elif chart == "routes":
        top = anomaly_monitor[
            (anomaly_monitor["SegmentType"].isin(["Route", "Route × Aircraft"]))
            & (anomaly_monitor["FlightCount"] >= 5)
        ].nlargest(20, "AnomalyRate")
        plot_col = "SegmentName"
        fig = px.bar(
            top.sort_values("AnomalyRate"),
            x="AnomalyRate", y=plot_col, orientation="h",
            color="PersistenceFlag",
            text="FlightCount",
            labels={"AnomalyRate": "Anomaly Rate", plot_col: ""},
            title="Top Persistent Anomaly Segments",
        )
        fig.update_traces(texttemplate="%{text} flights", textposition="outside")
        leader = top.iloc[0] if len(top) else None
        if leader is not None:
            what = "This ranking surfaces the routes or route-aircraft combinations where unusual flights happen often enough to matter."
            how = "Longer bars mean a higher share of unusual flights. The text label shows how many flights support the result, so you can tell the difference between a real pattern and a tiny sample."
            result = (
                f"{leader[plot_col]} is the strongest current review candidate at an anomaly rate of {leader['AnomalyRate'] * 100:.1f}% "
                f"across {int(leader['FlightCount'])} flights."
            )
            why = "This is the highest-value anomaly list because it combines recurrence with material traffic volume, making it suitable for formal operational review."
            action = "Open route-level or route-aircraft reviews for the top entries first, and assign priority according to anomaly rate, traffic volume, and commercial importance."
    else:
        fig = go.Figure()

    fig.update_layout(template=template, margin=dict(l=20, r=20, t=50, b=20))
    finding_div = chart_caption(what, how, result, why, action) if result else html.Div()
    return fig, finding_div


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🚀 Starting Ryanair Burnoff Dashboard …")
    print("   Open: http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
