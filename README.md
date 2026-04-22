# Ryanair Fuel Burn Analytics

This project predicts airline flight fuel burn and extends the model into three decision-focused analytics layers:

- sustainability and avoidable emissions
- business value and savings prioritization
- anomaly detection and recurring operational risk

The pipeline is built around a `LightGBM` validation benchmark and a Dash dashboard.

## Project Structure

- [01_feature_engineering.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/01_feature_engineering.py): loads `train.csv`, cleans data, and builds engineered features
- [02_model_training .py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/02_model_training%20.py): trains Ridge, Random Forest, XGBoost, and LightGBM models
- [03_evaluation_and_story.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/03_evaluation_and_story.py): evaluates the primary model and exports validation diagnostics
- [06_applications.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/06_applications.py): builds flight-level operational analytics and segment opportunity tables
- [04_dashboard.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/04_dashboard.py): launches the interactive dashboard
- [05_test_predictions.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/05_test_predictions.py): optional script for test-set inference
- [airport_coords.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/airport_coords.py): airport metadata helpers

## Folder Layout

- `train/`: input training data
- `test/`: optional held-out test data
- `data/`: generated CSV outputs
- `models/`: serialized trained models
- `figures/`: saved charts and diagnostics

## Environment

Use Python 3.10+.

Create a local virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Equivalent package list:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly dash dash-bootstrap-components lightgbm xgboost
```

Optional packages:

```bash
pip install statsmodels shap
```

- `statsmodels` enables VIF diagnostics in feature engineering
- `shap` enables SHAP summaries in `06_applications.py`

## Data Requirements

The project expects the training file at one of these locations:

- `train/train.csv`
- `data/train.csv`
- `train.csv`

You can also override the training path with:

```bash
export DATA_PATH=/full/path/to/train.csv
```

Optional baseline file for airline planning comparison:

- `predicted_fuel_consumption.csv`
- `data/predicted_fuel_consumption.csv`

## Reproducible Run Order

Run the scripts in this exact order from the `emerging_topics/` directory:

```bash
python3 01_feature_engineering.py
python3 "02_model_training .py"
python3 03_evaluation_and_story.py
python3 06_applications.py
python3 04_dashboard.py
```

If you want test predictions as well:

```bash
python3 05_test_predictions.py
```

## What Each Step Produces

### 1. Feature Engineering

Main outputs:

- `data/train_engineered.csv`
- `data/eda_summary.csv`
- `data/vif_analysis.csv` if `statsmodels` is installed
- `figures/eda_*.png`

### 2. Model Training

Main outputs:

- `data/val_predictions.csv`
- `data/model_results.csv`
- `data/feature_importance.csv`
- `data/hp_search_results.csv`
- `models/*.pkl`
- `figures/model_*.png`

### 3. Evaluation and Story Diagnostics

Main outputs:

- `data/route_error_analysis.csv`
- `data/bias_analysis.csv`
- `data/baseline_comparison.csv`
- `data/validation_diagnostics.csv`
- `figures/story_*.png`

### 4. Operational Analytics Layer

Main outputs:

- `data/flight_analytics.csv`
- `data/co2_analysis.csv`
- `data/cost_savings.csv`
- `data/anomalies.csv`
- `data/route_opportunity.csv`
- `data/segment_opportunity.csv`
- `data/anomaly_monitor.csv`
- `data/intervention_scenarios.csv`

### 5. Dashboard

The dashboard reads the generated CSVs above and runs locally at:

- `http://127.0.0.1:8050`

## Core Analytical Logic

The project uses the model prediction as an operational benchmark:

- `predicted burn` = expected structural burn
- `actual - predicted` = excess burn
- positive excess burn = potential avoidable fuel, CO2, and cost

Important derived metrics include:

- `ExcessBurn_kg`
- `ExcessBurnPct`
- `StructuralCO2_kg`
- `AvoidableCO2_kg`
- `AvoidableFuelCost_eur`
- `OpsPlanningGap_kg`
- `ResidualZ_segment`
- `OpportunityScore`

## Dashboard Scope

The dashboard includes:

- route map and EDA
- model performance and feature importance
- storyline and bias checks
- sustainability hotspots
- business value prioritization
- anomaly monitoring and recurring risk segments

## Reproducibility Notes

- Run all scripts from the `emerging_topics/` directory so relative paths resolve correctly.
- `02_model_training .py` includes hyperparameter tuning by default and may take significantly longer than the other stages.
- If `LightGBM` is available, it is used as the primary operational benchmark in downstream analytics.
- The dashboard depends on generated CSVs, so rerun steps 1 to 4 after any major code or data change.
- Some metrics are annualized using a fixed assumption of `550,000` flights/year.
- Cost and carbon values depend on fixed assumptions in [06_applications.py](/Users/pabloditerlizzi/Desktop/S10/ET/Emerging%20topics/emerging_topics/06_applications.py:1):
  - fuel price
  - CO2 factor
  - fuel-carry penalty
  - EU ETS price

## Interpretation Guardrails

- Model residuals are not causal proof.
- Predicted burn is an expected benchmark, not an optimal operational target.
- Missing variables such as weather, ATC constraints, taxi time, de-icing, and maintenance state can affect interpretation.
- Use anomaly and opportunity outputs to prioritize investigation, not to make automatic operational recommendations.

## Quick Validation Checklist

After running the full pipeline, confirm that these files exist:

- `data/train_engineered.csv`
- `data/val_predictions.csv`
- `data/model_results.csv`
- `data/validation_diagnostics.csv`
- `data/flight_analytics.csv`
- `data/route_opportunity.csv`
- `data/anomaly_monitor.csv`

Then launch:

```bash
python3 04_dashboard.py
```

If the dashboard loads and the sustainability, business value, and anomalies tabs render, the full reproducible workflow is in place.
