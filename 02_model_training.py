"""
02_model_training.py
---------------------
Trains four regression models with cross-validated hyperparameter tuning:
  1. Ridge Regression     – linear baseline; GridSearchCV over alpha
  2. Random Forest        – non-linear ensemble; RandomizedSearchCV
  3. XGBoost              – gradient boosting; native xgb.cv random search
  4. LightGBM             – upgraded primary candidate; lgb.cv random search

TUNE_HYPERPARAMS = True   runs full search (recommended; ~20-40 min on laptop)
TUNE_HYPERPARAMS = False  uses pre-validated defaults for quick iteration

Outputs:
  - models/ridge_model.pkl, rf_model.pkl, xgboost.pkl, lgbm_model.pkl
  - models/preprocessor.pkl
  - data/val_predictions.csv
  - data/model_results.csv        (RMSE / MAPE / R² / CV-RMSE per model)
  - data/feature_importance.csv
  - data/hp_search_results.csv    (all hyperparameter configs tried)
  - figures/model_*.png

Run: python 02_model_training.py
(Must run after 01_feature_engineering.py)
"""

import os
import pickle
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(BASE_DIR)  # ensure all relative paths resolve to the script's directory

try:
    import lightgbm as lgb
    LGBM_IMPORT_ERROR = None
except Exception as exc:
    lgb = None
    LGBM_IMPORT_ERROR = exc

try:
    import xgboost as xgb
    XGB_IMPORT_ERROR = None
except Exception as exc:
    xgb = None
    XGB_IMPORT_ERROR = exc

os.makedirs("models",  exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130})

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
TUNE_HYPERPARAMS = True   # False = skip search, use pre-validated defaults
N_CV_FOLDS       = 5
N_ITER_RF        = 15     # RandomizedSearchCV iterations for Random Forest
N_ITER_XGB       = 8      # Random configs tried via xgb.cv for XGBoost
N_ITER_LGBM      = 8      # Random configs tried via lgb.cv for LightGBM
RANDOM_STATE     = 42

# ── helpers ───────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def eval_model(name, y_true, y_pred, cv_rmse=None):
    r = {
        "Model":   name,
        "RMSE":    round(rmse(y_true, y_pred), 2),
        "MAE":     round(mean_absolute_error(y_true, y_pred), 2),
        "MAPE":    round(mean_absolute_percentage_error(y_true, y_pred) * 100, 3),
        "R2":      round(r2_score(y_true, y_pred), 4),
        "CV_RMSE": round(cv_rmse, 2) if cv_rmse is not None else None,
    }
    cv_str = f"  CV-RMSE={r['CV_RMSE']:8.1f}" if cv_rmse is not None else ""
    print(f"  {name:30s}  RMSE={r['RMSE']:8.1f}  MAE={r['MAE']:8.1f}  "
          f"MAPE={r['MAPE']:6.3f}%  R²={r['R2']:.4f}{cv_str}")
    return r


# ── 1. Load engineered data ───────────────────────────────────────────────────
print("Loading engineered dataset …")
df = pd.read_csv("data/train_engineered.csv")

TARGET = "Burnoff"

# ── 2. Feature sets ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "PlannedTripTime",
    "PlannedTOW",
    "PlannedZeroFuelWeight",
    "TeledyneRampWeight",
    "BlockTimeScheduled",
    "RouteDistanceKm",
    "LoadFactor",
    "TotalPassengers",
    "WeightDelta",
    "TypeWeightInteraction",
    "TypeTripTimeInteraction",
    "DepartureHour",
    "DepartureMonth",
    "DepartureDayOfWeek",
    "Freight",
]

# AOCDescription added per proposal §4.1 (was missing from original implementation)
CATEGORICAL_FEATURES = ["AircraftTypeGroup", "Carrier", "AOCDescription", "Season"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── 3. Train / val split ──────────────────────────────────────────────────────
df_clean = df.dropna(subset=ALL_FEATURES + [TARGET]).copy()
X = df_clean[ALL_FEATURES]
y = df_clean[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, shuffle=True
)
print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

# ── 4. Preprocessor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

CV_SPLITTER = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
hp_search_log = []

# ── 5. Hyperparameter search ──────────────────────────────────────────────────
# Pre-validated defaults used when TUNE_HYPERPARAMS = False
best_ridge_alpha    = 10.0
best_ridge_cv_rmse  = None

best_rf_defaults = dict(
    n_estimators=300, max_depth=15, min_samples_leaf=4, max_features="sqrt"
)
best_rf_params      = best_rf_defaults.copy()
best_rf_cv_rmse     = None

best_xgb_params = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1,
)
best_xgb_cv_rmse = None

best_lgbm_params = dict(
    n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
)
best_lgbm_cv_rmse = None

if not TUNE_HYPERPARAMS:
    print("\nTUNE_HYPERPARAMS=False — using pre-validated defaults, skipping search.")

# ── 5a. Ridge: GridSearchCV over log-spaced alphas ───────────────────────────
if TUNE_HYPERPARAMS:
    print("\n── Hyperparameter search: Ridge (GridSearchCV, 7 alphas × 5 folds) ──")
    ridge_search = GridSearchCV(
        Pipeline([("pre", clone(preprocessor)), ("model", Ridge())]),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]},
        cv=CV_SPLITTER,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    ridge_search.fit(X_train, y_train)
    best_ridge_alpha   = ridge_search.best_params_["model__alpha"]
    best_ridge_cv_rmse = -ridge_search.best_score_
    print(f"  Best alpha={best_ridge_alpha}  CV-RMSE={best_ridge_cv_rmse:.1f}")
    for params, score in zip(ridge_search.cv_results_["params"],
                              ridge_search.cv_results_["mean_test_score"]):
        hp_search_log.append({"Model": "Ridge", **params, "CV_RMSE": round(-score, 2)})

# ── 5b. Random Forest: RandomizedSearchCV ────────────────────────────────────
if TUNE_HYPERPARAMS:
    print(f"\n── Hyperparameter search: Random Forest (RandomizedSearch, {N_ITER_RF} iters × 5 folds) ──")
    rf_search = RandomizedSearchCV(
        Pipeline([
            ("pre", clone(preprocessor)),
            ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        param_distributions={
            "model__n_estimators":    [100, 200, 300, 400, 500],
            "model__max_depth":       [8, 10, 12, 15, 20, None],
            "model__min_samples_leaf":[1, 2, 4, 6, 8],
            "model__max_features":    ["sqrt", "log2", 0.4, 0.6, 0.8],
        },
        n_iter=N_ITER_RF,
        cv=CV_SPLITTER,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,        # sequential — RF itself uses n_jobs=-1 internally
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    rf_search.fit(X_train, y_train)
    best_rf_params    = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}
    best_rf_cv_rmse   = -rf_search.best_score_
    print(f"  Best params: {best_rf_params}  CV-RMSE={best_rf_cv_rmse:.1f}")
    for params, score in zip(rf_search.cv_results_["params"],
                              rf_search.cv_results_["mean_test_score"]):
        hp_search_log.append({"Model": "Random Forest",
                               **{k.replace("model__", ""): v for k, v in params.items()},
                               "CV_RMSE": round(-score, 2)})

# ── 5c. XGBoost: native xgb.cv random search ─────────────────────────────────
if TUNE_HYPERPARAMS and xgb is not None:
    print(f"\n── Hyperparameter search: XGBoost (xgb.cv, {N_ITER_XGB} configs × 5 folds) ──")
    _xgb_pre_tune = clone(preprocessor)
    X_tr_arr = _xgb_pre_tune.fit_transform(X_train)
    dtrain_xgb = xgb.DMatrix(X_tr_arr, label=y_train.values)

    rng = np.random.RandomState(RANDOM_STATE)
    xgb_param_dist = {
        "max_depth":        [4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":        [0.0, 0.05, 0.1, 0.5, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
        "min_child_weight": [1, 3, 5, 7],
    }
    _best_xgb_score = np.inf
    for i in range(N_ITER_XGB):
        trial = {k: rng.choice(v).item() for k, v in xgb_param_dist.items()}
        cv_result = xgb.cv(
            {**trial, "objective": "reg:squarederror",
             "seed": RANDOM_STATE, "verbosity": 0},
            dtrain_xgb,
            num_boost_round=800,
            nfold=N_CV_FOLDS,
            early_stopping_rounds=40,
            as_pandas=True,
            verbose_eval=False,
        )
        best_n     = int(cv_result["test-rmse-mean"].idxmin()) + 1
        trial_rmse = cv_result["test-rmse-mean"].min()
        hp_search_log.append({"Model": "XGBoost", **trial,
                               "n_estimators": best_n, "CV_RMSE": round(trial_rmse, 2)})
        print(f"  [{i+1}/{N_ITER_XGB}]  n_est={best_n:4d}  CV-RMSE={trial_rmse:.1f}")
        if trial_rmse < _best_xgb_score:
            _best_xgb_score  = trial_rmse
            best_xgb_params  = {**trial, "n_estimators": best_n}
            best_xgb_cv_rmse = trial_rmse
    print(f"  XGBoost best CV-RMSE={best_xgb_cv_rmse:.1f}  params={best_xgb_params}")

# ── 5d. LightGBM: native lgb.cv random search ────────────────────────────────
if TUNE_HYPERPARAMS and lgb is not None:
    print(f"\n── Hyperparameter search: LightGBM (lgb.cv, {N_ITER_LGBM} configs × 5 folds) ──")
    _lgbm_pre_tune = clone(preprocessor)
    X_tr_lgbm_arr = _lgbm_pre_tune.fit_transform(X_train)
    dtrain_lgbm = lgb.Dataset(X_tr_lgbm_arr, label=y_train.values, free_raw_data=False)

    rng = np.random.RandomState(RANDOM_STATE)
    lgbm_param_dist = {
        "max_depth":         [5, 6, 7, 8, 10],
        "num_leaves":        [31, 63, 127, 255],
        "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.1],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":         [0.0, 0.05, 0.1, 0.5, 1.0],
        "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
        "min_child_samples": [10, 20, 30, 50],
    }
    _best_lgbm_score = np.inf
    for i in range(N_ITER_LGBM):
        trial = {k: rng.choice(v).item() for k, v in lgbm_param_dist.items()}
        cv_params = {
            **trial,
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
            "n_jobs": -1,
            "seed": RANDOM_STATE,
        }
        try:
            cv_result = lgb.cv(
                cv_params,
                dtrain_lgbm,
                num_boost_round=1000,
                nfold=N_CV_FOLDS,
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            rmse_key = next(
                (k for k in cv_result if "rmse" in k.lower() and "mean" in k.lower()), None
            )
            if rmse_key is None:
                print(f"  [{i+1}/{N_ITER_LGBM}] Could not parse lgb.cv result keys: {list(cv_result.keys())}")
                continue
            best_n     = int(np.argmin(cv_result[rmse_key])) + 1
            trial_rmse = float(np.min(cv_result[rmse_key]))
        except Exception as e:
            print(f"  [{i+1}/{N_ITER_LGBM}] Trial failed: {e}")
            continue

        hp_search_log.append({"Model": "LightGBM", **trial,
                               "n_estimators": best_n, "CV_RMSE": round(trial_rmse, 2)})
        print(f"  [{i+1}/{N_ITER_LGBM}]  n_est={best_n:4d}  CV-RMSE={trial_rmse:.1f}")
        if trial_rmse < _best_lgbm_score:
            _best_lgbm_score  = trial_rmse
            best_lgbm_params  = {**trial, "n_estimators": best_n}
            best_lgbm_cv_rmse = trial_rmse
    print(f"  LightGBM best CV-RMSE={best_lgbm_cv_rmse:.1f}  params={best_lgbm_params}")

# Save HP search log regardless of whether tuning ran
if hp_search_log:
    pd.DataFrame(hp_search_log).to_csv("data/hp_search_results.csv", index=False)
    print("\nSaved data/hp_search_results.csv")

# ── 6. Final model training with best hyperparameters ────────────────────────
print("\n── Final model training ──")
results      = []
val_preds    = pd.DataFrame({"y_true": y_val.values}, index=y_val.index)
trained_models = {}

if LGBM_IMPORT_ERROR is not None:
    print(f"  LightGBM unavailable; skipping. Reason: {LGBM_IMPORT_ERROR}")
if XGB_IMPORT_ERROR is not None:
    print(f"  XGBoost unavailable; skipping. Reason: {XGB_IMPORT_ERROR}")

# ── Ridge ─────────────────────────────────────────────────────────────────────
print(f"\nTraining: Ridge Regression  (alpha={best_ridge_alpha})")
if TUNE_HYPERPARAMS:
    ridge_pipe = ridge_search.best_estimator_   # already fitted on full X_train
else:
    ridge_pipe = Pipeline([
        ("pre", clone(preprocessor)), ("model", Ridge(alpha=best_ridge_alpha))
    ])
    ridge_pipe.fit(X_train, y_train)

y_pred_ridge = ridge_pipe.predict(X_val)
val_preds["Ridge Regression"] = y_pred_ridge
results.append(eval_model("Ridge Regression", y_val.values, y_pred_ridge,
                           cv_rmse=best_ridge_cv_rmse))
trained_models["Ridge Regression"] = ridge_pipe
with open("models/ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_pipe, f)

# ── Random Forest ─────────────────────────────────────────────────────────────
print(f"\nTraining: Random Forest  (params={best_rf_params})")
if TUNE_HYPERPARAMS:
    rf_pipe = rf_search.best_estimator_         # already fitted on full X_train
else:
    rf_pipe = Pipeline([
        ("pre", clone(preprocessor)),
        ("model", RandomForestRegressor(
            **best_rf_params, n_jobs=-1, random_state=RANDOM_STATE
        )),
    ])
    rf_pipe.fit(X_train, y_train)

y_pred_rf = rf_pipe.predict(X_val)
val_preds["Random Forest"] = y_pred_rf
results.append(eval_model("Random Forest", y_val.values, y_pred_rf,
                           cv_rmse=best_rf_cv_rmse))
trained_models["Random Forest"] = rf_pipe
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_pipe, f)

# ── XGBoost ───────────────────────────────────────────────────────────────────
xgb_model     = None
xgb_pre_final = None
if xgb is not None:
    print(f"\nTraining: XGBoost  (params={best_xgb_params})")
    xgb_pre_final = clone(preprocessor)
    X_tr_xgb = xgb_pre_final.fit_transform(X_train)
    X_vl_xgb = xgb_pre_final.transform(X_val)

    n_est_xgb = best_xgb_params.pop("n_estimators", 500)
    xgb_model = xgb.XGBRegressor(
        **best_xgb_params,
        n_estimators=n_est_xgb,
        early_stopping_rounds=40,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_model.fit(X_tr_xgb, y_train,
                  eval_set=[(X_vl_xgb, y_val.values)],
                  verbose=False)
    y_pred_xgb = xgb_model.predict(X_vl_xgb)
    val_preds["XGBoost"] = y_pred_xgb
    results.append(eval_model("XGBoost", y_val.values, y_pred_xgb,
                               cv_rmse=best_xgb_cv_rmse))
    with open("models/xgboost.pkl", "wb") as f:
        pickle.dump((xgb_pre_final, xgb_model), f)

# ── LightGBM ──────────────────────────────────────────────────────────────────
lgbm_model    = None
num_pre_final = None
if lgb is not None:
    print(f"\nTraining: LightGBM  (params={best_lgbm_params})")
    num_pre_final = clone(preprocessor)
    X_tr_lgbm = num_pre_final.fit_transform(X_train)
    X_vl_lgbm = num_pre_final.transform(X_val)

    n_est_lgbm = best_lgbm_params.pop("n_estimators", 1000)
    lgbm_model = lgb.LGBMRegressor(
        **best_lgbm_params,
        n_estimators=n_est_lgbm,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    lgbm_model.fit(
        X_tr_lgbm, y_train,
        eval_set=[(X_vl_lgbm, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    y_pred_lgbm = lgbm_model.predict(X_vl_lgbm)
    val_preds["LightGBM"] = y_pred_lgbm
    results.append(eval_model("LightGBM", y_val.values, y_pred_lgbm,
                               cv_rmse=best_lgbm_cv_rmse))
    with open("models/lgbm_model.pkl", "wb") as f:
        pickle.dump((num_pre_final, lgbm_model), f)
    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(num_pre_final, f)

# ── 7. Save validation predictions ───────────────────────────────────────────
val_preds.to_csv("data/val_predictions.csv")
print("\nSaved data/val_predictions.csv")

# ── 8. Results table ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("data/model_results.csv", index=False)
print("\n--- Model Comparison ---")
print(results_df.to_string(index=False))

best_model_name  = results_df.sort_values("RMSE").iloc[0]["Model"]
best_model_preds = val_preds[best_model_name].values

# ── 9. Feature Importance ─────────────────────────────────────────────────────
rf_ohe        = rf_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
rf_feat_names = NUMERIC_FEATURES + list(rf_ohe)
rf_imp = pd.DataFrame({
    "Feature":    rf_feat_names,
    "Importance": rf_pipe.named_steps["model"].feature_importances_,
    "Model":      "Random Forest",
}).sort_values("Importance", ascending=False)

feat_imp_frames = [rf_imp.head(20)]

if lgbm_model is not None and num_pre_final is not None:
    ohe_cats   = num_pre_final.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feat_names = NUMERIC_FEATURES + list(ohe_cats)
    lgbm_imp = pd.DataFrame({
        "Feature":    feat_names[:len(lgbm_model.feature_importances_)],
        "Importance": lgbm_model.feature_importances_,
        "Model":      "LightGBM",
    }).sort_values("Importance", ascending=False)
    feat_imp_frames.insert(0, lgbm_imp.head(20))

# Ridge: absolute normalised coefficients as importance proxy
ridge_ohe        = ridge_pipe.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
ridge_feat_names = NUMERIC_FEATURES + list(ridge_ohe)
ridge_coefs      = np.abs(ridge_pipe.named_steps["model"].coef_)
ridge_imp = pd.DataFrame({
    "Feature":    ridge_feat_names[:len(ridge_coefs)],
    "Importance": ridge_coefs / ridge_coefs.sum(),
    "Model":      "Ridge Regression",
}).sort_values("Importance", ascending=False)
feat_imp_frames.append(ridge_imp.head(20))

# XGBoost
if xgb_model is not None and xgb_pre_final is not None:
    xgb_ohe        = xgb_pre_final.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    xgb_feat_names = NUMERIC_FEATURES + list(xgb_ohe)
    xgb_imp = pd.DataFrame({
        "Feature":    xgb_feat_names[:len(xgb_model.feature_importances_)],
        "Importance": xgb_model.feature_importances_,
        "Model":      "XGBoost",
    }).sort_values("Importance", ascending=False)
    feat_imp_frames.append(xgb_imp.head(20))

feat_imp_combined = pd.concat(feat_imp_frames)
feat_imp_combined.to_csv("data/feature_importance.csv", index=False)
print("\nSaved data/feature_importance.csv")

# ── 10. Figures ───────────────────────────────────────────────────────────────

# 10a. Model comparison — RMSE and MAPE
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
palette = ["#2563EB", "#7C3AED", "#DC2626", "#16A34A"]
for i, metric in enumerate(["RMSE", "MAPE"]):
    axes[i].bar(results_df["Model"], results_df[metric],
                color=palette[:len(results_df)], edgecolor="white")
    axes[i].set_title(f"Model Comparison — {metric}")
    axes[i].set_ylabel(metric)
    axes[i].tick_params(axis="x", rotation=15)
    for bar, val in zip(axes[i].patches, results_df[metric]):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01 * bar.get_height(),
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("figures/model_comparison.png")
plt.close()

# 10b. CV-RMSE vs Val-RMSE (shows tuning quality; overfitting gap visible)
cv_plot = results_df[["Model", "RMSE", "CV_RMSE"]].dropna(subset=["CV_RMSE"])
if not cv_plot.empty:
    x = np.arange(len(cv_plot))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, cv_plot["CV_RMSE"], w,
           label="CV-RMSE (during tuning)", color="#2563EB", alpha=0.85)
    ax.bar(x + w / 2, cv_plot["RMSE"], w,
           label="Val-RMSE (holdout)",     color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cv_plot["Model"], rotation=10)
    ax.set(title="Cross-Validation RMSE vs Holdout Val-RMSE\n(gap = overfitting signal)",
           ylabel="RMSE (kg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/model_cv_vs_val_rmse.png")
    plt.close()

# 10c. Top-20 feature importance
fig, ax = plt.subplots(figsize=(9, 7))
feature_plot_model = ("LightGBM" if "LightGBM" in feat_imp_combined["Model"].unique()
                      else "Random Forest")
top20 = feat_imp_combined[feat_imp_combined["Model"] == feature_plot_model].head(20)
ax.barh(top20["Feature"][::-1], top20["Importance"][::-1], color="#2563EB")
ax.set_title(f"{feature_plot_model} – Top 20 Feature Importances")
ax.set_xlabel("Gain")
plt.tight_layout()
plt.savefig("figures/feature_importance_lgbm.png")
plt.close()

# 10d. Predicted vs Actual — best model
fig, ax = plt.subplots(figsize=(7, 7))
sample_idx = np.random.choice(len(y_val), min(3000, len(y_val)), replace=False)
ax.scatter(y_val.values[sample_idx], best_model_preds[sample_idx],
           alpha=0.25, s=8, c="#2563EB")
lims = [min(y_val.min(), best_model_preds.min()),
        max(y_val.max(), best_model_preds.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set(title=f"{best_model_name}: Predicted vs Actual Burnoff",
       xlabel="Actual Burnoff (kg)", ylabel="Predicted Burnoff (kg)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/model_predicted_vs_actual.png")
plt.close()

# 10e. Residuals by aircraft type
val_df_plot = pd.DataFrame({
    "y_true":           y_val.values,
    "y_pred":           best_model_preds,
    "Residual":         best_model_preds - y_val.values,
    "AircraftTypeGroup":df_clean.loc[y_val.index, "AircraftTypeGroup"].values,
})
fig, ax = plt.subplots(figsize=(8, 5))
order = val_df_plot.groupby("AircraftTypeGroup")["Residual"].median().abs().sort_values().index
sns.boxplot(data=val_df_plot, x="AircraftTypeGroup", y="Residual", order=order,
            palette="Set2", ax=ax, width=0.5)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set(title=f"{best_model_name} Residuals by Aircraft Type",
       xlabel="Aircraft Type", ylabel="Residual (predicted − actual, kg)")
plt.tight_layout()
plt.savefig("figures/model_residuals_by_aircraft.png")
plt.close()

# 10f. Residuals histogram
fig, ax = plt.subplots(figsize=(8, 4))
residuals_all = best_model_preds - y_val.values
ax.hist(residuals_all, bins=80, color="#2563EB", edgecolor="white", linewidth=0.3)
ax.axvline(0, color="red", linestyle="--")
ax.set(title=f"{best_model_name} – Residual Distribution",
       xlabel="Residual (kg)", ylabel="Count")
plt.tight_layout()
plt.savefig("figures/model_residuals_hist.png")
plt.close()

# 10g. HP search RMSE distribution (if tuning ran)
if hp_search_log:
    hp_df = pd.DataFrame(hp_search_log)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, grp in hp_df.groupby("Model"):
        ax.scatter([model_name] * len(grp), grp["CV_RMSE"],
                   alpha=0.6, s=40, label=model_name)
    # Highlight best per model
    best_per_model = hp_df.groupby("Model")["CV_RMSE"].min()
    for i, (model_name, best_cv) in enumerate(best_per_model.items()):
        ax.scatter(model_name, best_cv, marker="*", s=200, zorder=5, color="gold",
                   edgecolors="black", linewidths=0.8)
    ax.set(title="Hyperparameter Search: CV-RMSE Distribution per Model\n(★ = best config)",
           xlabel="Model", ylabel="CV-RMSE (kg)")
    plt.tight_layout()
    plt.savefig("figures/model_hp_search_cv_rmse.png")
    plt.close()

print("\n✅ Model training complete. All outputs saved.")
