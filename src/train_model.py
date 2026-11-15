"""Step 3 - Model building with experiment tracking (MLflow).

Loads train/test splits from the workflow artifact folder, tunes several tree-based
models (Random Forest, Gradient Boosting, XGBoost) with GridSearchCV, logs every
parameter combination and the final metrics to MLflow, and saves the best model
(by cross-validated RMSE) to model/best_model.joblib so the pipeline can commit it.
"""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

TARGET = "Product_Store_Sales_Total"
RANDOM_STATE = 42

# Hyper-parameter grids to tune
MODELS = {
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200], "model__max_depth": [8, 12, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [3, 4]},
    ),
    "XGBoost": (
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, objective="reg:squarederror"),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [3, 4, 6], "model__subsample": [0.8, 1.0]},
    ),
}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = X.select_dtypes("object").columns.tolist()
    num = X.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    return {"rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--model-out", default="model/best_model.joblib")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    train = pd.read_csv(Path(args.artifacts) / "train.csv")
    test = pd.read_csv(Path(args.artifacts) / "test.csv")
    X_train, y_train = train.drop(columns=TARGET), train[TARGET]
    X_test, y_test = test.drop(columns=TARGET), test[TARGET]

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("SuperKart_Sales_Forecast")

    results, best = [], {"cv_rmse": np.inf}
    for name, (estimator, grid) in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
        search = GridSearchCV(pipe, grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            cv_rmse = float(-search.best_score_)

            # Log every tuned combination as a nested run
            cvr = search.cv_results_
            for i, params in enumerate(cvr["params"]):
                with mlflow.start_run(run_name=f"{name}_trial_{i}", nested=True):
                    mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                    mlflow.log_metric("cv_rmse", float(-cvr["mean_test_score"][i]))

            train_m = evaluate(search.best_estimator_, X_train, y_train)
            test_m = evaluate(search.best_estimator_, X_test, y_test)
            mlflow.log_params({k.replace("model__", "best_"): v for k, v in search.best_params_.items()})
            mlflow.log_metric("cv_rmse", cv_rmse)
            for k, v in train_m.items():
                mlflow.log_metric(f"train_{k}", v)
            for k, v in test_m.items():
                mlflow.log_metric(f"test_{k}", v)

        print(f"{name:17s} CV RMSE={cv_rmse:9.2f} | test RMSE={test_m['rmse']:9.2f} "
              f"| test R2={test_m['r2']:.4f} | best params={search.best_params_}")
        results.append({"model": name, "cv_rmse": cv_rmse, "train_rmse": train_m["rmse"],
                        "test_rmse": test_m["rmse"], "test_mae": test_m["mae"],
                        "test_r2": test_m["r2"], "best_params": search.best_params_})
        if cv_rmse < best["cv_rmse"]:
            best = {"cv_rmse": cv_rmse, "name": name, "estimator": search.best_estimator_}

    # Save best model and a metrics report (committed to the repo by the pipeline)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["estimator"], args.model_out)
    Path(args.model_out).with_name("metrics.json").write_text(
        json.dumps({"best_model": best["name"], "results": results}, indent=2, default=str))
    print(f"\nBest model: {best['name']} (CV RMSE {best['cv_rmse']:.2f}) saved to {args.model_out}")
