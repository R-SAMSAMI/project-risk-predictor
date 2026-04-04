from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "project_type",
    "region",
    "contract_type",
    "budget_musd",
    "planned_duration_days",
    "crew_size",
    "subcontractor_count",
    "change_order_count",
    "safety_incidents",
    "permit_delay_days",
    "client_decision_latency",
    "weather_severity",
    "material_risk",
    "labor_availability",
    "site_complexity",
    "site_density",
    "percent_self_performed",
    "equipment_utilization",
]

NUMERIC_COLUMNS = [
    "budget_musd",
    "planned_duration_days",
    "crew_size",
    "subcontractor_count",
    "change_order_count",
    "safety_incidents",
    "permit_delay_days",
    "client_decision_latency",
    "percent_self_performed",
    "equipment_utilization",
]

CATEGORICAL_COLUMNS = [
    "project_type",
    "region",
    "contract_type",
    "weather_severity",
    "material_risk",
    "labor_availability",
    "site_complexity",
    "site_density",
]


@dataclass
class TrainingBundle:
    delay_classifier: Pipeline
    cost_classifier: Pipeline
    delay_regressor: Pipeline
    metrics: dict[str, float]
    feature_importances: pd.DataFrame


def _build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def _build_classifier(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _build_regressor(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=260,
                    max_depth=11,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_models(data: pd.DataFrame, random_state: int = 42) -> TrainingBundle:
    X = data[FEATURE_COLUMNS]
    y_delay = data["delayed"]
    y_budget = data["over_budget"]
    y_delay_days = data["delay_days"]

    train_idx, test_idx = train_test_split(
        data.index,
        test_size=0.2,
        random_state=random_state,
        stratify=y_delay,
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_delay_train, y_delay_test = y_delay.loc[train_idx], y_delay.loc[test_idx]
    y_budget_train, y_budget_test = y_budget.loc[train_idx], y_budget.loc[test_idx]
    y_delay_days_train, y_delay_days_test = y_delay_days.loc[train_idx], y_delay_days.loc[test_idx]

    delay_classifier = _build_classifier(random_state)
    cost_classifier = _build_classifier(random_state + 7)
    delay_regressor = _build_regressor(random_state + 21)

    delay_classifier.fit(X_train, y_delay_train)
    cost_classifier.fit(X_train, y_budget_train)
    delay_regressor.fit(X_train, y_delay_days_train)

    delay_auc = roc_auc_score(y_delay_test, delay_classifier.predict_proba(X_test)[:, 1])
    budget_auc = roc_auc_score(y_budget_test, cost_classifier.predict_proba(X_test)[:, 1])
    delay_mae = mean_absolute_error(y_delay_days_test, delay_regressor.predict(X_test))

    feature_importances = _get_feature_importances(delay_classifier)

    return TrainingBundle(
        delay_classifier=delay_classifier,
        cost_classifier=cost_classifier,
        delay_regressor=delay_regressor,
        metrics={
            "delay_auc": round(float(delay_auc), 3),
            "budget_auc": round(float(budget_auc), 3),
            "delay_mae": round(float(delay_mae), 2),
            "records": len(data),
        },
        feature_importances=feature_importances,
    )


def _get_feature_importances(model_pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance = model.feature_importances_
    frame = pd.DataFrame({"feature": feature_names, "importance": importance})
    frame["feature"] = frame["feature"].str.replace("numeric__", "", regex=False)
    frame["feature"] = frame["feature"].str.replace("categorical__one_hot__", "", regex=False)
    return frame.sort_values("importance", ascending=False).head(12).reset_index(drop=True)


def predict_project(bundle: TrainingBundle, project_input: dict[str, object]) -> dict[str, float]:
    row = pd.DataFrame([project_input])[FEATURE_COLUMNS]
    delay_probability = bundle.delay_classifier.predict_proba(row)[0, 1]
    budget_probability = bundle.cost_classifier.predict_proba(row)[0, 1]
    expected_delay_days = bundle.delay_regressor.predict(row)[0]
    return {
        "delay_probability": float(delay_probability),
        "budget_probability": float(budget_probability),
        "expected_delay_days": max(0.0, float(expected_delay_days)),
    }
