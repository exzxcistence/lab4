"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


NOMINAL_COLUMNS = {
    "Purpose", "Sex_Marital_Status", "Guarantors",
    "Most_valuable_available_asset", "Concurrent_Credits",
    "Type_of_apartment", "Occupation", "Telephone", "Foreign_Worker",
}


def _preprocess(features: pd.DataFrame) -> ColumnTransformer:
    numeric = [
        name for name, dtype in features.dtypes.items()
        if name not in NOMINAL_COLUMNS and pd.api.types.is_numeric_dtype(dtype)
    ]
    categorical = [name for name in features.columns if name not in numeric]
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), categorical),
    ])


def _candidates(seed: int) -> dict:
    shallow_boost = GradientBoostingClassifier(
        n_estimators=180, learning_rate=0.035, max_depth=1, random_state=seed,
    )
    deeper_boost = GradientBoostingClassifier(
        n_estimators=160, learning_rate=0.035, max_depth=2, random_state=seed,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,), solver="lbfgs", alpha=8.0,
        max_iter=1000, random_state=seed,
    )
    return {
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, max_features=0.8, random_state=seed,
        ),
        "Gradient Boosting (depth 1)": shallow_boost,
        "Gradient Boosting (depth 2)": deeper_boost,
        "MLP (8 neurons)": mlp,
        "Soft Voting (GB + MLP)": VotingClassifier(
            estimators=[("shallow", shallow_boost), ("deeper", deeper_boost), ("mlp", mlp)],
            voting="soft",
        ),
    }


def _evaluate(target: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает модель на train и возвращает предсказания и метрики на test."""
    train = pd.DataFrame(data.train_features)
    test = pd.DataFrame(data.test_features, columns=train.columns)
    train_target = np.asarray(data.train_target).ravel()
    test_target = np.asarray(data.test_target).ravel()
    model_name = "Soft Voting (GB + MLP)"
    model = make_pipeline(_preprocess(train), _candidates(data.random_seed)[model_name])
    model.fit(train, train_target)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)
    return EnsembleResult(
        model_name=model_name,
        probabilities=probabilities,
        predictions=predictions,
        **_evaluate(test_target, probabilities),
    )
