"""Повторяет сравнение моделей и сохраняет измеренные метрики."""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline

from ensemble_models import _candidates, _evaluate, _preprocess, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


def main() -> None:
    root = Path(__file__).resolve().parent
    seed = 42
    data = pd.read_csv(root / "german.csv", sep=";")
    target = data.pop("Creditability")
    train, test, train_target, test_target = train_test_split(
        data, target, test_size=0.2, random_state=seed, stratify=target,
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates = []
    for name, estimator in _candidates(seed).items():
        model = make_pipeline(_preprocess(train), estimator)
        probabilities = cross_val_predict(
            model, train, train_target, cv=folds, method="predict_proba",
        )[:, 1]
        candidates.append({"model_name": name, **_evaluate(train_target, probabilities)})
    result = run_ensemble_experiment(EnsembleInput(train, train_target, test, test_target, seed))
    test_metrics = {
        metric: getattr(result, metric)
        for metric in ("roc_auc", "accuracy", "precision", "recall", "f1")
    }
    results = {
        "random_seed": seed,
        "test_size": 0.2,
        "cv_folds": folds.n_splits,
        "selection_metric": "roc_auc",
        "cross_validation": candidates,
        "model_name": result.model_name,
        **test_metrics,
    }
    output = json.dumps(results, indent=2) + "\n"
    (root / "experiment_results.json").write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
