"""Separacao experimental, selecao de limiar e avaliacao sem vazamento."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


@dataclass(frozen=True)
class ExperimentSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ClassificationMetrics:
    threshold: float
    accuracy: float
    precision_dropout: float
    recall_dropout: float
    f1_dropout: float
    f2_dropout: float
    specificity: float
    balanced_accuracy: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stratified_split(df: pd.DataFrame, *, random_state: int = 42) -> ExperimentSplit:
    """Cria particoes estratificadas e disjuntas de 70%, 15% e 15%."""

    train, remainder = train_test_split(
        df,
        test_size=0.30,
        stratify=df["y_true"],
        random_state=random_state,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        stratify=remainder["y_true"],
        random_state=random_state,
    )
    id_sets = [set(part["row_id"].tolist()) for part in (train, validation, test)]
    if id_sets[0] & id_sets[1] or id_sets[0] & id_sets[2] or id_sets[1] & id_sets[2]:
        raise RuntimeError("As particoes experimentais nao sao disjuntas")
    return ExperimentSplit(
        train=train.sort_index().copy(),
        validation=validation.sort_index().copy(),
        test=test.sort_index().copy(),
    )


def predict_class(risk: FloatArray, threshold: float) -> IntArray:
    return np.asarray(risk >= threshold, dtype=np.int64)


def metrics_at_threshold(y_true: IntArray, risk: FloatArray, threshold: float) -> ClassificationMetrics:
    y = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(risk, dtype=np.float64)
    predicted = predict_class(scores, threshold)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity_denominator = true_negative + false_positive
    specificity = float(true_negative / specificity_denominator) if specificity_denominator else 0.0
    has_both_classes = len(np.unique(y)) == 2
    return ClassificationMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y, predicted)),
        precision_dropout=float(precision_score(y, predicted, zero_division=0)),
        recall_dropout=float(recall_score(y, predicted, zero_division=0)),
        f1_dropout=float(f1_score(y, predicted, zero_division=0)),
        f2_dropout=float(fbeta_score(y, predicted, beta=2.0, zero_division=0)),
        specificity=specificity,
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)) if has_both_classes else float("nan"),
        roc_auc=float(roc_auc_score(y, scores)) if has_both_classes else float("nan"),
        pr_auc=float(average_precision_score(y, scores)) if has_both_classes else float("nan"),
        confusion_matrix=[[int(value) for value in row] for row in matrix.tolist()],
    )


def threshold_sweep(y_true: IntArray, risk: FloatArray) -> list[ClassificationMetrics]:
    return [metrics_at_threshold(y_true, risk, float(threshold)) for threshold in range(101)]


def select_f2_threshold(results: Sequence[ClassificationMetrics]) -> ClassificationMetrics:
    """Seleciona maior F2; a ordem crescente garante desempate pelo menor limiar."""

    if not results:
        raise ValueError("A varredura de limiar nao pode estar vazia")
    best_f2 = max(result.f2_dropout for result in results)
    return next(result for result in results if np.isclose(result.f2_dropout, best_f2))


def bootstrap_confidence_intervals(
    y_true: IntArray,
    risk: FloatArray,
    threshold: float,
    *,
    iterations: int = 1000,
    random_state: int = 42,
) -> Mapping[str, Mapping[str, float]]:
    """IC percentil de 95% com reamostragem estratificada por classe."""

    y = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(risk, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    metric_names = (
        "accuracy", "precision_dropout", "recall_dropout", "f1_dropout",
        "f2_dropout", "specificity", "balanced_accuracy", "roc_auc", "pr_auc",
    )
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(iterations):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        metrics = metrics_at_threshold(y[selected], scores[selected], threshold)
        for name in metric_names:
            samples[name].append(float(getattr(metrics, name)))
    return {
        name: {
            "lower": float(np.nanpercentile(values, 2.5)),
            "upper": float(np.nanpercentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def subgroup_metrics(df: pd.DataFrame, *, threshold: float, minimum_size: int = 30) -> pd.DataFrame:
    """Calcula diagnosticos descritivos; grupos pequenos sao apenas sinalizados."""

    working = df.copy()
    working["faixa_etaria"] = pd.cut(
        working["idade"], bins=[16, 22, 30, 40, np.inf],
        labels=["17-22", "23-30", "31-40", "41+"], include_lowest=True,
    )
    group_columns = {
        "faixa_etaria": "faixa_etaria",
        "turno_noturno": "noturno",
        "bolsa": "scholarship",
        "internacional": "internacional",
        "deslocado": "deslocado",
    }
    records: list[dict[str, Any]] = []
    for dimension, column in group_columns.items():
        for value, group in working.groupby(column, observed=True, dropna=False):
            y = group["y_true"].to_numpy(dtype=np.int64)
            risk = group["risco_evasao"].to_numpy(dtype=np.float64)
            metrics = metrics_at_threshold(y, risk, threshold)
            has_dropout = bool(np.any(y == 1))
            records.append(
                {
                    "dimension": dimension,
                    "group": str(value),
                    "n": len(group),
                    "sufficient": len(group) >= minimum_size,
                    "precision_dropout": metrics.precision_dropout,
                    "recall_dropout": metrics.recall_dropout if has_dropout else float("nan"),
                    "false_negative_rate": 1.0 - metrics.recall_dropout if has_dropout else float("nan"),
                    "f2_dropout": metrics.f2_dropout if has_dropout else float("nan"),
                    "accuracy": metrics.accuracy,
                }
            )
    return pd.DataFrame.from_records(records)


def representative_cases(df: pd.DataFrame, *, threshold: float) -> Mapping[str, int]:
    predicted = predict_class(df["risco_evasao"].to_numpy(dtype=np.float64), threshold)
    y = df["y_true"].to_numpy(dtype=np.int64)
    risk = df["risco_evasao"].to_numpy(dtype=np.float64)
    definitions = {
        "verdadeiro_positivo": ((y == 1) & (predicted == 1), "max"),
        "verdadeiro_negativo": ((y == 0) & (predicted == 0), "min"),
        "falso_positivo": ((y == 0) & (predicted == 1), "max"),
        "falso_negativo": ((y == 1) & (predicted == 0), "min"),
    }
    selected: dict[str, int] = {}
    for label, (mask, direction) in definitions.items():
        positions = np.flatnonzero(mask)
        if not len(positions):
            continue
        local = risk[positions]
        position = positions[int(np.argmax(local) if direction == "max" else np.argmin(local))]
        selected[label] = int(df.iloc[position]["row_id"])
    return selected


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
