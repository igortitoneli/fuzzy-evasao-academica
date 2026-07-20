from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data, evaluate
from src.evaluate import ClassificationMetrics
from src.experiment import run_experiment
from src.pipeline import fit_pipeline
from src.visualize import build_heatmap


def test_split_is_stratified_and_disjoint(synthetic_students: pd.DataFrame) -> None:
    split = evaluate.stratified_split(synthetic_students)
    assert (len(split.train), len(split.validation), len(split.test)) == (84, 18, 18)
    sets = [set(frame["row_id"]) for frame in (split.train, split.validation, split.test)]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    overall = synthetic_students["y_true"].mean()
    assert all(abs(frame["y_true"].mean() - overall) < 0.08 for frame in (split.train, split.validation, split.test))


def _metric(threshold: float, f2: float) -> ClassificationMetrics:
    return ClassificationMetrics(threshold, 0.0, 0.0, 0.0, 0.0, f2, 0.0, 0.0, 0.0, 0.0, [[0, 0], [0, 0]])


def test_threshold_tie_uses_lower_value() -> None:
    selected = evaluate.select_f2_threshold([_metric(10.0, 0.7), _metric(11.0, 0.7)])
    assert selected.threshold == 10.0


def test_pipeline_is_deterministic_and_predicts_every_row(synthetic_students: pd.DataFrame) -> None:
    split = evaluate.stratified_split(synthetic_students)
    first = fit_pipeline(split.train, random_state=42, folds=3)
    second = fit_pipeline(split.train, random_state=42, folds=3)
    first_risk = first.predict(split.validation).risco_evasao
    second_risk = second.predict(split.validation).risco_evasao
    assert len(first_risk) == len(split.validation)
    np.testing.assert_allclose(first_risk, second_risk)


def test_heatmap_has_requested_resolution(synthetic_students: pd.DataFrame) -> None:
    split = evaluate.stratified_split(synthetic_students)
    pipeline = fit_pipeline(split.train, folds=3)
    result = build_heatmap(
        pipeline.academic_model,
        name="academico",
        x_variable="nota_academica",
        y_variable="aprovadas",
        fixed_values={"sem_avaliacao": float(split.train["sem_avaliacao"].median())},
    )
    assert result.risks.shape == (20, 20)
    assert result.fixed_values.keys() == {"sem_avaliacao"}


def test_heatmap_can_keep_binary_axis_discrete(synthetic_students: pd.DataFrame) -> None:
    split = evaluate.stratified_split(synthetic_students)
    pipeline = fit_pipeline(split.train, folds=3)
    result = build_heatmap(
        pipeline.demographic_model,
        name="demografico",
        x_variable="idade",
        y_variable="deslocado",
        fixed_values={"noturno": 0.0, "internacional": 0.0},
        resolution=20,
        y_resolution=2,
    )
    assert result.risks.shape == (2, 20)
    np.testing.assert_array_equal(result.y_values, np.asarray([0.0, 1.0]))


def test_small_subgroups_are_marked_insufficient(synthetic_students: pd.DataFrame) -> None:
    frame = synthetic_students.iloc[:20].copy()
    frame["risco_evasao"] = np.linspace(0.0, 100.0, len(frame))
    groups = evaluate.subgroup_metrics(frame, threshold=50.0)
    assert not groups["sufficient"].any()


def test_end_to_end_experiment_is_offline(synthetic_students: pd.DataFrame) -> None:
    result = run_experiment(synthetic_students, folds=3, bootstrap_iterations=5)
    assert 0.0 <= result.threshold <= 100.0
    assert len(result.test_predictions.risco_evasao) == len(result.split.test)
    assert result.pipeline.final_model.rules


@pytest.mark.skipif(not Path(data.RAW_PATH).exists(), reason="cache UCI nao disponivel")
def test_cached_dataset_integration() -> None:
    frame = data.build_features(data.load_raw(use_cache=True))
    result = run_experiment(frame, folds=3, bootstrap_iterations=5)
    assert len(result.split.test) > 0
