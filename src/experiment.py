"""Orquestracao do experimento completo e exportacao de artefatos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np
import pandas as pd

from . import evaluate, visualize
from .evaluate import ClassificationMetrics, ExperimentSplit
from .pipeline import FuzzyPipeline, PredictionResult, StudentExplanation, fit_pipeline


@dataclass(frozen=True)
class ExperimentResult:
    split: ExperimentSplit
    pipeline: FuzzyPipeline
    train_predictions: PredictionResult
    validation_predictions: PredictionResult
    test_predictions: PredictionResult
    validation_sweep: tuple[ClassificationMetrics, ...]
    selected_validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    confidence_intervals: Mapping[str, Mapping[str, float]]
    subgroup_table: pd.DataFrame
    representative_row_ids: Mapping[str, int]
    explanations: Mapping[str, StudentExplanation]
    threshold_source: str

    @property
    def threshold(self) -> float:
        return self.test_metrics.threshold


def _attach_predictions(df: pd.DataFrame, predictions: PredictionResult) -> pd.DataFrame:
    return df.join(predictions.to_frame(df.index))


def run_experiment(
    df: pd.DataFrame,
    *,
    threshold_override: float | None = None,
    random_state: int = 42,
    folds: int = 5,
    bootstrap_iterations: int = 1000,
) -> ExperimentResult:
    split = evaluate.stratified_split(df, random_state=random_state)
    fitted = fit_pipeline(split.train, random_state=random_state, folds=folds)
    train_predictions = fitted.predict(split.train)
    validation_predictions = fitted.predict(split.validation)
    test_predictions = fitted.predict(split.test)
    validation_y = split.validation["y_true"].to_numpy(dtype=np.int64)
    sweep = tuple(evaluate.threshold_sweep(validation_y, validation_predictions.risco_evasao))
    selected = evaluate.select_f2_threshold(sweep)
    threshold = float(threshold_override) if threshold_override is not None else selected.threshold
    threshold_source = "cli_override" if threshold_override is not None else "validation_f2"
    test_y = split.test["y_true"].to_numpy(dtype=np.int64)
    test_metrics = evaluate.metrics_at_threshold(test_y, test_predictions.risco_evasao, threshold)
    intervals = evaluate.bootstrap_confidence_intervals(
        test_y,
        test_predictions.risco_evasao,
        threshold,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )
    test_with_predictions = _attach_predictions(split.test, test_predictions)
    subgroup_table = evaluate.subgroup_metrics(test_with_predictions, threshold=threshold)
    row_ids = evaluate.representative_cases(test_with_predictions, threshold=threshold)
    explanations = {
        label: fitted.explain(test_with_predictions.loc[test_with_predictions["row_id"] == row_id].iloc[0])
        for label, row_id in row_ids.items()
    }
    return ExperimentResult(
        split=split,
        pipeline=fitted,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_sweep=sweep,
        selected_validation_metrics=selected,
        test_metrics=test_metrics,
        confidence_intervals=intervals,
        subgroup_table=subgroup_table,
        representative_row_ids=row_ids,
        explanations=explanations,
        threshold_source=threshold_source,
    )


def export_experiment(result: ExperimentResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.pipeline.export_rules(destination)

    prediction_frames: list[pd.DataFrame] = []
    for split_name, frame, predictions in (
        ("train", result.split.train, result.train_predictions),
        ("validation", result.split.validation, result.validation_predictions),
        ("test", result.split.test, result.test_predictions),
    ):
        exported = _attach_predictions(frame, predictions)
        exported.insert(0, "split", split_name)
        prediction_frames.append(exported)
    prediction_table = pd.concat(prediction_frames).sort_values(["split", "row_id"])
    prediction_table.to_csv(destination / "predictions.csv", index=False)
    result.subgroup_table.to_csv(destination / "subgroup_metrics.csv", index=False)

    metrics_payload = {
        "threshold_source": result.threshold_source,
        "selected_on_validation": result.selected_validation_metrics.to_dict(),
        "test": result.test_metrics.to_dict(),
        "bootstrap_95_ci": result.confidence_intervals,
        "split_sizes": {
            "train": len(result.split.train),
            "validation": len(result.split.validation),
            "test": len(result.split.test),
        },
        "representative_row_ids": dict(result.representative_row_ids),
    }
    evaluate.save_json(metrics_payload, destination / "metrics.json")
    evaluate.save_json(
        [item.to_dict() for item in result.validation_sweep],
        destination / "validation_threshold_sweep.json",
    )
    evaluate.save_json(
        {label: asdict(explanation) for label, explanation in result.explanations.items()},
        destination / "case_explanations.json",
    )

    visualize.plot_system_diagram(destination / "system_diagram.png")
    for name, model in result.pipeline.models.items():
        visualize.plot_memberships(model, destination / f"memberships_{name}.png")
    visualize.plot_split_distribution(
        {"Treino": result.split.train, "Validacao": result.split.validation, "Teste": result.split.test},
        destination / "split_distribution.png",
    )
    visualize.plot_threshold_selection(
        result.validation_sweep,
        result.selected_validation_metrics.threshold,
        destination / "validation_f2_threshold.png",
    )
    test_frame = _attach_predictions(result.split.test, result.test_predictions)
    visualize.plot_test_evaluation(
        test_frame["y_true"].to_numpy(dtype=np.int64),
        result.test_predictions.risco_evasao,
        result.test_metrics,
        destination,
    )
    visualize.plot_risk_distribution(test_frame, result.threshold, destination / "risk_distribution_test.png")
    heatmaps = visualize.generate_heatmaps(result.pipeline, result.split.train, destination)
    evaluate.save_json(
        {
            name: {
                "x_variable": item.x_variable,
                "y_variable": item.y_variable,
                "fixed_values": dict(item.fixed_values),
                "resolution": list(item.risks.shape),
            }
            for name, item in heatmaps.items()
        },
        destination / "heatmap_metadata.json",
    )
    for label, explanation in result.explanations.items():
        visualize.plot_case_explanation(explanation, label, destination / f"case_{label}.png")

    # Mantem nomes historicos apontando para figuras do protocolo atual.
    for current_name, legacy_name in (
        ("confusion_matrix_test.png", "confusion_matrix.png"),
        ("risk_distribution_test.png", "risk_distribution.png"),
        ("validation_f2_threshold.png", "threshold_sweep.png"),
        ("curvas_demografico.png", "heatmap_demografico.png"),
    ):
        shutil.copyfile(destination / current_name, destination / legacy_name)
