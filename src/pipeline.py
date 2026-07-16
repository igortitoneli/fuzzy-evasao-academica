"""Treinamento empilhado e inferencia dos quatro sistemas fuzzy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from .blocks import academic, demographic, final, social
from .membership import BlockExplanation, FuzzyVariable, MamdaniModel, induce_rules

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PredictionResult:
    risco_academico: FloatArray
    risco_social: FloatArray
    risco_demografico: FloatArray
    risco_evasao: FloatArray

    def to_frame(self, index: pd.Index | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "risco_academico": self.risco_academico,
                "risco_social": self.risco_social,
                "risco_demografico": self.risco_demografico,
                "risco_evasao": self.risco_evasao,
            },
            index=index,
        )


@dataclass(frozen=True)
class StudentExplanation:
    input_values: Mapping[str, float]
    risks: Mapping[str, float]
    blocks: Mapping[str, BlockExplanation]


def _inputs(df: pd.DataFrame, variables: tuple[FuzzyVariable, ...]) -> Mapping[str, FloatArray]:
    return {
        variable.name: df[variable.name].to_numpy(dtype=np.float64)
        for variable in variables
    }


def _fit_block(name: str, variables: tuple[FuzzyVariable, ...], df: pd.DataFrame) -> MamdaniModel:
    return induce_rules(name, variables, _inputs(df, variables), df["y_true"].to_numpy(dtype=np.float64))


def _predict_block(model: MamdaniModel, df: pd.DataFrame) -> FloatArray:
    return model.infer(_inputs(df, model.variables)).risks


@dataclass(frozen=True)
class FuzzyPipeline:
    academic_model: MamdaniModel
    social_model: MamdaniModel
    demographic_model: MamdaniModel
    final_model: MamdaniModel

    @property
    def models(self) -> Mapping[str, MamdaniModel]:
        return {
            "academico": self.academic_model,
            "social": self.social_model,
            "demografico": self.demographic_model,
            "final": self.final_model,
        }

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        academic_risk = _predict_block(self.academic_model, df)
        social_risk = _predict_block(self.social_model, df)
        demographic_risk = _predict_block(self.demographic_model, df)
        final_inputs = pd.DataFrame(
            {
                "risco_academico": academic_risk,
                "risco_social": social_risk,
                "risco_demografico": demographic_risk,
            },
            index=df.index,
        )
        final_risk = _predict_block(self.final_model, final_inputs)
        return PredictionResult(academic_risk, social_risk, demographic_risk, final_risk)

    def explain(self, row: pd.Series, *, top_k: int = 5) -> StudentExplanation:
        block_explanations: dict[str, BlockExplanation] = {}
        intermediate: dict[str, float] = {}
        for name, model, output_name in (
            ("academico", self.academic_model, "risco_academico"),
            ("social", self.social_model, "risco_social"),
            ("demografico", self.demographic_model, "risco_demografico"),
        ):
            values = {variable.name: float(row[variable.name]) for variable in model.variables}
            explanation = model.explain(values, top_k=top_k)
            block_explanations[name] = explanation
            intermediate[output_name] = explanation.risk
        final_explanation = self.final_model.explain(intermediate, top_k=top_k)
        block_explanations["final"] = final_explanation
        risks = {**intermediate, "risco_evasao": final_explanation.risk}
        input_names = {variable.name for model in self.models.values() for variable in model.variables}
        input_values = {name: float(row[name]) for name in input_names if name in row.index}
        return StudentExplanation(input_values=input_values, risks=risks, blocks=block_explanations)

    def export_rules(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            records = [
                {
                    "rule": rule.label,
                    "consequent": rule.consequent,
                    "support": rule.support,
                    "estimated_risk": rule.estimated_risk,
                    "weight": rule.weight,
                }
                for rule in model.rules
            ]
            pd.DataFrame.from_records(records).to_csv(destination / f"rules_{name}.csv", index=False)
            with (destination / f"rules_{name}.json").open("w", encoding="utf-8") as handle:
                json.dump([asdict(rule) for rule in model.rules], handle, ensure_ascii=False, indent=2)


def fit_pipeline(train_df: pd.DataFrame, *, random_state: int = 42, folds: int = 5) -> FuzzyPipeline:
    """Ajusta subsistemas em OOF e aprende o sistema final sem vazamento de stacking."""

    if folds < 2:
        raise ValueError("folds precisa ser pelo menos 2")
    y = train_df["y_true"].to_numpy(dtype=np.int64)
    oof = np.full((len(train_df), 3), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    specs = (
        ("academico", academic.variables()),
        ("social", social.variables()),
        ("demografico", demographic.variables()),
    )

    for fit_indices, holdout_indices in splitter.split(train_df, y):
        fold_train = train_df.iloc[fit_indices]
        fold_holdout = train_df.iloc[holdout_indices]
        for column_index, (name, variables) in enumerate(specs):
            model = _fit_block(name, variables, fold_train)
            oof[holdout_indices, column_index] = _predict_block(model, fold_holdout)

    if np.isnan(oof).any():
        raise RuntimeError("Predicoes OOF incompletas")
    final_train = pd.DataFrame(
        {
            "risco_academico": oof[:, 0],
            "risco_social": oof[:, 1],
            "risco_demografico": oof[:, 2],
            "y_true": y,
        },
        index=train_df.index,
    )
    final_model = _fit_block("final", final.variables(), final_train)
    return FuzzyPipeline(
        academic_model=_fit_block("academico", academic.variables(), train_df),
        social_model=_fit_block("social", social.variables(), train_df),
        demographic_model=_fit_block("demografico", demographic.variables(), train_df),
        final_model=final_model,
    )
