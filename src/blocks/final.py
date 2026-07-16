"""Definicao linguistica do sistema fuzzy final."""

from __future__ import annotations

from ..membership import FuzzyVariable, trapmf, trimf


def _risk_variable(name: str, label: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        label,
        0.0,
        100.0,
        {
            "baixo": trapmf((0.0, 0.0, 30.0, 45.0)),
            "medio": trimf((30.0, 45.0, 70.0)),
            "alto": trapmf((45.0, 70.0, 100.0, 100.0)),
        },
    )


def variables() -> tuple[FuzzyVariable, ...]:
    return (
        _risk_variable("risco_academico", "Risco academico"),
        _risk_variable("risco_social", "Risco social"),
        _risk_variable("risco_demografico", "Risco demografico"),
    )
