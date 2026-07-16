"""Definicao linguistica do subsistema social e financeiro."""

from __future__ import annotations

from ..membership import FuzzyVariable, binary_terms, trapmf, trimf


def variables() -> tuple[FuzzyVariable, ...]:
    return (
        FuzzyVariable("debtor", "Devedor", 0.0, 1.0, binary_terms(), binary=True),
        FuzzyVariable(
            "tuition_late",
            "Mensalidade atrasada",
            0.0,
            1.0,
            binary_terms(),
            binary=True,
        ),
        FuzzyVariable(
            "scholarship", "Possui bolsa", 0.0, 1.0, binary_terms(), binary=True
        ),
        FuzzyVariable(
            "capital_educacional",
            "Capital educacional familiar",
            0.0,
            3.0,
            {
                "baixo": trapmf((0.0, 0.0, 0.5, 1.5)),
                "medio": trimf((0.5, 1.5, 2.5)),
                "alto": trapmf((1.5, 2.5, 3.0, 3.0)),
            },
        ),
    )
