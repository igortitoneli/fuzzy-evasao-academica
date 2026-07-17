"""Definicao linguistica do subsistema demografico."""

from __future__ import annotations

from ..membership import FuzzyVariable, binary_terms, trapmf, trimf


def variables() -> tuple[FuzzyVariable, ...]:
    return (
        FuzzyVariable(
            "idade",
            "Idade no ingresso",
            17.0,
            70.0,
            {
                "jovem": trapmf((17.0, 17.0, 20.0, 25.0)),
                "adulta": trimf((20.0, 25.0, 40.0)),
                "mais_velha": trapmf((25.0, 40.0, 70.0, 70.0)),
            },
        ),
        FuzzyVariable("deslocado", "Deslocado", 0.0, 1.0, binary_terms(), binary=True),
        FuzzyVariable(
            "noturno", "Curso noturno", 0.0, 1.0, binary_terms(), binary=True
        ),
        FuzzyVariable(
            "internacional",
            "Estudante internacional",
            0.0,
            1.0,
            binary_terms(),
            binary=True,
        ),
    )
