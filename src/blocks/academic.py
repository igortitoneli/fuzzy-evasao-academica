"""Definicao linguistica do subsistema academico."""

from __future__ import annotations

from ..membership import FuzzyVariable, trapmf, trimf


def variables() -> tuple[FuzzyVariable, ...]:
    return (
        FuzzyVariable(
            "nota_academica",
            "Nota academica",
            0.0,
            20.0,
            {
                "baixa": trapmf((0.0, 0.0, 8.0, 10.0)),
                "media": trimf((8.0, 10.0, 15.0)),
                "alta": trapmf((10.0, 15.0, 20.0, 20.0)),
            },
        ),
        FuzzyVariable(
            "aprovadas",
            "Unidades aprovadas",
            0.0,
            26.0,
            {
                "poucas": trapmf((0.0, 0.0, 1.0, 2.0)),
                "medias": trimf((1.0, 2.0, 5.0)),
                "muitas": trapmf((2.0, 5.0, 26.0, 26.0)),
            },
        ),
        FuzzyVariable(
            "sem_avaliacao",
            "Unidades sem avaliacao",
            0.0,
            12.0,
            {
                "poucas": trapmf((0.0, 0.0, 0.5, 1.0)),
                "medias": trimf((0.5, 1.0, 3.0)),
                "muitas": trapmf((1.0, 3.0, 12.0, 12.0)),
            },
        ),
    )
