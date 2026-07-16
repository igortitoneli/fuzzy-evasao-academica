from __future__ import annotations

import numpy as np
import pytest

from src.membership import (
    FuzzyRule,
    FuzzyVariable,
    MamdaniModel,
    binary_terms,
    induce_rules,
    trapmf,
)


def test_memberships_are_bounded_and_keep_shape() -> None:
    values = np.linspace(0.0, 10.0, 21)
    degrees = trapmf((0.0, 0.0, 4.0, 6.0)).evaluate(values)
    assert degrees.shape == values.shape
    assert np.all((degrees >= 0.0) & (degrees <= 1.0))


def test_symmetric_mamdani_output_has_centroid_50() -> None:
    variable = FuzzyVariable("flag", "Flag", 0.0, 1.0, binary_terms(), binary=True)
    rules = (
        FuzzyRule((("flag", "nao"),), "baixo", 10.0, 20.0, 1.0),
        FuzzyRule((("flag", "sim"),), "alto", 10.0, 80.0, 1.0),
    )
    model = MamdaniModel("symmetric", (variable,), rules, 0.5)
    risk = model.infer({"flag": np.asarray([0.5])}).risks[0]
    assert risk == pytest.approx(50.0, abs=0.15)


def test_induction_is_deterministic_and_prunes_low_support() -> None:
    variable = FuzzyVariable("flag", "Flag", 0.0, 1.0, binary_terms(), binary=True)
    inputs = {"flag": np.asarray([0.0, 0.0, 0.0, 1.0])}
    target = np.asarray([0.0, 0.0, 1.0, 1.0])
    first = induce_rules("test", (variable,), inputs, target, minimum_support=2.0)
    second = induce_rules("test", (variable,), inputs, target, minimum_support=2.0)
    assert first.rules == second.rules
    assert len(first.rules) == 1
    assert first.rules[0].antecedents == (("flag", "nao"),)


def test_uncovered_input_falls_back_to_prior() -> None:
    variable = FuzzyVariable("flag", "Flag", 0.0, 1.0, binary_terms(), binary=True)
    model = MamdaniModel("empty", (variable,), (), 0.3)
    result = model.infer({"flag": np.asarray([1.0])})
    assert result.risks[0] == pytest.approx(30.0)


def test_explanation_orders_activations_by_strength() -> None:
    variable = FuzzyVariable("flag", "Flag", 0.0, 1.0, binary_terms(), binary=True)
    rules = (
        FuzzyRule((("flag", "nao"),), "baixo", 10.0, 20.0, 0.5),
        FuzzyRule((("flag", "sim"),), "alto", 10.0, 80.0, 1.0),
    )
    explanation = MamdaniModel("ordered", (variable,), rules, 0.5).explain({"flag": 0.75})
    strengths = [activation.firing_strength for activation in explanation.activations]
    assert strengths == sorted(strengths, reverse=True)
