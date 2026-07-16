"""Fixtures sinteticas que mantem a suite independente da rede."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_students() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 120
    grade = rng.uniform(4.0, 18.0, size)
    approved = rng.uniform(0.0, 10.0, size)
    without_evaluation = rng.uniform(0.0, 5.0, size)
    debtor = rng.integers(0, 2, size)
    tuition_late = rng.integers(0, 2, size)
    scholarship = rng.integers(0, 2, size)
    age = rng.integers(17, 60, size)
    displaced = rng.integers(0, 2, size)
    evening = rng.integers(0, 2, size)
    international = rng.integers(0, 2, size)
    logit = (
        (12.0 - grade) / 3.0
        + (4.0 - approved) / 2.0
        + debtor
        + tuition_late
        + (age > 35) * 0.6
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.random(size) < probability).astype(np.int64)
    return pd.DataFrame(
        {
            "row_id": np.arange(size),
            "y_true": target,
            "Target": np.where(target == 1, "Dropout", "Graduate"),
            "nota_academica": grade,
            "aprovadas": approved,
            "sem_avaliacao": without_evaluation,
            "debtor": debtor,
            "tuition_late": tuition_late,
            "tuition_ok": 1 - tuition_late,
            "scholarship": scholarship,
            "capital_educacional": rng.uniform(0.0, 3.0, size),
            "idade": age,
            "deslocado": displaced,
            "noturno": evening,
            "internacional": international,
        }
    )
