"""Carga e preparacao do dataset UCI 697 usado no experimento."""

from __future__ import annotations

import os
from typing import Final, cast

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "..", "data", "raw.csv")

# Mapeamento simplificado dos codigos de escolaridade (Mother's/Father's qualification)
# para uma escala ordinal 0-3, usada para compor o "capital educacional" da familia.
# 0 = sem escolaridade / nao sabe ler-escrever
# 1 = ensino basico (1o-3o ciclo)
# 2 = ensino secundario (completo ou incompleto) / tecnico
# 3 = ensino superior (bacharelado, licenciatura, mestrado, doutorado)
_QUALIFICATION_LEVEL: Final[dict[int, int]] = {
    1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3,
    9: 2, 10: 2, 11: 1, 12: 2, 14: 2,
    18: 2, 19: 1, 22: 2, 26: 1, 27: 1,
    29: 1, 30: 1, 34: 0, 35: 0, 36: 0,
    37: 1, 38: 1, 39: 2, 40: 3, 41: 3,
    42: 3, 43: 3, 44: 3,
}


def _qualification_to_level(code: int | float) -> int:
    return _QUALIFICATION_LEVEL.get(int(code), 1)


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """Busca o dataset via ucimlrepo (id=697), cacheando localmente em data/raw.csv."""
    if use_cache and os.path.exists(RAW_PATH):
        return pd.read_csv(RAW_PATH)

    try:
        dataset = fetch_ucirepo(id=697)
    except Exception as error:
        raise RuntimeError(
            "Nao foi possivel baixar o dataset UCI 697. Conecte-se a internet e "
            "execute novamente para criar data/raw.csv; depois disso o projeto roda offline."
        ) from error
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return cast(pd.DataFrame, df)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as variaveis usadas pelos blocos fuzzy a partir das colunas originais.

    Segue a recomendacao do slide 4: remove 'Enrolled' e trata o problema como
    Dropout vs Graduate binario.
    """
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy()
    df["row_id"] = df.index.astype(int)
    df["y_true"] = (df["Target"] == "Dropout").astype(int)

    # --- bloco academico ---
    admission_norm = df["Admission grade"] / 10.0  # escala 0-200 -> 0-20
    sem1_grade = df["Curricular units 1st sem (grade)"]
    sem2_grade = df["Curricular units 2nd sem (grade)"]
    df["nota_academica"] = pd.concat([admission_norm, sem1_grade, sem2_grade], axis=1).mean(axis=1)
    df["aprovadas"] = df[["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]].mean(axis=1)
    df["sem_avaliacao"] = df[["Curricular units 1st sem (without evaluations)",
                               "Curricular units 2nd sem (without evaluations)"]].mean(axis=1)

    # --- bloco social/financeiro ---
    df["debtor"] = df["Debtor"].astype(int)
    df["tuition_ok"] = df["Tuition fees up to date"].astype(int)
    df["tuition_late"] = 1 - df["tuition_ok"]
    df["scholarship"] = df["Scholarship holder"].astype(int)
    mother_level = df["Mother's qualification"].map(_qualification_to_level)
    father_level = df["Father's qualification"].map(_qualification_to_level)
    df["capital_educacional"] = (mother_level + father_level) / 2.0  # escala 0-3

    # --- bloco demografico ---
    df["idade"] = df["Age at enrollment"]
    df["deslocado"] = df["Displaced"].astype(int)
    df["noturno"] = 1 - df["Daytime/evening attendance"].astype(int)  # 1 = curso noturno
    df["internacional"] = df["International"].astype(int)

    return df
