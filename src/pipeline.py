"""Encadeia os 4 SIN: academico, social, demografico -> final (RISCO_EVASAO)."""

from .blocks import academic, social, demographic, final


def run(df):
    """Recebe o dataframe ja com as features derivadas (ver data.build_features)
    e retorna o mesmo dataframe com as 4 colunas de risco adicionadas.
    """
    df = df.copy()
    df["risco_academico"] = academic.compute(df)
    df["risco_social"] = social.compute(df)
    df["risco_demografico"] = demographic.compute(df)
    df["risco_evasao"] = final.compute(
        df["risco_academico"].to_numpy(),
        df["risco_social"].to_numpy(),
        df["risco_demografico"].to_numpy(),
    )
    return df
