"""SIN 4 - Sistema final. Slide 9: combina RISCO_ACADEMICO, RISCO_SOCIAL e
RISCO_DEMOGRAFICO (0-100 cada) -> RISCO_EVASAO (0-100).

O bloco academico tem peso dominante nas regras, conforme a enfase do slide
("SE risco academico e alto -> evasao alta", independente dos demais).
"""

from ..membership import AND, OR, defuzzify, trapmf, trimf


def fuzzify_risco(x):
    baixo = trapmf(x, [0, 0, 30, 45])
    medio = trimf(x, [30, 50, 70])
    alto = trapmf(x, [55, 70, 100, 100])
    return baixo, medio, alto


def compute(risco_academico, risco_social, risco_demografico):
    aca_baixo, aca_medio, aca_alto = fuzzify_risco(risco_academico)
    soc_baixo, soc_medio, soc_alto = fuzzify_risco(risco_social)
    dem_baixo, dem_medio, dem_alto = fuzzify_risco(risco_demografico)

    # --- regras finais (slide 9 + regras de cobertura) ---
    r_alto = OR(
        aca_alto,                                   # SE academico alto -> alta
        AND(aca_medio, soc_alto),                    # SE academico medio E social alto -> alta
        AND(soc_alto, dem_alto),
        AND(aca_medio, dem_alto),
    )
    r_medio = OR(
        AND(aca_medio, soc_medio),
        AND(aca_baixo, soc_alto),
        AND(aca_medio, soc_baixo),
        AND(aca_medio, dem_medio),
        AND(aca_baixo, dem_alto),
    )
    r_baixo = OR(
        AND(aca_baixo, soc_baixo, dem_baixo),        # SE todos baixos -> baixa
        AND(aca_baixo, soc_baixo),
        AND(aca_baixo, soc_medio, dem_baixo),
    )

    return defuzzify({"alto": r_alto, "medio": r_medio, "baixo": r_baixo})
