"""Utilitarios genericos de fuzzificacao (Mamdani) e defuzzificacao por media
ponderada (height method) usados pelos 4 blocos do SIN.

Cada bloco fuzzy segue o mesmo padrao:
  1. fuzzifica as entradas em conjuntos linguisticos (baixo/medio/alto) via
     funcoes de pertinencia triangulares/trapezoidais (skfuzzy.membership).
  2. calcula a forca de disparo de cada regra (AND=min, OR=max).
  3. defuzzifica por media ponderada usando centroides singleton para a
     saida (baixo=15, medio=50, alto=85), equivalente ao metodo da altura
     (height defuzzification) - uma simplificacao valida e eficiente do
     Mamdani classico (centroide de agregacao), adequada para escoragem em
     lote sobre milhares de linhas.
"""

import numpy as np
import skfuzzy as fuzz

OUTPUT_CENTROIDS = {"baixo": 15.0, "medio": 50.0, "alto": 85.0}


def trimf(x, abc):
    return fuzz.trimf(np.asarray(x, dtype=float), abc)


def trapmf(x, abcd):
    return fuzz.trapmf(np.asarray(x, dtype=float), abcd)


def AND(*terms):
    return np.minimum.reduce(terms)


def OR(*terms):
    return np.maximum.reduce(terms)


def defuzzify(rule_strengths_by_level):
    """rule_strengths_by_level: dict {'baixo': array, 'medio': array, 'alto': array}
    com a forca agregada (max das regras que concluem naquele nivel) para cada linha.
    Retorna o risco defuzzificado (0-100) por media ponderada dos centroides.
    """
    num = np.zeros_like(next(iter(rule_strengths_by_level.values())), dtype=float)
    den = np.zeros_like(num)
    for level, strength in rule_strengths_by_level.items():
        centroid = OUTPUT_CENTROIDS[level]
        num += strength * centroid
        den += strength
    # quando nenhuma regra dispara (den=0), assume risco medio (50) como neutro
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(den > 1e-9, num / np.where(den > 1e-9, den, 1.0), 50.0)
    return np.clip(risk, 0, 100)
