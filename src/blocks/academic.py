"""SIN 1 - Bloco academico. Slide 6: nota, unidades aprovadas, unidades sem
avaliacao -> RISCO_ACADEMICO (0-100).
"""

from ..membership import AND, OR, defuzzify, trapmf, trimf

# ranges do slide 6 (escala 0-20 para nota; "+" aberto ate o maximo observado)
def fuzzify_nota(nota):
    baixa = trapmf(nota, [0, 0, 8, 10])
    media = trimf(nota, [8, 11.5, 15])
    alta = trapmf(nota, [13, 15, 20, 20])
    return baixa, media, alta


def fuzzify_aprovadas(aprovadas):
    baixa = trapmf(aprovadas, [0, 0, 1, 2])
    media = trimf(aprovadas, [2, 3.5, 5])
    alta = trapmf(aprovadas, [5, 7, 26, 26])
    return baixa, media, alta


def fuzzify_sem_avaliacao(sem_aval):
    baixa = trapmf(sem_aval, [0, 0, 0.5, 1])
    media = trimf(sem_aval, [1, 2, 3])
    alta = trapmf(sem_aval, [3, 4, 12, 12])
    return baixa, media, alta


def compute(df):
    nota = df["nota_academica"].to_numpy()
    aprovadas = df["aprovadas"].to_numpy()
    sem_aval = df["sem_avaliacao"].to_numpy()

    nota_baixa, nota_media, nota_alta = fuzzify_nota(nota)
    aprov_baixa, aprov_media, aprov_alta = fuzzify_aprovadas(aprovadas)
    sem_aval_baixa, sem_aval_media, sem_aval_alta = fuzzify_sem_avaliacao(sem_aval)

    # --- regras (slide 6 + regras de cobertura no mesmo espirito) ---
    r_alto = OR(
        AND(nota_baixa, aprov_baixa),          # SE nota baixa E aprovadas poucas -> alto
        sem_aval_alta,                          # SE sem avaliacao muitas -> alto
        AND(nota_baixa, aprov_media),
        AND(nota_media, aprov_baixa),
    )
    r_medio = OR(
        nota_media,
        AND(nota_baixa, aprov_alta),
        AND(nota_alta, aprov_baixa),
        sem_aval_media,
    )
    r_baixo = OR(
        AND(nota_alta, aprov_alta),             # SE nota alta E aprovadas muitas -> baixo
        AND(nota_alta, aprov_media),
        AND(nota_media, aprov_alta),
    )

    return defuzzify({"alto": r_alto, "medio": r_medio, "baixo": r_baixo})
