"""SIN 3 - Bloco demografico. Slide 8: idade, deslocado, turno,
internacional -> RISCO_DEMOGRAFICO (0-100).
"""

from ..membership import AND, OR, defuzzify, trapmf, trimf


def fuzzify_idade(idade):
    jovem = trapmf(idade, [17, 17, 20, 25])
    adulto = trimf(idade, [23, 31.5, 40])
    velha = trapmf(idade, [35, 45, 70, 70])
    return jovem, adulto, velha


def compute(df):
    idade = df["idade"].to_numpy(dtype=float)
    deslocado = df["deslocado"].to_numpy(dtype=float)
    noturno = df["noturno"].to_numpy(dtype=float)
    internacional = df["internacional"].to_numpy(dtype=float)

    nao_deslocado = 1 - deslocado

    idade_jovem, idade_adulto, idade_velha = fuzzify_idade(idade)

    # --- regras (slide 8 + regras de cobertura) ---
    # Displaced, turno e internacional entram como fatores contextuais que
    # reforcam ou atenuam o risco associado a faixa etaria, nao como
    # determinantes isolados.
    r_alto = OR(
        AND(idade_velha, deslocado),                 # idade alta E deslocado -> alto
        AND(idade_adulto, deslocado, noturno),
    )
    r_medio = OR(
        AND(noturno, idade_velha),                    # noturno E idade alta -> medio
        idade_adulto,
        AND(idade_velha, nao_deslocado),
        AND(idade_jovem, deslocado),
        AND(internacional, idade_velha),
    )
    r_baixo = OR(
        AND(idade_jovem, nao_deslocado),              # idade baixa E nao deslocado -> baixo
        AND(idade_jovem, nao_deslocado, 1 - noturno),
    )

    return defuzzify({"alto": r_alto, "medio": r_medio, "baixo": r_baixo})
