"""SIN 2 - Bloco social/financeiro. Slide 7: Debtor, Tuition fees up to date,
Scholarship holder (binarias, entram como grau de pertinencia 0/1) e capital
educacional da familia (continuo, fuzzificado) -> RISCO_SOCIAL (0-100).
"""

from ..membership import AND, OR, defuzzify, trapmf, trimf


def fuzzify_capital(cap):
    baixo = trapmf(cap, [0, 0, 0.5, 1.5])
    medio = trimf(cap, [0.5, 1.5, 2.5])
    alto = trapmf(cap, [1.5, 2.5, 3, 3])
    return baixo, medio, alto


def compute(df):
    debtor = df["debtor"].to_numpy(dtype=float)
    tuition_ok = df["tuition_ok"].to_numpy(dtype=float)
    scholarship = df["scholarship"].to_numpy(dtype=float)
    cap = df["capital_educacional"].to_numpy(dtype=float)  # escala 0-3

    not_debtor = 1 - debtor
    tuition_late = 1 - tuition_ok
    no_scholarship = 1 - scholarship

    cap_baixo, cap_medio, cap_alto = fuzzify_capital(cap)

    # --- regras (slide 7 + regras de cobertura) ---
    r_alto = OR(
        AND(debtor, tuition_late),                       # devedor E mensalidade atrasada -> alto
        AND(no_scholarship, debtor, cap_baixo),           # sem bolsa + divida + baixo capital -> alto
    )
    r_medio = OR(
        AND(no_scholarship, debtor),                      # sem bolsa E com divida -> medio/alto
        AND(cap_baixo, tuition_ok),                        # baixo capital educacional isolado -> medio
        AND(not_debtor, tuition_late),                     # mensalidade atrasada sem ser devedor -> medio
        cap_medio,
    )
    r_baixo = OR(
        AND(scholarship, tuition_ok),                      # possui bolsa E mensalidade em dia -> baixo
        AND(not_debtor, tuition_ok, cap_alto),
    )

    return defuzzify({"alto": r_alto, "medio": r_medio, "baixo": r_baixo})
