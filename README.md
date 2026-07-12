# Sistema de Inferência Nebulosa (SIN) para Risco de Evasão Acadêmica

Implementação do trabalho descrito em `apresentacao_sin_evasao_academica.pptx`:
um sistema fuzzy (Mamdani) que converte o problema de classificação rígida
(Dropout/Graduate) em um **risco contínuo de evasão (0–100)**, explicável por
regras linguísticas.

## Plano de implementação (derivado do PPTX)

| Etapa | Descrição | Onde |
|---|---|---|
| 1 | Buscar dataset UCI *Predict Students' Dropout and Academic Success* (id=697) via `ucimlrepo`, cachear localmente | `src/data.py` |
| 2 | Remover classe `Enrolled`; tratar como binário Dropout(1) vs Graduate(0); derivar variáveis de entrada dos 3 grupos (acadêmico, social/financeiro, demográfico) | `src/data.py` |
| 3 | **SIN 1 — Bloco acadêmico**: nota (admissão + notas dos 2 semestres), unidades aprovadas, unidades sem avaliação → `RISCO_ACADEMICO` | `src/blocks/academic.py` |
| 4 | **SIN 2 — Bloco social/financeiro**: devedor, mensalidade em dia, bolsa, capital educacional dos pais → `RISCO_SOCIAL` | `src/blocks/social.py` |
| 5 | **SIN 3 — Bloco demográfico**: idade, deslocado, turno, internacional → `RISCO_DEMOGRAFICO` | `src/blocks/demographic.py` |
| 6 | **SIN 4 — Sistema final**: combina os 3 riscos intermediários (0–100) → `RISCO_EVASAO` | `src/blocks/final.py` |
| 7 | Pipeline encadeando os 4 blocos | `src/pipeline.py` |
| 8 | Avaliação: risco ≥ limiar ⇒ previsto Dropout; matriz de confusão, acurácia, precisão/recall/F1, varredura de limiar | `src/evaluate.py` |
| 9 | CLI que roda tudo e salva outputs | `main.py` |

## Decisões de projeto (não explícitas no slide, assumidas para viabilizar a implementação)

- **Fuzzificação**: funções triangulares/trapezoidais (`skfuzzy.membership`),
  com os intervalos "baixo/médio/alto" do slide 6 usados como suporte da
  pertinência.
- **Defuzzificação**: média ponderada por centróides singleton
  (baixo=15, médio=50, alto=85) — equivalente ao *height method*, uma
  simplificação do centroide de Mamdani. Escolhida por ser vetorizável com
  NumPy e rodar sobre as 3630 linhas do dataset em menos de 1 segundo por
  bloco (uma simulação Mamdani completa via `skfuzzy.control`, linha a
  linha, seria ~1000x mais lenta).
- **Nota acadêmica**: dataset não tem uma única "nota" — combinei nota de
  admissão (normalizada de 0–200 para 0–20) e as notas do 1º/2º semestre
  numa média simples.
- **Capital educacional**: `Mother's/Father's qualification` são códigos
  categóricos (1–44). Mapeei para uma escala ordinal 0–3 (sem escolaridade →
  ensino superior) e tirei a média dos pais — ver `_QUALIFICATION_LEVEL` em
  `src/data.py`.
- **Regras de cobertura**: o slide dá só 2-3 regras de exemplo por bloco.
  Completei a base de regras (mesmo espírito, combinações restantes dos
  termos linguísticos) para que todo o espaço de entrada tenha alguma regra
  disparando — documentado nos comentários de cada bloco.
- **Bloco final**: acadêmico tem peso dominante nas regras (reflete a
  ênfase do slide 9: "SE risco acadêmico é alto → evasão alta",
  independente dos outros dois).

## Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --threshold 60
```

Outputs em `outputs/`: `metrics.json`, `predictions.csv`,
`confusion_matrix.png`, `risk_distribution.png`, `threshold_sweep.png`.

### Versão notebook (recomendada para visualizar os gráficos)

`evasao_academica.ipynb` roda o mesmo pipeline seção por seção (dataset →
SIN1 acadêmico → SIN2 social → SIN3 demográfico → SIN4 final → avaliação),
plotando também as **curvas de pertinência** de cada variável fuzzy — útil
para inspecionar visualmente os conjuntos linguísticos (baixo/médio/alto)
por trás de cada bloco.

```bash
.venv/bin/pip install jupyter
.venv/bin/jupyter notebook evasao_academica.ipynb
# ou, para re-executar do zero via linha de comando:
.venv/bin/jupyter nbconvert --to notebook --execute --inplace evasao_academica.ipynb
```

## Resultado obtido (limiar=60, 3630 alunos Dropout/Graduate)

| Métrica | Valor |
|---|---|
| Acurácia | 0.828 |
| Precisão (Dropout) | 0.894 |
| Recall (Dropout) | 0.635 |
| F1 (Dropout) | 0.742 |

Matriz de confusão `[[TN, FP], [FN, TP]]`: `[[2102, 107], [519, 902]]`

Limiar 60 já é o ótimo por F1 na varredura 30–80 (ver `threshold_sweep.png`).

Risco médio por classe real — confirma que o sistema captura bem o sinal,
principalmente via o bloco acadêmico:

| Target | risco_academico | risco_social | risco_demografico | **risco_evasao** |
|---|---|---|---|---|
| Dropout | 65.2 | 47.2 | 44.5 | **68.7** |
| Graduate | 37.7 | 36.8 | 40.8 | **37.3** |

A distribuição do risco (`risk_distribution.png`) mostra boa separação entre
as duas classes nos extremos (15–20 e ~85), com uma faixa de sobreposição
em torno de 50 — exatamente os casos "cinzentos" que a saída contínua
consegue expressar melhor que uma classe rígida (ideia central do slide 11).

## Estrutura

```
src/
  data.py              fetch/cache/preprocess do dataset UCI
  membership.py        helpers de fuzzificação e defuzzificação
  blocks/
    academic.py        SIN 1
    social.py          SIN 2
    demographic.py      SIN 3
    final.py            SIN 4
  pipeline.py          encadeia os 4 blocos
  evaluate.py          limiar, matriz de confusão, métricas, plots
main.py                CLI
outputs/               resultados gerados
```

## Próximos passos (do slide 12, ainda em aberto)

- Ajustar faixas/regras a partir da matriz de confusão (ex.: reduzir os 519
  falsos negativos — Dropout previsto como Graduate).
- Discutir casos de sobreposição (risco ~50) onde a saída contínua é mais
  informativa que o Target binário.
