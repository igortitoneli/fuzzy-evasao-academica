# Sistema Fuzzy para Risco de Evasão Acadêmica

Projeto acadêmico que estima um **risco contínuo de evasão entre 0 e 100**
para apoiar intervenções antes que o estudante abandone o curso. O objetivo
não é prever evasão no momento da admissão, nem substituir uma avaliação
humana: o sistema organiza sinais acadêmicos, sociais e demográficos em uma
saída interpretável.

O modelo usa o dataset UCI
*Predict Students' Dropout and Academic Success* (id 697). A classe
`Enrolled` é removida e a avaliação considera `Dropout` contra `Graduate`.

## Arquitetura

Três sistemas fuzzy aprendem riscos intermediários:

1. **Acadêmico** — nota, unidades aprovadas e unidades sem avaliação;
2. **Social/financeiro** — dívida, mensalidade atrasada, bolsa e capital
   educacional familiar;
3. **Demográfico** — idade, deslocamento, turno noturno e condição de
   estudante internacional.

Um quarto sistema aprende a combinar os três riscos em `RISCO_EVASAO`.

As funções de pertinência de entrada continuam definidas por conhecimento de
domínio. As regras, porém, não são codificadas manualmente: elas são induzidas
automaticamente a partir da partição de treino por uma grade fuzzy ponderada.
Cada regra exportada contém suporte, risco estimado, peso e consequente.

## Inferência Mamdani

O motor compartilhado pelos quatro sistemas executa:

- conjunção dos antecedentes por mínimo;
- implicação pelo recorte da função de saída;
- agregação das regras por máximo;
- defuzzificação pelo centroide da área agregada.

O universo de saída possui 1.001 pontos entre 0 e 100 e três conjuntos:

- baixo: trapezoidal `[0, 0, 30, 45]`;
- médio: triangular `[30, 50, 70]`;
- alto: trapezoidal `[55, 70, 100, 100]`.

Se nenhuma regra aprendida cobrir uma entrada, o sistema retorna a prevalência
de evasão observada exclusivamente no treino. O uso desse fallback aparece na
explicação local.

## Indução automática das regras

Para cada combinação de termos linguísticos, o treinamento:

1. calcula a ativação de cada estudante;
2. soma o suporte fuzzy e elimina suporte inferior a 1;
3. estima a taxa de evasão com suavização empírico-bayesiana de força 5;
4. escolhe o consequente baixo/médio/alto com maior pertinência;
5. usa essa pertinência como peso da regra.

Variáveis binárias possuem termos complementares `não` e `sim`. Nos dados
reais elas continuam sendo 0 ou 1. Valores intermediários são usados somente
nas superfícies teóricas para representar graus de verdade fuzzy.

## Protocolo experimental

O experimento usa uma separação estratificada e determinística:

- 70% treino;
- 15% validação;
- 15% teste;
- semente aleatória 42.

Os três subsistemas geram previsões *out-of-fold* em cinco dobras dentro do
treino. O sistema final é aprendido com esses riscos, evitando vazamento de
empilhamento. Depois, os subsistemas são reajustados com todo o treino.

O limiar de intervenção é escolhido apenas na validação, maximizando F2 entre
0 e 100. F2 dá mais importância ao recall para reduzir estudantes em risco que
passariam despercebidos. Empates escolhem o menor limiar. O teste é usado uma
única vez para a avaliação final.

São reportados acurácia, precisão, recall, F1, F2, especificidade, acurácia
balanceada, ROC-AUC, PR-AUC, matriz de confusão e intervalos bootstrap de 95%.

## Como executar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

No primeiro uso, o programa precisa de internet para baixar o UCI 697 e criar
`data/raw.csv`. Esse arquivo permanece fora do Git; execuções posteriores usam
o cache e funcionam offline.

Opções úteis:

```bash
# Sobrescreve explicitamente o limiar selecionado na validação
.venv/bin/python main.py --threshold 55

# Execução rápida para desenvolvimento
.venv/bin/python main.py --folds 3 --bootstrap-iterations 100

# Força novo download do dataset
.venv/bin/python main.py --no-cache
```

Quando `--threshold` é usado, `metrics.json` registra
`threshold_source: cli_override`. Sem essa opção, registra
`validation_f2`.

## Notebook e apresentação

[`evasao_academica.ipynb`](evasao_academica.ipynb) organiza o trabalho em:

1. problema e objetivo;
2. dados e separação experimental;
3. conceitos fuzzy e arquitetura;
4. funções de pertinência e regras aprendidas;
5. superfícies acadêmicas e finais em três cenários e curvas sociais e
   demográficas cujas linhas representam as combinações das entradas binárias;
6. seleção do limiar na validação;
7. resultado final no teste;
8. explicações individuais e análise de erros;
9. análise descritiva por subgrupos;
10. limitações e conclusão.

Execute o notebook somente depois que `data/raw.csv` estiver disponível:

```bash
MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/jupyter nbconvert --to notebook --execute \
  --inplace evasao_academica.ipynb
```

## Artefatos gerados

O diretório `outputs/` recebe:

- predições identificadas por treino, validação e teste;
- métricas finais e intervalos de confiança;
- varredura de limiar da validação;
- regras aprendidas em CSV e JSON;
- métricas descritivas por subgrupo;
- explicações dos casos representativos;
- diagrama do sistema e funções de pertinência;
- mapas de calor acadêmico, social, demográfico e final;
- matriz de confusão, ROC, precisão-recall e distribuição de risco.

Resultados antigos calculados sobre o dataset inteiro não representam mais o
protocolo do projeto e não devem ser usados na apresentação. Os valores finais
devem ser gerados novamente pelo CLI ou notebook.

## Verificação

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python -m pytest
.venv/bin/mypy src main.py
```

Os testes unitários usam dados sintéticos e não exigem rede. O teste de
integração com o dataset real é ignorado automaticamente quando o cache não
existe.

## Limitações e uso responsável

- O estudo usa um único dataset e não demonstra generalização para outra
  instituição.
- As relações observadas são preditivas, não causais.
- Algumas entradas incluem desempenho durante o curso; portanto, o sistema
  apoia intervenção durante a trajetória acadêmica, não triagem na admissão.
- Métricas por subgrupo com menos de 30 estudantes são marcadas como
  insuficientes para interpretação.
- Diferenças entre grupos não constituem certificação de justiça ou ausência
  de viés.
- Um risco alto deve iniciar acolhimento e investigação humana, nunca punição
  ou decisão automática sobre o estudante.
