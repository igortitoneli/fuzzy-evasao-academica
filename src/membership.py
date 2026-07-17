"""Nucleo tipado de inferencia Mamdani e inducao automatica de regras fuzzy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import skfuzzy as fuzz
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

OUTPUT_UNIVERSE: FloatArray = np.linspace(0.0, 100.0, 1001, dtype=np.float64)


@dataclass(frozen=True)
class MembershipFunction:
    """Funcao de pertinencia triangular, trapezoidal ou binaria linear."""

    kind: str
    parameters: tuple[float, ...] = ()

    def evaluate(self, values: FloatArray | float) -> FloatArray:
        array = np.atleast_1d(np.asarray(values, dtype=np.float64))
        if self.kind == "trimf":
            result = fuzz.trimf(array, self.parameters)
        elif self.kind == "trapmf":
            result = fuzz.trapmf(array, self.parameters)
        elif self.kind == "binary_false":
            result = 1.0 - np.clip(array, 0.0, 1.0)
        elif self.kind == "binary_true":
            result = np.clip(array, 0.0, 1.0)
        else:
            raise ValueError(f"Tipo de pertinencia desconhecido: {self.kind}")
        return np.asarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


@dataclass(frozen=True)
class FuzzyVariable:
    """Definicao linguistica de uma variavel de entrada."""

    name: str
    label: str
    minimum: float
    maximum: float
    terms: Mapping[str, MembershipFunction]
    binary: bool = False


@dataclass(frozen=True)
class FuzzyRule:
    """Regra aprendida, com metadados suficientes para auditoria."""

    antecedents: tuple[tuple[str, str], ...]
    consequent: str
    support: float
    estimated_risk: float
    weight: float

    @property
    def label(self) -> str:
        conditions = " E ".join(f"{name}={term}" for name, term in self.antecedents)
        return f"SE {conditions} ENTAO risco={self.consequent}"


@dataclass(frozen=True)
class RuleActivation:
    """Forca de uma regra para uma observacao especifica."""

    label: str
    consequent: str
    firing_strength: float
    weight: float
    support: float
    estimated_risk: float


@dataclass(frozen=True)
class InferenceResult:
    """Saidas crisp e, opcionalmente, ativacoes das regras."""

    risks: FloatArray
    rule_strengths: FloatArray | None = None


@dataclass(frozen=True)
class BlockExplanation:
    """Explicacao local de um unico sistema fuzzy."""

    risk: float
    used_prior_fallback: bool
    activations: tuple[RuleActivation, ...]


def trimf(parameters: tuple[float, float, float]) -> MembershipFunction:
    return MembershipFunction("trimf", parameters)


def trapmf(parameters: tuple[float, float, float, float]) -> MembershipFunction:
    return MembershipFunction("trapmf", parameters)


def binary_terms() -> Mapping[str, MembershipFunction]:
    return {
        "nao": MembershipFunction("binary_false"),
        "sim": MembershipFunction("binary_true"),
    }


OUTPUT_TERMS: Mapping[str, MembershipFunction] = {
    "baixo": trapmf((0.0, 0.0, 30.0, 50.0)),
    "medio": trimf((30.0, 50.0, 70.0)),
    "alto": trapmf((50.0, 70.0, 100.0, 100.0)),
}
OUTPUT_CURVES: Mapping[str, FloatArray] = {
    name: membership.evaluate(OUTPUT_UNIVERSE)
    for name, membership in OUTPUT_TERMS.items()
}


def fuzzify_inputs(
    variables: Sequence[FuzzyVariable],
    inputs: Mapping[str, FloatArray],
) -> Mapping[str, Mapping[str, FloatArray]]:
    """Calcula todos os graus de pertinencia para um lote de entradas."""

    result: dict[str, dict[str, FloatArray]] = {}
    for variable in variables:
        if variable.name not in inputs:
            raise KeyError(f"Entrada ausente: {variable.name}")
        values = np.asarray(inputs[variable.name], dtype=np.float64)
        result[variable.name] = {
            term: membership.evaluate(values)
            for term, membership in variable.terms.items()
        }
    return result


def validate_membership_coverage(
    variables: Sequence[FuzzyVariable],
    *,
    minimum_coverage: float = 0.49,
    resolution: int = 10_001,
) -> None:
    """Garante que nenhuma variavel tenha buracos entre termos linguisticos.

    Uma particao triangular/trapezoidal bem conectada cruza termos vizinhos
    em pertinencia 0,5. A tolerancia 0,49 absorve apenas o erro da grade
    numerica usada na verificacao.
    """

    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage precisa estar entre 0 e 1")
    if resolution < 2:
        raise ValueError("resolution precisa ser pelo menos 2")

    for variable in variables:
        universe = np.linspace(
            variable.minimum,
            variable.maximum,
            resolution,
            dtype=np.float64,
        )
        term_degrees = np.vstack(
            [membership.evaluate(universe) for membership in variable.terms.values()]
        )
        coverage = np.max(term_degrees, axis=0)
        weakest_index = int(np.argmin(coverage))
        weakest_coverage = float(coverage[weakest_index])
        if weakest_coverage < minimum_coverage:
            weakest_value = float(universe[weakest_index])
            raise ValueError(
                f"A variavel '{variable.name}' possui cobertura fuzzy insuficiente "
                f"em {weakest_value:.4f}: maior pertinencia={weakest_coverage:.4f}; "
                f"esperado >= {minimum_coverage:.2f}. Ajuste os termos para se "
                "encontrarem aproximadamente em 0.5."
            )


def _rule_strengths(
    rules: Sequence[FuzzyRule],
    memberships: Mapping[str, Mapping[str, FloatArray]],
    sample_count: int,
) -> FloatArray:
    strengths = np.zeros((sample_count, len(rules)), dtype=np.float64)
    for rule_index, rule in enumerate(rules):
        antecedent_degrees = [memberships[name][term] for name, term in rule.antecedents]
        if antecedent_degrees:
            strengths[:, rule_index] = np.minimum.reduce(antecedent_degrees) * rule.weight
    return strengths


@dataclass(frozen=True)
class MamdaniModel:
    """Sistema Mamdani aprendido com agregacao max-min e centroide real."""

    name: str
    variables: tuple[FuzzyVariable, ...]
    rules: tuple[FuzzyRule, ...]
    dropout_prior: float

    def infer(self, inputs: Mapping[str, FloatArray], *, include_strengths: bool = False) -> InferenceResult:
        first = next(iter(inputs.values()), np.empty(0, dtype=np.float64))
        sample_count = len(first)
        memberships = fuzzify_inputs(self.variables, inputs)
        strengths = _rule_strengths(self.rules, memberships, sample_count)
        aggregated = np.zeros((sample_count, len(OUTPUT_UNIVERSE)), dtype=np.float64)
        for rule_index, rule in enumerate(self.rules):
            aggregated = np.maximum(
                aggregated,
                np.minimum(strengths[:, rule_index, None], OUTPUT_CURVES[rule.consequent][None, :]),
            )
        areas = np.asarray(np.trapezoid(aggregated, OUTPUT_UNIVERSE, axis=1), dtype=np.float64)
        moments = np.asarray(
            np.trapezoid(aggregated * OUTPUT_UNIVERSE[None, :], OUTPUT_UNIVERSE, axis=1),
            dtype=np.float64,
        )
        risks = np.full(sample_count, self.dropout_prior * 100.0, dtype=np.float64)
        covered = areas > 1e-12
        risks[covered] = moments[covered] / areas[covered]

        return InferenceResult(risks=risks, rule_strengths=strengths if include_strengths else None)

    def explain(self, inputs: Mapping[str, float], *, top_k: int = 5) -> BlockExplanation:
        batch = {name: np.asarray([value], dtype=np.float64) for name, value in inputs.items()}
        result = self.infer(batch, include_strengths=True)
        if result.rule_strengths is None:
            raise RuntimeError("As forcas das regras nao foram calculadas")
        strengths = result.rule_strengths[0]
        order = np.argsort(strengths)[::-1]
        activations: list[RuleActivation] = []
        for rule_index in order:
            firing = float(strengths[rule_index])
            if firing <= 0.0 or len(activations) >= top_k:
                break
            rule = self.rules[int(rule_index)]
            activations.append(
                RuleActivation(
                    label=rule.label,
                    consequent=rule.consequent,
                    firing_strength=firing,
                    weight=rule.weight,
                    support=rule.support,
                    estimated_risk=rule.estimated_risk,
                )
            )
        return BlockExplanation(
            risk=float(result.risks[0]),
            used_prior_fallback=not activations,
            activations=tuple(activations),
        )


def induce_rules(
    name: str,
    variables: Sequence[FuzzyVariable],
    inputs: Mapping[str, FloatArray],
    target: FloatArray,
    *,
    minimum_support: float = 1.0,
    prior_strength: float = 5.0,
) -> MamdaniModel:
    """Induz uma base completa de candidatos por grade fuzzy ponderada."""

    validate_membership_coverage(variables)
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("O alvo precisa ser um vetor nao vazio")
    prior = float(np.mean(y))
    memberships = fuzzify_inputs(variables, inputs)
    term_choices = [tuple(variable.terms.keys()) for variable in variables]
    rules: list[FuzzyRule] = []

    for terms in product(*term_choices):
        antecedents = tuple((variable.name, term) for variable, term in zip(variables, terms, strict=True))
        activation = np.minimum.reduce(
            [memberships[variable.name][term] for variable, term in zip(variables, terms, strict=True)]
        )
        support = float(np.sum(activation))
        if support < minimum_support:
            continue
        smoothed_rate = float(
            (np.dot(activation, y) + prior_strength * prior) / (support + prior_strength)
        )
        risk_value = smoothed_rate * 100.0
        consequent_memberships = {
            term_name: float(term.evaluate(risk_value)[0])
            for term_name, term in OUTPUT_TERMS.items()
        }
        consequent = max(consequent_memberships, key=consequent_memberships.__getitem__)
        weight = consequent_memberships[consequent]
        rules.append(
            FuzzyRule(
                antecedents=antecedents,
                consequent=consequent,
                support=support,
                estimated_risk=risk_value,
                weight=weight,
            )
        )

    return MamdaniModel(name=name, variables=tuple(variables), rules=tuple(rules), dropout_prior=prior)
