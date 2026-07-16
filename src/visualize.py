"""Visualizacoes reproduziveis para o relatorio e a apresentacao."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_curve, roc_curve

from .evaluate import ClassificationMetrics
from .membership import FuzzyVariable, MamdaniModel, OUTPUT_TERMS, OUTPUT_UNIVERSE
from .pipeline import FuzzyPipeline, StudentExplanation

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HeatmapResult:
    name: str
    x_variable: str
    y_variable: str
    fixed_values: Mapping[str, float]
    x_values: FloatArray
    y_values: FloatArray
    risks: FloatArray


def _save(fig: plt.Figure, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_system_diagram(path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    positions = {
        "Academico": (0.18, 0.78), "Social/financeiro": (0.18, 0.50),
        "Demografico": (0.18, 0.22), "Sistema final": (0.58, 0.50),
        "Risco de evasao\n0-100": (0.86, 0.50),
    }
    colors = {"Sistema final": "#f4a261", "Risco de evasao\n0-100": "#e76f51"}
    for label, (x, y) in positions.items():
        ax.text(
            x, y, label, ha="center", va="center", fontsize=11,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": colors.get(label, "#d9ed92"), "edgecolor": "#264653"},
        )
    for source in ("Academico", "Social/financeiro", "Demografico"):
        ax.annotate("", xy=positions["Sistema final"], xytext=positions[source], arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=positions["Risco de evasao\n0-100"], xytext=positions["Sistema final"], arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.set_title("Arquitetura hierarquica do sistema de inferencia fuzzy", fontsize=14)
    _save(fig, path)


def plot_memberships(model: MamdaniModel, path: str | Path) -> None:
    variables = list(model.variables)
    fig, axes = plt.subplots(len(variables) + 1, 1, figsize=(8, 2.7 * (len(variables) + 1)))
    axes_array = np.atleast_1d(axes)
    for ax, variable in zip(axes_array[:-1], variables, strict=True):
        universe = np.linspace(variable.minimum, variable.maximum, 300, dtype=np.float64)
        for term, membership in variable.terms.items():
            ax.plot(universe, membership.evaluate(universe), label=term, linewidth=2)
        ax.set_title(variable.label)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel("Pertinencia")
        ax.legend()
    output_ax = axes_array[-1]
    for term, membership in OUTPUT_TERMS.items():
        output_ax.plot(OUTPUT_UNIVERSE, membership.evaluate(OUTPUT_UNIVERSE), label=term, linewidth=2)
    output_ax.set_title("Risco de saida")
    output_ax.set_xlabel("Risco (0-100)")
    output_ax.set_ylabel("Pertinencia")
    output_ax.legend()
    fig.suptitle(f"Funcoes de pertinencia — sistema {model.name}", y=1.01, fontsize=14)
    _save(fig, path)


def plot_split_distribution(parts: Mapping[str, pd.DataFrame], path: str | Path) -> None:
    records = [
        {"Particao": name, "Classe": target, "Quantidade": int(count)}
        for name, frame in parts.items()
        for target, count in frame["Target"].value_counts().items()
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(pd.DataFrame.from_records(records), x="Particao", y="Quantidade", hue="Classe", ax=ax)
    ax.set_title("Distribuicao estratificada das classes")
    _save(fig, path)


def plot_threshold_selection(results: Sequence[ClassificationMetrics], selected: float, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    thresholds = [item.threshold for item in results]
    ax.plot(thresholds, [item.f2_dropout for item in results], label="F2", linewidth=2)
    ax.plot(thresholds, [item.recall_dropout for item in results], label="Recall", alpha=0.8)
    ax.plot(thresholds, [item.precision_dropout for item in results], label="Precisao", alpha=0.8)
    ax.axvline(selected, color="black", linestyle="--", label=f"Selecionado: {selected:.0f}")
    ax.set(xlabel="Limiar", ylabel="Metrica", title="Selecao do limiar na validacao")
    ax.legend()
    _save(fig, path)


def plot_test_evaluation(
    y_true: NDArray[np.int64],
    risk: FloatArray,
    metrics: ClassificationMetrics,
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    matrix = np.asarray(metrics.confusion_matrix, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xticklabels=["Graduate", "Dropout"], yticklabels=["Graduate", "Dropout"], xlabel="Previsto", ylabel="Real", title=f"Teste — limiar {metrics.threshold:.0f}")
    _save(fig, destination / "confusion_matrix_test.png")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, risk)
    precision, recall, _ = precision_recall_curve(y_true, risk)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(false_positive_rate, true_positive_rate, linewidth=2, label=f"AUC={metrics.roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Taxa de falsos positivos", ylabel="Recall", title="Curva ROC")
    axes[0].legend()
    axes[1].plot(recall, precision, linewidth=2, label=f"AP={metrics.pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precisao", title="Curva precisao-recall")
    axes[1].legend()
    _save(fig, destination / "roc_pr_test.png")


def plot_risk_distribution(df: pd.DataFrame, threshold: float, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in (("Graduate", "tab:green"), ("Dropout", "tab:red")):
        ax.hist(df.loc[df["Target"] == label, "risco_evasao"], bins=25, alpha=0.6, label=label, color=color)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Limiar={threshold:.0f}")
    ax.set(xlabel="Risco de evasao", ylabel="Estudantes", title="Distribuicao de risco no teste")
    ax.legend()
    _save(fig, path)


def build_heatmap(
    model: MamdaniModel,
    *,
    name: str,
    x_variable: str,
    y_variable: str,
    fixed_values: Mapping[str, float],
    resolution: int = 20,
) -> HeatmapResult:
    definitions = {variable.name: variable for variable in model.variables}
    x_definition = definitions[x_variable]
    y_definition = definitions[y_variable]
    x_values = np.linspace(x_definition.minimum, x_definition.maximum, resolution, dtype=np.float64)
    y_values = np.linspace(y_definition.minimum, y_definition.maximum, resolution, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    inputs: dict[str, FloatArray] = {}
    for variable in model.variables:
        if variable.name == x_variable:
            inputs[variable.name] = x_grid.ravel()
        elif variable.name == y_variable:
            inputs[variable.name] = y_grid.ravel()
        else:
            inputs[variable.name] = np.full(x_grid.size, fixed_values[variable.name], dtype=np.float64)
    risks = model.infer(inputs).risks.reshape(resolution, resolution)
    return HeatmapResult(name, x_variable, y_variable, dict(fixed_values), x_values, y_values, risks)


def plot_heatmap(result: HeatmapResult, labels: Mapping[str, str], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    image = ax.imshow(
        result.risks,
        origin="lower",
        extent=(
            float(result.x_values.min()), float(result.x_values.max()),
            float(result.y_values.min()), float(result.y_values.max()),
        ),
        aspect="auto",
        vmin=0,
        vmax=100,
        cmap="RdYlGn_r",
    )
    valid_levels = [level for level in (30.0, 50.0, 70.0) if result.risks.min() < level < result.risks.max()]
    if valid_levels:
        contours = ax.contour(result.x_values, result.y_values, result.risks, levels=valid_levels, colors="black", linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=8)
    fixed_text = ", ".join(f"{labels.get(key, key)}={value:.2f}" for key, value in result.fixed_values.items())
    ax.set(
        xlabel=labels.get(result.x_variable, result.x_variable),
        ylabel=labels.get(result.y_variable, result.y_variable),
        title=f"{result.name}\nFixos: {fixed_text}",
    )
    fig.colorbar(image, ax=ax, label="Risco (0-100)")
    _save(fig, path)


def generate_heatmaps(pipeline: FuzzyPipeline, train: pd.DataFrame, output_dir: str | Path) -> Mapping[str, HeatmapResult]:
    destination = Path(output_dir)
    train_risks = pipeline.predict(train)
    specifications = {
        "academico": (
            pipeline.academic_model, "nota_academica", "aprovadas",
            {"sem_avaliacao": float(train["sem_avaliacao"].median())},
        ),
        "social": (
            pipeline.social_model, "debtor", "tuition_late",
            {"scholarship": float(train["scholarship"].mode().iloc[0]), "capital_educacional": float(train["capital_educacional"].median())},
        ),
        "demografico": (
            pipeline.demographic_model, "idade", "deslocado",
            {"noturno": float(train["noturno"].mode().iloc[0]), "internacional": float(train["internacional"].mode().iloc[0])},
        ),
        "final": (
            pipeline.final_model, "risco_academico", "risco_social",
            {"risco_demografico": float(np.median(train_risks.risco_demografico))},
        ),
    }
    labels = {variable.name: variable.label for model in pipeline.models.values() for variable in model.variables}
    results: dict[str, HeatmapResult] = {}
    for name, (model, x_variable, y_variable, fixed_values) in specifications.items():
        result = build_heatmap(model, name=f"Superficie do sistema {name}", x_variable=x_variable, y_variable=y_variable, fixed_values=fixed_values)
        results[name] = result
        plot_heatmap(result, labels, destination / f"heatmap_{name}.png")
    return results


def plot_case_explanation(explanation: StudentExplanation, label: str, path: str | Path) -> None:
    names = ["academico", "social", "demografico", "final"]
    values = [explanation.blocks[name].risk for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, values, color=["#457b9d", "#e9c46a", "#8a5a9e", "#e76f51"])
    axes[0].set(ylim=(0, 100), ylabel="Risco", title=f"Riscos — {label.replace('_', ' ')}")
    rule_labels: list[str] = []
    strengths: list[float] = []
    for block_name in names:
        if explanation.blocks[block_name].activations:
            activation = explanation.blocks[block_name].activations[0]
            rule_labels.append(f"{block_name}: {activation.consequent}")
            strengths.append(activation.firing_strength)
    axes[1].barh(rule_labels, strengths, color="#2a9d8f")
    axes[1].set(xlim=(0, 1), xlabel="Forca de disparo", title="Regra mais forte por sistema")
    _save(fig, path)
