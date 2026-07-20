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


def _save(fig: plt.Figure, path: str | Path, *, tight_layout: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
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

    # --- setas primeiro (atras dos blocos) ---
    for source in ("Academico", "Social/financeiro", "Demografico"):
        ax.annotate(
            "", xy=positions["Sistema final"], xytext=positions[source],
            arrowprops={"arrowstyle": "->", "lw": 1.8, "shrinkA": 12, "shrinkB": 12},
            zorder=1,
        )
    ax.annotate(
        "", xy=positions["Risco de evasao\n0-100"], xytext=positions["Sistema final"],
        arrowprops={"arrowstyle": "->", "lw": 1.8, "shrinkA": 12, "shrinkB": 12},
        zorder=1,
    )

    # --- blocos por cima ---
    for label, (x, y) in positions.items():
        ax.text(
            x, y, label, ha="center", va="center", fontsize=11,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": colors.get(label, "#d9ed92"), "edgecolor": "#264653"},
            zorder=2,
        )
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
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    ) * 100.0
    percentage_labels = np.asarray(
        [[f"{value:.1f}%" for value in row] for row in percentages],
        dtype=object,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    sns.heatmap(
        percentages,
        annot=percentage_labels,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=100.0,
        cbar=False,
        ax=axes[1],
    )
    for ax in axes:
        ax.set(
            xticklabels=["Graduate", "Dropout"],
            yticklabels=["Graduate", "Dropout"],
            xlabel="Previsto",
            ylabel="Real",
        )
    axes[0].set_title(f"Contagens — limiar {metrics.threshold:.0f}")
    axes[1].set_title("Percentual dentro da classe real")
    fig.suptitle("Matriz de confusao no teste")
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
    y_resolution: int | None = None,
) -> HeatmapResult:
    definitions = {variable.name: variable for variable in model.variables}
    x_definition = definitions[x_variable]
    y_definition = definitions[y_variable]
    x_values = np.linspace(x_definition.minimum, x_definition.maximum, resolution, dtype=np.float64)
    resolved_y_resolution = y_resolution if y_resolution is not None else resolution
    y_values = np.linspace(
        y_definition.minimum,
        y_definition.maximum,
        resolved_y_resolution,
        dtype=np.float64,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    inputs: dict[str, FloatArray] = {}
    for variable in model.variables:
        if variable.name == x_variable:
            inputs[variable.name] = x_grid.ravel()
        elif variable.name == y_variable:
            inputs[variable.name] = y_grid.ravel()
        else:
            inputs[variable.name] = np.full(x_grid.size, fixed_values[variable.name], dtype=np.float64)
    risks = model.infer(inputs).risks.reshape(resolved_y_resolution, resolution)
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


def plot_heatmap_facets(
    results: Sequence[HeatmapResult],
    labels: Mapping[str, str],
    path: str | Path,
    *,
    columns: int,
    title: str,
) -> None:
    """Plota varias superficies continuas com a mesma escala de risco."""

    rows = int(np.ceil(len(results) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 5.0 * rows), squeeze=False)
    image = None
    for ax, result in zip(axes.flat, results, strict=False):
        image = ax.imshow(
            result.risks,
            origin="lower",
            extent=(
                float(result.x_values.min()), float(result.x_values.max()),
                float(result.y_values.min()), float(result.y_values.max()),
            ),
            aspect="auto",
            vmin=0.0,
            vmax=100.0,
            cmap="RdYlGn_r",
        )
        valid_levels = [
            level
            for level in (30.0, 50.0, 70.0)
            if result.risks.min() < level < result.risks.max()
        ]
        if valid_levels:
            contours = ax.contour(
                result.x_values,
                result.y_values,
                result.risks,
                levels=valid_levels,
                colors="black",
                linewidths=0.8,
            )
            ax.clabel(contours, inline=True, fontsize=8)
        ax.set(
            xlabel=labels.get(result.x_variable, result.x_variable),
            ylabel=labels.get(result.y_variable, result.y_variable),
            title=result.name,
        )
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.12, top=0.82, wspace=0.20)
    if image is not None:
        colorbar_axis = fig.add_axes((0.93, 0.17, 0.012, 0.60))
        fig.colorbar(image, cax=colorbar_axis, label="Risco (0-100)")
    _save(fig, path, tight_layout=False)


def plot_social_scenarios(results: Sequence[HeatmapResult], path: str | Path) -> None:
    """Varia capital educacional e representa os estados binarios como linhas."""

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", 8)
    line_index = 0
    for result in results:
        scholarship = "sim" if result.fixed_values["scholarship"] else "não"
        tuition_late = "sim" if result.fixed_values["tuition_late"] else "não"
        for debtor_index, debtor_value in enumerate(result.y_values):
            debtor = "sim" if debtor_value else "não"
            ax.plot(
                result.x_values,
                result.risks[debtor_index],
                color=colors[line_index],
                marker=("o", "s", "^", "D", "v", "P", "X", "*")[line_index],
                markevery=3,
                linewidth=2,
                label=f"Bolsa {scholarship} · Atrasada {tuition_late} · Devedor {debtor}",
            )
            line_index += 1
    ax.set(
        xlabel="Capital educacional familiar",
        ylabel="Risco (0-100)",
        ylim=(0, 100),
        title="Sistema social — risco por capital educacional",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Combinações binárias", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.68, bottom=0.12, top=0.90)
    _save(fig, path, tight_layout=False)


def plot_demographic_scenarios(results: Sequence[HeatmapResult], path: str | Path) -> None:
    """Varia idade e representa todos os estados binarios como linhas."""

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("tab10", 8)
    line_index = 0
    for result in results:
        attendance = "noturno" if result.fixed_values["noturno"] else "diurno"
        nationality = "internacional" if result.fixed_values["internacional"] else "nacional"
        for displaced_index, displaced_value in enumerate(result.y_values):
            displaced = "deslocado" if displaced_value else "não deslocado"
            ax.plot(
                result.x_values,
                result.risks[displaced_index],
                color=colors[line_index],
                marker=("o", "s", "^", "D", "v", "P", "X", "*")[line_index],
                markevery=3,
                linewidth=2,
                label=f"{attendance.title()} · {nationality.title()} · {displaced.title()}",
            )
            line_index += 1
    ax.set(
        xlabel="Idade",
        ylabel="Risco (0-100)",
        ylim=(0, 100),
        title="Sistema demográfico — risco por idade",
    )
    ax.grid(alpha=0.2)
    ax.legend(title="Combinações binárias", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.68, bottom=0.12, top=0.90)
    _save(fig, path, tight_layout=False)


def generate_heatmaps(pipeline: FuzzyPipeline, train: pd.DataFrame, output_dir: str | Path) -> Mapping[str, HeatmapResult]:
    destination = Path(output_dir)
    labels = {variable.name: variable.label for model in pipeline.models.values() for variable in model.variables}
    results: dict[str, HeatmapResult] = {}

    academic_results: list[HeatmapResult] = []
    for level, value in (("Baixa", 0.0), ("Media", 1.0), ("Alta", 3.0)):
        result = build_heatmap(
            pipeline.academic_model,
            name=f"Sem avaliacao: {level.lower()} ({value:g})",
            x_variable="nota_academica",
            y_variable="aprovadas",
            fixed_values={"sem_avaliacao": value},
        )
        results[f"academico_{level.lower()}"] = result
        academic_results.append(result)
    plot_heatmap_facets(
        academic_results,
        labels,
        destination / "heatmap_academico.png",
        columns=3,
        title="Sistema acadêmico — três níveis de unidades sem avaliação",
    )

    social_results: list[HeatmapResult] = []
    for scholarship, tuition_late in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        result = build_heatmap(
            pipeline.social_model,
            name="Cenário social",
            x_variable="capital_educacional",
            y_variable="debtor",
            fixed_values={"scholarship": scholarship, "tuition_late": tuition_late},
            resolution=20,
            y_resolution=2,
        )
        key = f"social_{'com' if scholarship else 'sem'}_bolsa_mensalidade_{'atrasada' if tuition_late else 'em_dia'}"
        results[key] = result
        social_results.append(result)
    plot_social_scenarios(social_results, destination / "heatmap_social.png")

    demographic_results: list[HeatmapResult] = []
    for evening, international in ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        result = build_heatmap(
            pipeline.demographic_model,
            name=f"{'Noturno' if evening else 'Diurno'}; {'internacional' if international else 'nacional'}",
            x_variable="idade",
            y_variable="deslocado",
            fixed_values={"noturno": evening, "internacional": international},
            resolution=20,
            y_resolution=2,
        )
        key = f"demografico_{'noturno' if evening else 'diurno'}_{'internacional' if international else 'nacional'}"
        results[key] = result
        demographic_results.append(result)
    plot_demographic_scenarios(demographic_results, destination / "curvas_demografico.png")

    final_results: list[HeatmapResult] = []
    for level, value in (("Baixo", 20.0), ("Medio", 50.0), ("Alto", 80.0)):
        result = build_heatmap(
            pipeline.final_model,
            name=f"Risco demografico: {level.lower()} ({value:g})",
            x_variable="risco_academico",
            y_variable="risco_social",
            fixed_values={"risco_demografico": value},
        )
        results[f"final_{level.lower()}"] = result
        final_results.append(result)
    plot_heatmap_facets(
        final_results,
        labels,
        destination / "heatmap_final.png",
        columns=3,
        title="Sistema final — três níveis de risco demográfico",
    )
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
