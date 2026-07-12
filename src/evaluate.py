"""Slide 10: converte o risco continuo (0-100) em classe via limiar e mede
desempenho contra o Target real do dataset."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def predict_class(risco_evasao, threshold=60):
    """risco >= limiar -> previsto Dropout (1); caso contrario Graduate (0)."""
    return (risco_evasao >= threshold).astype(int)


def metrics_at_threshold(y_true, risco_evasao, threshold=60):
    y_pred = predict_class(risco_evasao, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_dropout": precision_score(y_true, y_pred, zero_division=0),
        "recall_dropout": recall_score(y_true, y_pred, zero_division=0),
        "f1_dropout": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }


def threshold_sweep(y_true, risco_evasao, thresholds=range(30, 81, 5)):
    return [metrics_at_threshold(y_true, risco_evasao, t) for t in thresholds]


def best_threshold(sweep_results, key="f1_dropout"):
    return max(sweep_results, key=lambda r: r[key])


def plot_confusion_matrix(cm, threshold, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Graduate", "Dropout"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusao (limiar={threshold})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_risk_distribution(df, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in [("Graduate", "tab:green"), ("Dropout", "tab:red")]:
        subset = df.loc[df["Target"] == label, "risco_evasao"]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
    ax.axvline(threshold, color="black", linestyle="--", label=f"limiar={threshold}")
    ax.set_xlabel("RISCO_EVASAO (0-100)")
    ax.set_ylabel("numero de alunos")
    ax.set_title("Distribuicao do risco previsto por classe real")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_threshold_sweep(sweep_results, path):
    thresholds = [r["threshold"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, [r["accuracy"] for r in sweep_results], marker="o", label="acuracia")
    ax.plot(thresholds, [r["precision_dropout"] for r in sweep_results], marker="o", label="precisao (Dropout)")
    ax.plot(thresholds, [r["recall_dropout"] for r in sweep_results], marker="o", label="recall (Dropout)")
    ax.plot(thresholds, [r["f1_dropout"] for r in sweep_results], marker="o", label="F1 (Dropout)")
    ax.set_xlabel("limiar de risco")
    ax.set_ylabel("metrica")
    ax.set_title("Sensibilidade das metricas ao limiar de decisao")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
