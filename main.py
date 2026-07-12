"""CLI: busca o dataset (UCI id=697), roda o SIN de 4 blocos e avalia contra o Target.

Uso:
    .venv/bin/python main.py [--threshold 60] [--no-cache]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # roda headless (CLI), sem display disponivel

from src import data, evaluate, pipeline

OUTPUT_DIR = evaluate.OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=60)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Carregando dataset (UCI id=697)...")
    raw = data.load_raw(use_cache=not args.no_cache)
    df = data.build_features(raw)
    print(f"  {len(df)} alunos (Dropout/Graduate) apos remover 'Enrolled'.")

    print("Rodando SIN (4 blocos fuzzy)...")
    df = pipeline.run(df)

    y_true = df["y_true"].to_numpy()
    risco = df["risco_evasao"].to_numpy()

    metrics = evaluate.metrics_at_threshold(y_true, risco, threshold=args.threshold)
    print(f"\n=== Resultado no limiar {args.threshold} ===")
    print(f"Acuracia:            {metrics['accuracy']:.3f}")
    print(f"Precisao (Dropout):  {metrics['precision_dropout']:.3f}")
    print(f"Recall (Dropout):    {metrics['recall_dropout']:.3f}")
    print(f"F1 (Dropout):        {metrics['f1_dropout']:.3f}")
    print(f"Matriz de confusao [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")

    sweep = evaluate.threshold_sweep(y_true, risco)
    best = evaluate.best_threshold(sweep)
    print(f"\nMelhor limiar por F1: {best['threshold']} (F1={best['f1_dropout']:.3f}, "
          f"acuracia={best['accuracy']:.3f})")

    evaluate.save_json(
        {"chosen_threshold": metrics, "sweep": sweep, "best_by_f1": best},
        os.path.join(OUTPUT_DIR, "metrics.json"),
    )
    evaluate.plot_confusion_matrix(
        metrics["confusion_matrix"], args.threshold,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    )
    evaluate.plot_risk_distribution(
        df, args.threshold, os.path.join(OUTPUT_DIR, "risk_distribution.png"),
    )
    evaluate.plot_threshold_sweep(sweep, os.path.join(OUTPUT_DIR, "threshold_sweep.png"))

    cols = ["risco_academico", "risco_social", "risco_demografico", "risco_evasao", "Target"]
    df[cols].to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    print(f"\nOutputs salvos em {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
