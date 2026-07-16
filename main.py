"""CLI do experimento fuzzy de risco de evasao academica."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from src import data, evaluate
from src.experiment import export_experiment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Sobrescreve o limiar selecionado por F2 na validacao.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Forca novo download do UCI.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", default=str(evaluate.OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.threshold is not None and not 0.0 <= args.threshold <= 100.0:
        raise SystemExit("--threshold deve estar entre 0 e 100")
    print("Carregando e preparando o dataset UCI 697...")
    frame = data.build_features(data.load_raw(use_cache=not args.no_cache))
    print(f"Executando experimento estratificado com {len(frame)} estudantes...")
    result = run_experiment(
        frame,
        threshold_override=args.threshold,
        random_state=args.random_state,
        folds=args.folds,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    export_experiment(result, args.output_dir)
    metrics = result.test_metrics
    print(f"Limiar: {metrics.threshold:.0f} ({result.threshold_source})")
    print(f"F2 teste: {metrics.f2_dropout:.3f}")
    print(f"Recall teste: {metrics.recall_dropout:.3f}")
    print(f"Precisao teste: {metrics.precision_dropout:.3f}")
    print(f"Acuracia teste: {metrics.accuracy:.3f}")
    print(f"Artefatos salvos em {args.output_dir}/")


if __name__ == "__main__":
    main()
