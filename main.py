"""Punto de entrada del proyecto Lab02-DL-2026-01."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INNER_FOLDS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MC_SAMPLES,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TARGET,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHT_DECAY,
    HYPERPARAM_GRID,
)
from data_loader import CognitiveMultiLabelDataset, load_dataframe  # noqa: E402
from evaluation import apply_threshold, compute_all_metrics, hamming_loss  # noqa: E402
from models import ShallowMultiLabelNet  # noqa: E402
from preprocessing import prepare_experiment_data, split_for_validation  # noqa: E402
from uncertainty import mc_dropout_predict, summarize_uncertainty  # noqa: E402


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fija semillas para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resuelve el dispositivo solicitado por linea de comandos."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se solicito CUDA, pero no hay GPU disponible.")
    return device


def build_data_loader(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construye un DataLoader simple."""
    dataset = CognitiveMultiLabelDataset(X, Y)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Entrena una epoca con BCEWithLogitsLoss."""
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
) -> dict:
    """
    Evalua el modelo y retorna todas las metricas disponibles:
    hamming_loss, exact_match_accuracy, f1_macro, f1_micro, f1_weighted.
    """
    model.eval()
    all_probs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    probs_np = np.vstack(all_probs)
    targets_np = np.vstack(all_targets)
    preds_np = apply_threshold(probs_np, threshold=threshold)

    metrics = compute_all_metrics(targets_np, preds_np, threshold=threshold)
    metrics["probabilities"] = probs_np
    metrics["targets"] = targets_np
    return metrics


def run_training_cycle(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    threshold: float,
    seed: int,
    device: torch.device,
) -> dict:
    """Entrena y evalua una configuracion puntual del modelo."""
    set_seed(seed)

    train_loader = build_data_loader(X_train, Y_train, batch_size, shuffle=True, seed=seed)
    eval_loader  = build_data_loader(X_eval,  Y_eval,  batch_size, shuffle=False, seed=seed)

    model = ShallowMultiLabelNet(
        input_dim=X_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_dim=Y_train.shape[1],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    for _ in range(epochs):
        history.append(train_one_epoch(model, train_loader, optimizer, criterion, device))

    metrics = evaluate_model(model, eval_loader, threshold=threshold, device=device)

    return {
        "model": model,
        "train_losses": history,
        "final_train_loss": history[-1],
        **{k: v for k, v in metrics.items() if k not in ("probabilities", "targets")},
        "probabilities": metrics["probabilities"],
        "targets": metrics["targets"],
    }


# ---------------------------------------------------------------------------
# Busqueda de hiperparametros en la validacion interna
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    X_outer_train: np.ndarray,
    Y_outer_train: np.ndarray,
    inner_splits: list,
    hyperparam_grid: list,
    batch_size: int,
    epochs: int,
    threshold: float,
    seed: int,
    device: torch.device,
) -> dict:
    """
    Recorre el grid de hiperparametros usando los folds internos.

    Para cada configuracion del grid calcula el hamming_loss promedio
    en los folds de validacion internos y retorna la mejor configuracion.
    """

    best_config = None
    best_score = float("inf")
    grid_results = []

    for config_idx, config in enumerate(hyperparam_grid):
        fold_scores = []

        for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
            result = run_training_cycle(
                X_train=X_outer_train[inner_train_idx],
                Y_train=Y_outer_train[inner_train_idx],
                X_eval=X_outer_train[inner_val_idx],
                Y_eval=Y_outer_train[inner_val_idx],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                batch_size=batch_size,
                epochs=epochs,
                threshold=threshold,
                seed=seed + config_idx * 1000 + inner_fold_idx,
                device=device,
            )
            fold_scores.append(result["hamming_loss"])

        mean_score = float(np.mean(fold_scores))
        std_score  = float(np.std(fold_scores))

        grid_results.append({
            "config": config,
            "inner_hamming_mean": mean_score,
            "inner_hamming_std": std_score,
        })

        if mean_score < best_score:
            best_score = mean_score
            best_config = config

    return {
        "best_config": best_config,
        "best_inner_hamming": best_score,
        "grid_results": grid_results,
    }


# ---------------------------------------------------------------------------
# Incertidumbre con Monte Carlo Dropout
# ---------------------------------------------------------------------------

def run_mc_dropout_evaluation(
    model: nn.Module,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    threshold: float,
    n_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    """
    Evalua el modelo con Monte Carlo Dropout sobre el conjunto de prueba.

    Retorna predicciones con incertidumbre epistemica estimada.
    """
    set_seed(seed)

    dataset = CognitiveMultiLabelDataset(X_test, Y_test)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_mean_probs = []
    all_uncertainty = []
    all_targets     = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        mean_probs, uncertainty = mc_dropout_predict(model, inputs, n_samples=n_samples)
        all_mean_probs.append(mean_probs.cpu())
        all_uncertainty.append(uncertainty.cpu())
        all_targets.append(targets)

    mean_probs_np  = torch.cat(all_mean_probs,  dim=0).numpy()
    uncertainty_np = torch.cat(all_uncertainty, dim=0).numpy()
    targets_np     = torch.cat(all_targets,     dim=0).numpy()

    preds_np = apply_threshold(mean_probs_np, threshold=threshold)
    metrics  = compute_all_metrics(targets_np, preds_np, threshold=threshold)

    unc_summary = summarize_uncertainty(torch.tensor(uncertainty_np))

    return {
        "mc_metrics": metrics,
        "uncertainty_summary": unc_summary,
        "mean_probs": mean_probs_np,
        "uncertainty": uncertainty_np,
    }


# ---------------------------------------------------------------------------
# Flujo principal del experimento
# ---------------------------------------------------------------------------

def train_one_experiment(
    data_path: str | Path,
    target_name: str = DEFAULT_TARGET,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    threshold: float = DEFAULT_THRESHOLD,
    outer_folds: int = DEFAULT_OUTER_FOLDS,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    seed: int = DEFAULT_RANDOM_SEED,
    device_name: str = "cpu",
    use_hyperparam_search: bool = True,
    mc_samples: int = DEFAULT_MC_SAMPLES,
) -> dict:
    """
    Ejecuta el flujo completo del laboratorio:

    1. Carga y prepara los datos.
    2. Validacion cruzada anidada (outer/inner).
    3. Busqueda de hiperparametros en los folds internos.
    4. Reentrena con la mejor config en cada fold externo.
    5. Evalua con todas las metricas (hamming, exact match, F1).
    6. Estima incertidumbre con Monte Carlo Dropout.
    """

    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor o igual que 1.")
    if epochs < 1:
        raise ValueError("epochs debe ser mayor o igual que 1.")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("outer_folds e inner_folds deben ser al menos 2.")

    # Cargar datos
    dataframe = load_dataframe(data_path)
    X, Y, classes, class_to_idx = prepare_experiment_data(dataframe, target_name)
    device = resolve_device(device_name)

    outer_splits = split_for_validation(Y, n_splits=outer_folds, random_state=seed)
    outer_results = []

    for outer_fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
        X_outer_train, Y_outer_train = X[outer_train_idx], Y[outer_train_idx]
        X_outer_test,  Y_outer_test  = X[outer_test_idx],  Y[outer_test_idx]

        inner_splits = split_for_validation(
            Y_outer_train,
            n_splits=inner_folds,
            random_state=seed + outer_fold_idx,
        )

        # --- Busqueda de hiperparametros (validacion interna) ---
        if use_hyperparam_search:
            search_result = run_hyperparameter_search(
                X_outer_train=X_outer_train,
                Y_outer_train=Y_outer_train,
                inner_splits=inner_splits,
                hyperparam_grid=HYPERPARAM_GRID,
                batch_size=batch_size,
                epochs=epochs,
                threshold=threshold,
                seed=seed + outer_fold_idx,
                device=device,
            )
            best_config      = search_result["best_config"]
            best_inner_score = search_result["best_inner_hamming"]
            grid_results     = search_result["grid_results"]
        else:
            # Sin busqueda: usar configuracion por defecto
            best_config = {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
            }
            # Calcular scores internos con la config por defecto
            fold_scores = []
            for inner_fold_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_splits, start=1):
                r = run_training_cycle(
                    X_train=X_outer_train[inner_train_idx],
                    Y_train=Y_outer_train[inner_train_idx],
                    X_eval=X_outer_train[inner_val_idx],
                    Y_eval=Y_outer_train[inner_val_idx],
                    **best_config,
                    batch_size=batch_size,
                    epochs=epochs,
                    threshold=threshold,
                    seed=seed + outer_fold_idx * 100 + inner_fold_idx,
                    device=device,
                )
                fold_scores.append(r["hamming_loss"])
            best_inner_score = float(np.mean(fold_scores))
            grid_results = []

        # --- Reentrenar con la mejor config sobre todo el fold externo ---
        final_result = run_training_cycle(
            X_train=X_outer_train,
            Y_train=Y_outer_train,
            X_eval=X_outer_test,
            Y_eval=Y_outer_test,
            hidden_dim=best_config["hidden_dim"],
            dropout=best_config["dropout"],
            learning_rate=best_config["learning_rate"],
            weight_decay=best_config["weight_decay"],
            batch_size=batch_size,
            epochs=epochs,
            threshold=threshold,
            seed=seed + outer_fold_idx * 1000,
            device=device,
        )

        # --- Incertidumbre con MC Dropout ---
        mc_result = run_mc_dropout_evaluation(
            model=final_result["model"],
            X_test=X_outer_test,
            Y_test=Y_outer_test,
            threshold=threshold,
            n_samples=mc_samples,
            batch_size=batch_size,
            seed=seed + outer_fold_idx * 9999,
            device=device,
        )

        outer_results.append({
            "outer_fold": outer_fold_idx,
            # Validacion interna
            "best_config": best_config,
            "inner_hamming_mean": best_inner_score,
            "grid_results": grid_results,
            # Evaluacion externa (todas las metricas)
            "outer_hamming_loss": final_result["hamming_loss"],
            "outer_exact_match": final_result["exact_match_accuracy"],
            "outer_f1_macro": final_result["f1_macro"],
            "outer_f1_micro": final_result["f1_micro"],
            "outer_f1_weighted": final_result["f1_weighted"],
            "final_train_loss": final_result["final_train_loss"],
            # Incertidumbre
            "mc_hamming_loss": mc_result["mc_metrics"]["hamming_loss"],
            "mc_exact_match": mc_result["mc_metrics"]["exact_match_accuracy"],
            "mc_f1_macro": mc_result["mc_metrics"]["f1_macro"],
            "uncertainty_mean": mc_result["uncertainty_summary"]["mean_uncertainty"],
            "uncertainty_high_fraction": mc_result["uncertainty_summary"]["high_uncertainty_fraction"],
        })

    # --- Resumen global ---
    def _mean_std(key):
        vals = np.array([r[key] for r in outer_results], dtype=np.float32)
        return float(vals.mean()), float(vals.std())

    hamming_mean, hamming_std     = _mean_std("outer_hamming_loss")
    exact_mean,   exact_std       = _mean_std("outer_exact_match")
    f1macro_mean, f1macro_std     = _mean_std("outer_f1_macro")
    f1micro_mean, f1micro_std     = _mean_std("outer_f1_micro")
    mc_hamming_mean, _            = _mean_std("mc_hamming_loss")
    unc_mean, _                   = _mean_std("uncertainty_mean")

    return {
        "target_name": target_name,
        "classes": classes,
        "X_shape": X.shape,
        "Y_shape": Y.shape,
        "device": str(device),
        "use_hyperparam_search": use_hyperparam_search,
        "outer_folds": outer_results,
        "summary": {
            "mean_hamming_loss":         hamming_mean,
            "std_hamming_loss":          hamming_std,
            "mean_exact_match_accuracy": exact_mean,
            "std_exact_match_accuracy":  exact_std,
            "mean_f1_macro":             f1macro_mean,
            "std_f1_macro":              f1macro_std,
            "mean_f1_micro":             f1micro_mean,
            "std_f1_micro":              f1micro_std,
            "mc_mean_hamming_loss":      mc_hamming_mean,
            "mean_uncertainty":          unc_mean,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lab02-DL-2026-01: red poco profunda, validacion anidada con "
                    "busqueda de hiperparametros, metricas multilabel y MC Dropout."
    )
    parser.add_argument("--data-path", required=True,
                        help="Ruta al archivo de datos (.csv o .sav).")
    parser.add_argument("--target-name", default=DEFAULT_TARGET,
                        help=f"Experimento activo. Por defecto: {DEFAULT_TARGET}.")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--mc-samples", type=int, default=DEFAULT_MC_SAMPLES,
                        help="Pasadas de MC Dropout para estimar incertidumbre.")
    parser.add_argument("--no-hyperparam-search", action="store_true",
                        help="Desactiva la busqueda de hiperparametros (usa config por defecto).")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Experimento: {args.target_name}")
    print(f"  Busqueda de hiperparametros: {not args.no_hyperparam_search}")
    print(f"  MC Dropout samples: {args.mc_samples}")
    print("=" * 60)

    results = train_one_experiment(
        data_path=args.data_path,
        target_name=args.target_name,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        threshold=args.threshold,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        device_name=args.device,
        use_hyperparam_search=not args.no_hyperparam_search,
        mc_samples=args.mc_samples,
    )

    s = results["summary"]
    print("\n--- RESUMEN GLOBAL ---")
    print(f"Forma de X : {results['X_shape']}")
    print(f"Forma de Y : {results['Y_shape']}")
    print(f"Clases     : {results['classes']}")
    print(f"Dispositivo: {results['device']}")
    print()
    print(f"Hamming Loss        : {s['mean_hamming_loss']:.4f} +/- {s['std_hamming_loss']:.4f}")
    print(f"Exact Match Acc     : {s['mean_exact_match_accuracy']:.4f} +/- {s['std_exact_match_accuracy']:.4f}")
    print(f"F1 Macro            : {s['mean_f1_macro']:.4f} +/- {s['std_f1_macro']:.4f}")
    print(f"F1 Micro            : {s['mean_f1_micro']:.4f} +/- {s['std_f1_micro']:.4f}")
    print(f"MC Hamming Loss     : {s['mc_mean_hamming_loss']:.4f}")
    print(f"Incertidumbre media : {s['mean_uncertainty']:.4f}")

    print("\n--- DETALLE POR FOLD EXTERNO ---")
    for fold in results["outer_folds"]:
        cfg = fold["best_config"]
        print(
            f"\nFold {fold['outer_fold']} | "
            f"Mejor config: hidden={cfg['hidden_dim']} dropout={cfg['dropout']} "
            f"lr={cfg['learning_rate']}"
        )
        print(
            f"  Inner Hamming   : {fold['inner_hamming_mean']:.4f}  |  "
            f"Outer Hamming   : {fold['outer_hamming_loss']:.4f}"
        )
        print(
            f"  Exact Match     : {fold['outer_exact_match']:.4f}  |  "
            f"F1 Macro        : {fold['outer_f1_macro']:.4f}  |  "
            f"F1 Micro        : {fold['outer_f1_micro']:.4f}"
        )
        print(
            f"  MC Hamming      : {fold['mc_hamming_loss']:.4f}  |  "
            f"Incertidumbre   : {fold['uncertainty_mean']:.4f}  |  "
            f"Alta incert.    : {fold['uncertainty_high_fraction']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
