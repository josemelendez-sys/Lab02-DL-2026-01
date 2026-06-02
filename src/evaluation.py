"""Metricas para evaluar salidas multilabel."""

import numpy as np


def apply_threshold(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convierte probabilidades a etiquetas binarias usando un umbral fijo."""

    return (probabilities >= threshold).astype(np.float32)


def hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Hamming Loss.

    Compara cada componente del vector real con la predicha,
    cuenta cuantas quedan distintas y promedia ese error.
    Valor ideal: 0.0
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular Hamming loss."
        )

    mistakes = np.not_equal(y_true, y_pred)
    return float(mistakes.mean())


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula Exact Match Accuracy (Subset Accuracy).

    Una prediccion se considera correcta solo si todos los bits del
    vector de etiquetas coinciden exactamente con el ground truth.
    Valor ideal: 1.0
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular exact match."
        )

    # Una fila es correcta si todos sus elementos coinciden
    exact_matches = np.all(y_true == y_pred, axis=1)
    return float(exact_matches.mean())


def f1_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Calcula F1 multilabel.

    Parametros
    ----------
    average : 'macro' promedia el F1 de cada clase sin ponderar,
              'micro' agrupa todos los TP/FP/FN antes de calcular,
              'weighted' promedia ponderando por soporte de cada clase.
    Valor ideal: 1.0
    """

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true e y_pred deben tener la misma forma para calcular F1."
        )

    if average == "micro":
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        return (2 * precision * recall / denom) if denom > 0 else 0.0

    n_classes = y_true.shape[1]
    f1_per_class = []
    support_per_class = []

    for c in range(n_classes):
        tp = float(np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 1)))
        fp = float(np.sum((y_pred[:, c] == 1) & (y_true[:, c] == 0)))
        fn = float(np.sum((y_pred[:, c] == 0) & (y_true[:, c] == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = precision + recall
        f1_c = (2 * precision * recall / denom) if denom > 0 else 0.0
        f1_per_class.append(f1_c)
        support_per_class.append(float(np.sum(y_true[:, c])))

    if average == "macro":
        return float(np.mean(f1_per_class))

    if average == "weighted":
        total_support = sum(support_per_class)
        if total_support == 0:
            return 0.0
        weighted = sum(
            f1 * sup for f1, sup in zip(f1_per_class, support_per_class)
        )
        return float(weighted / total_support)

    raise ValueError(f"average debe ser 'macro', 'micro' o 'weighted'. Recibido: {average}")


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    probabilities: np.ndarray | None = None,
) -> dict:
    """
    Calcula todas las metricas disponibles y las retorna en un diccionario.

    Parametros
    ----------
    y_true       : etiquetas reales (binarizadas)
    y_pred       : etiquetas predichas (binarizadas)
    threshold    : umbral usado (solo informativo)
    probabilities: probabilidades crudas (opcional, para referencia futura)
    """

    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "exact_match_accuracy": exact_match_accuracy(y_true, y_pred),
        "f1_macro": f1_multilabel(y_true, y_pred, average="macro"),
        "f1_micro": f1_multilabel(y_true, y_pred, average="micro"),
        "f1_weighted": f1_multilabel(y_true, y_pred, average="weighted"),
        "threshold_used": threshold,
    }
