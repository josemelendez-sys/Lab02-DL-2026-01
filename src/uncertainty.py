"""Estimacion de incertidumbre mediante Monte Carlo Dropout."""

import numpy as np
import torch
import torch.nn as nn


def enable_dropout_during_inference(model: nn.Module) -> None:
    """
    Activa las capas Dropout durante inferencia para Monte Carlo Dropout.

    Por defecto, model.eval() desactiva el Dropout. Este metodo
    recorre el modelo y vuelve a poner en modo train solo las capas
    Dropout, dejando el resto (BatchNorm, etc.) en modo eval.
    """

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    inputs: torch.Tensor,
    n_samples: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Realiza n_samples pasadas hacia adelante con Dropout activo.

    Retorna
    -------
    mean_probs : tensor (N, C) con la probabilidad media de cada clase.
    uncertainty : tensor (N, C) con la desviacion estandar entre muestras,
                  usada como proxy de incertidumbre epistemica.
    """

    enable_dropout_during_inference(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.unsqueeze(0))  # (1, N, C)

    # stack -> (n_samples, N, C)
    stacked = torch.cat(all_probs, dim=0)

    mean_probs = stacked.mean(dim=0)       # (N, C)
    uncertainty = stacked.std(dim=0)       # (N, C) desviacion estandar

    return mean_probs, uncertainty


def summarize_uncertainty(
    uncertainty: torch.Tensor,
) -> dict:
    """
    Resume la incertidumbre de un batch en estadisticas simples.

    Parametros
    ----------
    uncertainty : tensor (N, C) con std por muestra y clase.

    Retorna
    -------
    Diccionario con estadisticas globales y por clase.
    """

    unc_np = uncertainty.cpu().numpy()

    # Incertidumbre media por muestra (promedio sobre clases)
    per_sample = unc_np.mean(axis=1)

    # Incertidumbre media por clase (promedio sobre muestras)
    per_class = unc_np.mean(axis=0)

    return {
        "mean_uncertainty": float(unc_np.mean()),
        "max_uncertainty": float(unc_np.max()),
        "min_uncertainty": float(unc_np.min()),
        "per_sample_mean": per_sample.tolist(),
        "per_class_mean": per_class.tolist(),
        "high_uncertainty_fraction": float(
            (unc_np > 0.2).mean()
        ),  # fraccion de predicciones con std > 0.2
    }
