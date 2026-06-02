"""Funciones para cargar datos y construir el Dataset de PyTorch."""

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_COLUMNS


def _parse_sav_fallback(path: Path) -> pd.DataFrame:
    """
    Parser minimal de archivos SPSS .sav comprimidos.
    Se usa automaticamente si pyreadstat no esta instalado.
    Solo soporta variables numericas (que es el caso de este dataset).
    """
    with open(path, "rb") as f:
        data = f.read()

    # Cabecera SPSS
    offset = 4 + 60  # rec_type(4) + prod_name(60)
    case_size = struct.unpack_from("<i", data, offset + 4)[0]
    compressed = struct.unpack_from("<i", data, offset + 8)[0]
    n_cases = struct.unpack_from("<i", data, offset + 16)[0]
    bias = struct.unpack_from("<d", data, offset + 20)[0]
    offset += 28 + 9 + 8 + 64 + 3  # resto de la cabecera

    # Leer registros de variables
    var_names = []
    while offset < len(data) - 4:
        rec_type = struct.unpack_from("<i", data, offset)[0]
        if rec_type != 2:
            break
        has_label = struct.unpack_from("<i", data, offset + 8)[0]
        n_missing = struct.unpack_from("<i", data, offset + 12)[0]
        raw_name = data[offset + 24 : offset + 32].decode("latin-1").strip()
        offset += 32
        if has_label:
            label_len = struct.unpack_from("<i", data, offset)[0]
            offset += 4 + ((label_len + 3) & ~3)
        if n_missing > 0:
            offset += abs(n_missing) * 8
        if raw_name:
            var_names.append(raw_name)

    # Saltar registros tipo 7 hasta llegar al tipo 999
    while offset < len(data) - 4:
        rec_type = struct.unpack_from("<i", data, offset)[0]
        if rec_type == 999:
            offset += 8
            break
        if rec_type == 7:
            elem_size = struct.unpack_from("<i", data, offset + 8)[0]
            n_elems = struct.unpack_from("<i", data, offset + 12)[0]
            offset += 16 + elem_size * n_elems
        else:
            break

    # Decodificar datos comprimidos
    col_names = [
        "ID", "Día", "Mes", "Año", "Estación", "País", "Ciudad",
        "CalleLugar", "NumeroPiso", "Miguel2", "González2", "Avenida2",
        "Imperial2", "A682", "Caldera2", "Copiapo2",
        "GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5",
    ]

    rows = []
    row: list = []
    i = offset
    while len(rows) < n_cases and i + 8 <= len(data):
        codes = data[i : i + 8]
        i += 8
        for code in codes:
            if code == 252:
                break
            if code == 0 or code == 255:
                row.append(float("nan"))
            elif code == 253:
                val = struct.unpack_from("<d", data, i)[0]
                i += 8
                row.append(val)
            elif code == 254:
                row.append(float("nan"))
            else:
                row.append(float(code) - bias)
            if len(row) == case_size:
                rows.append(row[:])
                row = []
        if len(rows) >= n_cases:
            break

    return pd.DataFrame(rows, columns=col_names[: case_size])


def load_dataframe(data_path: str | Path) -> pd.DataFrame:
    """Carga un DataFrame desde CSV o SAV."""

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".sav":
        try:
            import pyreadstat
            dataframe, _ = pyreadstat.read_sav(path)
            return dataframe
        except ImportError:
            # Fallback: parser propio minimal
            return _parse_sav_fallback(path)

    raise ValueError(
        "Formato no soportado. Usa un archivo .csv o .sav dentro de dataset/."
    )


def validate_feature_columns(
    dataframe: pd.DataFrame, feature_columns: list[str] | None = None
) -> None:
    """Verifica que las columnas de entrada existan en el dataset."""

    columns = feature_columns or FEATURE_COLUMNS
    missing_columns = [column for column in columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(
            "Faltan columnas de entrada en el dataset: "
            + ", ".join(missing_columns)
        )


def build_input_matrix(
    dataframe: pd.DataFrame, feature_columns: list[str] | None = None
) -> np.ndarray:
    """Extrae la matriz X usando las columnas de entrada del laboratorio."""

    columns = feature_columns or FEATURE_COLUMNS
    validate_feature_columns(dataframe, columns)
    return dataframe[columns].astype("float32").to_numpy()


class CognitiveMultiLabelDataset(Dataset):
    """Dataset simple de PyTorch para entradas tabulares y targets multilabel."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.Y[index]
