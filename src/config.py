"""Constantes simples para mantener el proyecto ordenado."""

FEATURE_COLUMNS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGET_COLUMNS = [
    "GDS",
    "GDS_R1",
    "GDS_R2",
    "GDS_R3",
    "GDS_R4",
    "GDS_R5",
]

ID_COLUMN = "ID"

DEFAULT_TARGET = "GDS_R2"
DEFAULT_HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.3
DEFAULT_THRESHOLD = 0.5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_RANDOM_SEED = 42
DEFAULT_MC_SAMPLES = 30

# Grid de hiperparametros para la busqueda interna
HYPERPARAM_GRID = [
    {"hidden_dim": 32,  "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 0.0},
    {"hidden_dim": 32,  "dropout": 0.4, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_dim": 64,  "dropout": 0.2, "learning_rate": 5e-4, "weight_decay": 0.0},
    {"hidden_dim": 64,  "dropout": 0.4, "learning_rate": 1e-3, "weight_decay": 1e-4},
    {"hidden_dim": 128, "dropout": 0.3, "learning_rate": 1e-3, "weight_decay": 1e-4},
]
