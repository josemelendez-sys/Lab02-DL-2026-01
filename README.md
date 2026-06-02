# Lab02-DL-2026-01

Proyecto para el Laboratorio 02 de Deep Learning.

La idea del laboratorio es tratar el problema como seis experimentos independientes:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

Cada experimento toma una sola columna objetivo y la transforma a formato one-hot.

## Que esta implementado

- Carga de datos desde `csv` o `sav` (con parser de respaldo si `pyreadstat` no esta disponible)
- Seleccion de columnas de entrada
- Codificacion one-hot del target activo
- `Dataset` de PyTorch
- Red neuronal poco profunda (`ShallowMultiLabelNet`)
- Entrenamiento con `BCEWithLogitsLoss`
- **Validacion cruzada anidada** (outer/inner folds estratificados)
- **Busqueda de hiperparametros** en la validacion interna sobre un grid predefinido
- **Metricas multilabel completas**: Hamming Loss, Exact Match Accuracy, F1 Macro/Micro/Weighted
- **Incertidumbre con Monte Carlo Dropout**: estimacion epistemica por muestra y clase

## Estructura

```text
Lab02-DL-2026-01/
|-- dataset/
|   |-- 15_atributos_R0-R5.csv   <- dataset incluido
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py          <- constantes y grid de hiperparametros
|   |-- data_loader.py     <- carga CSV/SAV + Dataset PyTorch
|   |-- evaluation.py      <- hamming, exact match, F1
|   |-- models.py          <- ShallowMultiLabelNet
|   |-- preprocessing.py   <- one-hot encoding + folds estratificados
|   `-- uncertainty.py     <- Monte Carlo Dropout
|-- main.py
|-- environment.yml
`-- README.md
```

## Instalacion del entorno

```bash
conda create -n ml_clases -c conda-forge python pandas scikit-learn pyreadstat
conda activate ml_clases
pip install torch
```

## Uso

### Experimento basico

```bash
python main.py --data-path dataset/15_atributos_R0-R5.csv --target-name GDS_R2
```

### Prueba rapida (pocas epocas)

```bash
python main.py --data-path dataset/15_atributos_R0-R5.csv --target-name GDS_R2 --epochs 5
```

### Sin busqueda de hiperparametros (config por defecto)

```bash
python main.py --data-path dataset/15_atributos_R0-R5.csv --target-name GDS_R2 --no-hyperparam-search
```

### Con mas muestras MC Dropout

```bash
python main.py --data-path dataset/15_atributos_R0-R5.csv --target-name GDS_R2 --mc-samples 50
```

## Grid de hiperparametros (src/config.py)

| hidden_dim | dropout | learning_rate | weight_decay |
|-----------|---------|---------------|--------------|
| 32        | 0.2     | 0.001         | 0.0          |
| 32        | 0.4     | 0.001         | 0.0001       |
| 64        | 0.2     | 0.0005        | 0.0          |
| 64        | 0.4     | 0.001         | 0.0001       |
| 128       | 0.3     | 0.001         | 0.0001       |

## Argumentos

| Argumento               | Descripcion                                        | Defecto  |
|-------------------------|----------------------------------------------------|----------|
| `--data-path`           | Ruta al archivo `.csv` o `.sav`                    | —        |
| `--target-name`         | Experimento: GDS, GDS_R1 ... GDS_R5                | GDS_R2   |
| `--epochs`              | Epocas por fold                                    | 20       |
| `--outer-folds`         | Folds externos para evaluacion final               | 5        |
| `--inner-folds`         | Folds internos para busqueda de hiperparametros    | 3        |
| `--mc-samples`          | Pasadas de MC Dropout                              | 30       |
| `--no-hyperparam-search`| Desactiva la busqueda y usa la config por defecto  | false    |
| `--threshold`           | Umbral para binarizar probabilidades               | 0.5      |
| `--device`              | `cpu`, `cuda` o `auto`                             | cpu      |
| `--batch-size`          | Tamano de batch                                    | 32       |
| `--seed`                | Semilla para reproducibilidad                      | 42       |
