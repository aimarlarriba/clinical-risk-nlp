"""
pipeline.py: Funciones centrales del pipeline de clasificación supervisada.
Proporciona utilidades para la carga, partición estratificada reproducible y
análisis descriptivo del corpus clínico de Reddit (C-SSRS, N=500 usuarios).
"""

import sys
from typing import Any, Dict
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.modulo_comun_equipo import (
    RANDOM_SEED,
    RUTA_CSV_DEFECTO,
    cargar_particion_oficial_equipo,
)


def cargar_datos_clasificacion(
    ruta_csv: str = RUTA_CSV_DEFECTO,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Carga y particiona el corpus oficial en 70% Train, 15% Dev y 15% Test estratificados.

    Entradas:
        - ruta_csv: ruta al archivo CSV con columnas 'User', 'Post' y 'Label'.
        - seed: semilla entera para garantizar la reproducibilidad de la partición.

    Salidas:
        - Diccionario con textos, etiquetas, índices y embeddings 384D comunes divididos.

    Precondiciones:
        - Existe el archivo CSV en `ruta_csv`.

    Postcondiciones:
        - Las particiones son disjuntas (350 Train / 75 Dev / 75 Test) y estratificadas.

    Complejidad:
        - Temporal: O(N * L) para lectura/limpieza de N documentos y O(N) para el particionado.
        - Espacial: O(N * D) con D=384.
    """
    return cargar_particion_oficial_equipo(ruta_csv=ruta_csv, seed=seed)


def resumen_estadistico_corpus(ruta_csv: str = RUTA_CSV_DEFECTO) -> pd.DataFrame:
    """
    Calcula y devuelve el conteo absoluto y la prevalencia porcentual por clase C-SSRS.

    Precondiciones:
        - `ruta_csv` existe y contiene la columna 'Label'.

    Postcondiciones:
        - Devuelve DataFrame ordenado por frecuencia con columnas ['Clase C-SSRS', 'Total Usuarios', 'Prevalencia (%)'].

    Complejidad:
        - Temporal: O(N), Espacial: O(K) con K=5 clases.
    """
    df = pd.read_csv(ruta_csv)
    dist = df["Label"].value_counts().reset_index()
    dist.columns = ["Clase C-SSRS", "Total Usuarios"]
    dist["Prevalencia (%)"] = (dist["Total Usuarios"] / len(df)) * 100.0
    return dist


if __name__ == "__main__":
    print("[*] Resumen del corpus Reddit C-SSRS (N=500):")
    df_dist = resumen_estadistico_corpus()
    print(df_dist.to_string(index=False))
