"""
pipeline.py: Funciones centrales del pipeline de clasificación supervisada.
Proporciona utilidades para la carga, preprocesamiento de textos y extracción de features
para el corpus clínico de Reddit (C-SSRS, N=500 usuarios).
"""

import os
import sys
from typing import Tuple, List, Any, Dict
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.modulo_comun_equipo import (
    cargar_particion_oficial_equipo,
    limpiar_texto_reddit,
    RUTA_CSV_DEFECTO,
    RANDOM_SEED
)


def cargar_datos_clasificacion(
    ruta_csv: str = RUTA_CSV_DEFECTO,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Carga y particiona el corpus oficial en 70% Train, 15% Dev y 15% Test.
    Precondiciones:
        - Existe el archivo CSV en 'ruta_csv'.
    Postcondiciones:
        - Devuelve diccionario con textos, etiquetas y embeddings divididos limpiamente.
    """
    return cargar_particion_oficial_equipo(ruta_csv=ruta_csv, seed=seed)


def resumen_estadistico_corpus(ruta_csv: str = RUTA_CSV_DEFECTO) -> pd.DataFrame:
    """Calcula y devuelve el conteo y distribución de prevalencia de clases C-SSRS."""
    df = pd.read_csv(ruta_csv)
    dist = df["Label"].value_counts().reset_index()
    dist.columns = ["Clase C-SSRS", "Total Usuarios"]
    dist["Prevalencia (%)"] = (dist["Total Usuarios"] / len(df)) * 100.0
    return dist


if __name__ == "__main__":
    print("[*] Resumen del corpus Reddit C-SSRS (N=500):")
    df_dist = resumen_estadistico_corpus()
    print(df_dist.to_string(index=False))
