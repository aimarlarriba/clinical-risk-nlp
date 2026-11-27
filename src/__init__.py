"""
Paquete principal de text-classification-MD:
Clasificación supervisada multiclase desbalanceada sobre el corpus clínico Reddit C-SSRS (N=500).

Módulos principales del pipeline supervisado:
  - main.py: Orquestador CLI integral (baseline, RQ1, RQ2, mejoras, ablación SVM y tabla maestra).
  - pipeline.py: Carga reproducible y estadísticas descriptivas del corpus C-SSRS.
  - src/modulo_comun_equipo.py: Partición común 70/15/15 (seed=42), embeddings 384D y baseline Ward.
  - src/extraccion_mentalbert.py: Extractor jerárquico en 2 niveles sobre MentalBERT (768D).
  - src/metricas_clasificacion.py: Métricas clínicas, construcción híbrida inductiva, selección en Dev y costes.
  - src/experimento_rq1.py: Estudio factorial de agregación temporal (Mean/Max vs. truncamiento 512).
  - src/experimento_rq2.py: Asimetría de costes, curva de Pareto y modelo híbrido neuro-simbólico con SVM.
  - src/experimento_mejoras.py: Comparativa de variantes exclusivas SVM (L2, RBF, Ordinal Frank & Hall).
  - src/experimento_svm_opt.py: Ablación de regularización C, normalización L2 y umbral theta* en Dev.
"""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    CLASS_MAP,
    CLASS_NAMES,
)

__version__ = "2.0.0"
__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_FILE",
    "PROCESSED_DATA_DIR",
    "FIGURES_DIR",
    "CLASS_MAP",
    "CLASS_NAMES",
]
