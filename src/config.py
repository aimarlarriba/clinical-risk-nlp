"""
Configuración centralizada del proyecto clinical-risk-nl.
Define rutas absolutas reproducibles, hiperparámetros por defecto
y el mapeo ordinal de la escala clínica C-SSRS (Gaur et al., 2019).
"""

from pathlib import Path

# --- Rutas del Proyecto ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_FILE = DATA_DIR / "500_Reddit_users_posts_labels.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data_processed"
FIGURES_DIR = PROCESSED_DATA_DIR / "graficas_clasificacion"

# --- Hiperparámetros por Defecto ---
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
MENTALBERT_MODEL_NAME = "mental/mental-bert-base-uncased"
DEFAULT_N_SAMPLES = 500
RANDOM_SEED = 42

# --- Ontología Clínica Ordinal C-SSRS (Gaur et al., 2019) ---
CLASS_MAP = {
    "Supportive": 0,
    "Indicator": 1,
    "Ideation": 2,
    "Behavior": 3,
    "Attempt": 4,
}

CLASS_NAMES = {idx: nombre for nombre, idx in CLASS_MAP.items()}


def ensure_directories():
    """Crea los directorios esenciales del proyecto si aún no existen."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
