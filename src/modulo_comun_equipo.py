"""
Módulo común del equipo para la Práctica 2 (Clasificación Supervisada).
Garantiza que todos los integrantes trabajen exactamente con:
  - La misma partición 70% Train / 15% Dev / 15% Test
  - La misma semilla (RANDOM_SEED = 42)
  - El mismo corte estratificado (stratify = y)
  - El mismo Baseline (Clustering Jerárquico sobre Embeddings 384D)
"""

import os
import sys
import ast
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

RANDOM_SEED = 42
RUTA_CSV_DEFECTO = "data/500_Reddit_users_posts_labels.csv"
RUTA_CACHE_384D = "data/embeddings_384d_comun.npy"
_CACHE_CORPUS_EQUIPO = {}


def limpiar_texto_reddit(val):
    """
    Convierte listas de posts serializadas como string ['post1', 'post2'] en texto plano.

    Entradas:
        - val: valor crudo de la columna 'Post' del CSV.
    Salidas:
        - Cadena de texto unificada con los posts separados por espacio.
    Precondiciones:
        - Ninguna (admite valores nulos o strings arbitrarios).
    Complejidad:
        - Temporal: O(L), Espacial: O(L) donde L es la longitud del historial.
    """
    val_str = str(val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            lista = ast.literal_eval(val_str)
            if isinstance(lista, list):
                return " ".join(str(x) for x in lista)
        except Exception:
            pass
    return val_str


def cargar_particion_oficial_equipo(ruta_csv=RUTA_CSV_DEFECTO, ruta_cache_emb=RUTA_CACHE_384D, seed=RANDOM_SEED):
    """
    Carga el corpus clínico y genera la partición estratificada oficial 70/15/15.

    Entradas:
        - ruta_csv: ruta al archivo CSV oficial de Reddit C-SSRS.
        - ruta_cache_emb: ruta a la matriz caché .npy de embeddings comunes (384D).
        - seed: semilla entera para la partición reproducible (oficial: 42).
    Salidas:
        - Diccionario con tuplas (Train, Dev, Test) para 'usuarios', 'textos', 'X_384', 'y' e 'idx'.
    Precondiciones:
        - Existe el archivo CSV en `ruta_csv` con columnas 'User', 'Post' y 'Label'.
    Postcondiciones:
        - Devuelve particiones disjuntas estratificadas: Train=350 (70%), Dev=75 (15%), Test=75 (15%).
    Complejidad:
        - Temporal: O(N * L) en la primera carga (O(N) en llamadas cacheadas); Espacial: O(N * D) con D=384.
    """
    clave_cache = (str(ruta_csv), str(ruta_cache_emb))
    if clave_cache not in _CACHE_CORPUS_EQUIPO:
        df = pd.read_csv(ruta_csv)
        textos = np.array(df["Post"].fillna("").apply(limpiar_texto_reddit).tolist())
        y = np.array(df["Label"].astype(str).tolist())
        usuarios = np.array(df["User"].astype(str).tolist())

        # Cargar o generar los Embeddings 384D comunes para todo el equipo
        if os.path.exists(ruta_cache_emb):
            X_384d = np.load(ruta_cache_emb)
        else:
            from sentence_transformers import SentenceTransformer
            print("[*] Calculando Embeddings 384D comunes (all-MiniLM-L6-v2)...")
            modelo_st = SentenceTransformer("all-MiniLM-L6-v2")
            X_384d = modelo_st.encode(textos.tolist(), show_progress_bar=True)
            os.makedirs(os.path.dirname(ruta_cache_emb), exist_ok=True)
            np.save(ruta_cache_emb, X_384d)

        _CACHE_CORPUS_EQUIPO[clave_cache] = (usuarios, textos, y, X_384d)

    usuarios, textos, y, X_384d = _CACHE_CORPUS_EQUIPO[clave_cache]

    indices = np.arange(len(y))

    # 1º Corte: 70% Train (350 muestras) y 30% Temporal (150 muestras)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.30, random_state=seed, stratify=y
    )

    # 2º Corte: Dividir el 30% Temporal en 15% Dev (75 muestras) y 15% Test (75 muestras)
    idx_dev, idx_test, y_dev, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    return {
        "usuarios": (usuarios[idx_train], usuarios[idx_dev], usuarios[idx_test]),
        "textos": (textos[idx_train], textos[idx_dev], textos[idx_test]),
        "X_384": (X_384d[idx_train], X_384d[idx_dev], X_384d[idx_test]),
        "y": (y_train, y_dev, y_test),
        "idx": (idx_train, idx_dev, idx_test),
        "clases": np.unique(y)
    }


class BaselineClusteringJerarquico384D:
    """
    Baseline común del equipo:
    Agrupa los embeddings 384D de Train mediante Clustering Jerárquico,
    asigna a cada clúster la clase mayoritaria en Train, y predice nuevas muestras
    (Dev/Test) asignándolas al centroide del clúster más cercano.
    """
    def __init__(self, n_clusters=5, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.centroides = None
        self.mapa_clase = {}
        self.probabilidades_cluster = None
        self.clases_ = None

    def fit(self, X_train, y_train):
        """
        Ajusta los clústeres aglomerativos sobre Train y calcula centroides y mapa mayoritario.

        Precondiciones:
            - len(X_train) == len(y_train) >= self.n_clusters.
        Postcondiciones:
            - Centroides y distribución de clases por clúster inicializados sin acceso a Dev/Test.
        Complejidad:
            - Temporal: O(N_tr^2 * D) para clustering jerárquico Ward; Espacial: O(N_tr^2).
        """
        self.clases_ = np.unique(y_train)
        hc = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        clusters_tr = hc.fit_predict(X_train)

        self.centroides = np.zeros((self.n_clusters, X_train.shape[1]))
        self.probabilidades_cluster = np.zeros((self.n_clusters, len(self.clases_)))

        for c in range(self.n_clusters):
            mask = (clusters_tr == c)
            if not np.any(mask):
                continue
            self.centroides[c] = X_train[mask].mean(axis=0)
            y_c = y_train[mask]
            for idx_cls, cls_name in enumerate(self.clases_):
                self.probabilidades_cluster[c, idx_cls] = np.sum(y_c == cls_name) / len(y_c)
            self.mapa_clase[c] = self.clases_[np.argmax(self.probabilidades_cluster[c])]
        return self

    def obtener_cluster_id(self, X):
        """
        Asigna cada muestra al índice del centroide más cercano bajo distancia euclídea.

        Precondiciones: El modelo ha sido ajustado mediante `fit`.
        Complejidad: Temporal O(N * K * D) donde K = self.n_clusters y D = 384.
        """
        dist = cdist(X, self.centroides, metric="euclidean")
        return np.argmin(dist, axis=1)

    def predict(self, X):
        """Predice la etiqueta mayoritaria del clúster asignado para cada muestra en X."""
        clusters = self.obtener_cluster_id(X)
        return np.array([self.mapa_clase[c] for c in clusters])

    def predict_proba(self, X):
        """Devuelve las frecuencias empíricas de clase en Train asociadas al clúster asignado."""
        clusters = self.obtener_cluster_id(X)
        return self.probabilidades_cluster[clusters]


if __name__ == "__main__":
    datos = cargar_particion_oficial_equipo()
    y_tr, y_dev, y_te = datos["y"]
    print("=== PARTICIÓN OFICIAL DEL EQUIPO (SEED = 42, STRATIFY = Y) ===")
    print(f"Total Train (70%): {len(y_tr)} | Total Dev (15%): {len(y_dev)} | Total Test (15%): {len(y_te)}")
    df_dist = pd.DataFrame({
        "Train (70%)": pd.Series(y_tr).value_counts(),
        "Dev (15%)": pd.Series(y_dev).value_counts(),
        "Test (15%)": pd.Series(y_te).value_counts()
    })
    print("\nDistribución de clases por partición:")
    print(df_dist)

