"""
Módulo compartido de evaluación clínica, construcción híbrida y auditoría de costes.
Cumple con el protocolo metodológico de AUDITORIA_DOCENTE.md:
  - Separación estricta Train / Dev / Test (cero fuga de información).
  - Selección de hiperparámetros sobre escalas logarítmicas justificadas en Dev.
  - Cálculo uniforme de métricas clínicas (Accuracy, Macro-F1, Weighted-F1, F1 por clase, Kappa).
  - Perfilado empírico de tiempo de entrenamiento, latencia (ms/doc) y pico de memoria RAM.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from src.modulo_comun_equipo import (
    RUTA_CSV_DEFECTO,
    cargar_particion_oficial_equipo,
    limpiar_texto_reddit,
)

# Orden clínico ordinal de la escala C-SSRS (Gaur et al., 2019)
CLASES_ORDEN_CSSRS: List[str] = ["Supportive", "Indicator", "Ideation", "Behavior", "Attempt"]

# Rejilla logarítmica base 10 en pasos de 0.5 décadas para regularización SVM: C in [10^-2, 10^1]
GRID_C_LOGARITMICO: Tuple[float, ...] = (0.01, 0.0316, 0.1, 0.3162, 1.0, 3.1623, 10.0)


def evaluar_predicciones_clinicas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases_orden: Sequence[str] = CLASES_ORDEN_CSSRS,
    y_scores_attempt: Optional[np.ndarray] = None,
    clase_critica: str = "Attempt",
) -> Dict[str, Any]:
    """
    Calcula el conjunto completo de métricas clínicas para clasificación multiclase desbalanceada.

    Entradas:
        - y_true: vector de etiquetas reales de tamaño N.
        - y_pred: vector de etiquetas predichas de tamaño N.
        - clases_orden: secuencia ordenada de clases para el cálculo de Kappa cuadrático.
        - y_scores_attempt: puntuaciones continuas (margen o probabilidad) para la clase crítica.
        - clase_critica: nombre de la etiqueta minoritaria de alto riesgo.

    Salidas:
        - Diccionario con Accuracy, Balanced Accuracy, Macro-F1, Weighted-F1,
          métricas específicas de la clase crítica, F1 detallado por clase, Kappa cuadrático,
          y opcionalmente ROC-AUC y PR-AUC.

    Precondiciones:
        - len(y_true) == len(y_pred) > 0.

    Complejidad:
        - Temporal: O(N + K) donde N es el tamaño de evaluación y K el número de clases.
        - Espacial: O(N + K).
    """
    clases_lista = list(clases_orden)
    f1_clases = f1_score(y_true, y_pred, labels=clases_lista, average=None, zero_division=0)
    f1_por_clase = {c: float(v) for c, v in zip(clases_lista, f1_clases)}

    resultado: Dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1_Attempt": float(f1_score(y_true, y_pred, labels=[clase_critica], average="macro", zero_division=0)),
        "Recall_Attempt": float(recall_score(y_true, y_pred, labels=[clase_critica], average="macro", zero_division=0)),
        "Precision_Attempt": float(precision_score(y_true, y_pred, labels=[clase_critica], average="macro", zero_division=0)),
        "Quadratic_Kappa": float(cohen_kappa_score(y_true, y_pred, labels=clases_lista, weights="quadratic")),
        "F1_por_clase": f1_por_clase,
    }

    if y_scores_attempt is not None:
        y_bin = (np.asarray(y_true) == clase_critica).astype(int)
        if len(np.unique(y_bin)) > 1:
            resultado["ROC_AUC"] = float(roc_auc_score(y_bin, y_scores_attempt))
            resultado["PR_AUC"] = float(average_precision_score(y_bin, y_scores_attempt))
        else:
            resultado["ROC_AUC"] = 0.0
            resultado["PR_AUC"] = 0.0

    return resultado


def perfilar_entrenamiento_e_inferencia(
    crear_modelo_fn: Callable[[], Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
) -> Tuple[Any, np.ndarray, Dict[str, float]]:
    """
    Entrena un clasificador e infiere sobre Test midiendo tiempo de ajuste,
    latencia por documento (ms/doc) y pico de memoria RAM asignada (MB).

    Precondiciones:
        - X_tr y X_te comparten el mismo número de dimensiones D.
        - crear_modelo_fn devuelve una instancia no ajustada con métodos fit/predict.

    Postcondiciones:
        - Devuelve (modelo_ajustado, y_pred_te, dict_costes).

    Complejidad:
        - Temporal: O(T_fit(N_tr, D) + T_predict(N_te, D)).
        - Espacial: O(M_modelo + N_tr * D).
    """
    tracemalloc.start()
    modelo = crear_modelo_fn()

    t0 = time.perf_counter()
    modelo.fit(X_tr, y_tr)
    fit_time_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred_te = modelo.predict(X_te)
    infer_time_s = time.perf_counter() - t1

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_te = max(len(X_te), 1)
    costes = {
        "fit_time_s": float(fit_time_s),
        "infer_latency_ms_doc": float((infer_time_s / n_te) * 1000.0),
        "peak_ram_mb": float(peak_bytes / (1024.0 * 1024.0)),
    }
    return modelo, y_pred_te, costes


def construir_espacio_hibrido(
    X_mb_tr: Optional[np.ndarray],
    X_mb_dev: Optional[np.ndarray],
    X_mb_te: Optional[np.ndarray],
    textos_tr: np.ndarray,
    textos_dev: np.ndarray,
    textos_te: np.ndarray,
    y_tr: np.ndarray,
    clase_critica: str = "Attempt",
    k_tfidf: int = 100,
    alpha: float = 2.0,
    solo_tfidf: bool = False,
    objetivo_chi2: str = "binario",
    normalizar_l2: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye de forma estrictamente inductiva (fit solo en Train) la representación
    léxica TF-IDF filtrada por chi2 y opcionalmente la concatena con embeddings densos:
        x_i = [ e_i^MentalBERT || alpha * v_i^TFIDF ]

    Justificación analítica de hiperparámetros por defecto:
        - `k_tfidf=100` (~13% de las 768 dimensiones densas) introduce anclaje léxico
          discriminativo manteniendo la razón dimensiones/muestras acotada (868 vs N_tr=350).
        - `alpha=2.0` compensa la menor norma euclídea media del subvector disperso TF-IDF
          frente a la norma del bloque continuo MentalBERT (~8-10) sin anular la escala original.
        - `objetivo_chi2`: 'binario' selecciona términos asociados a `clase_critica`;
          'multiclase' evalúa dependencia chi2 sobre todas las clases C-SSRS.
        - `normalizar_l2`: si es True, normaliza L2 cada bloque por separado y renormaliza
          el vector concatenado resultante.

    Precondiciones:
        - textos_tr, textos_dev, textos_te corresponden a particiones disjuntas.
        - Si solo_tfidf=False, X_mb_tr, X_mb_dev, X_mb_te deben no ser None.

    Postcondiciones:
        - Devuelve (X_tr, X_dev, X_te) sin fuga de datos desde Dev ni Test.

    Complejidad:
        - Temporal: O(N_tr * L_doc + |V| * log|V|) para vectorización y selección chi2.
        - Espacial: O(N * (D_mb + K)).
    """
    vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
    X_tfidf_tr = vectorizador.fit_transform(textos_tr)
    X_tfidf_dev = vectorizador.transform(textos_dev)
    X_tfidf_te = vectorizador.transform(textos_te)

    k_efectivo = min(k_tfidf, X_tfidf_tr.shape[1])
    selector = SelectKBest(chi2, k=k_efectivo)
    y_objetivo = (y_tr == clase_critica) if objetivo_chi2 == "binario" else y_tr
    X_sel_tr = selector.fit_transform(X_tfidf_tr, y_objetivo).toarray()
    X_sel_dev = selector.transform(X_tfidf_dev).toarray()
    X_sel_te = selector.transform(X_tfidf_te).toarray()

    if normalizar_l2:
        X_sel_tr = normalize(X_sel_tr, norm="l2")
        X_sel_dev = normalize(X_sel_dev, norm="l2")
        X_sel_te = normalize(X_sel_te, norm="l2")

    if solo_tfidf:
        return X_sel_tr, X_sel_dev, X_sel_te

    if X_mb_tr is None or X_mb_dev is None or X_mb_te is None:
        raise ValueError("Los embeddings densos no pueden ser None cuando solo_tfidf=False.")

    if normalizar_l2:
        X_mb_tr = normalize(X_mb_tr, norm="l2")
        X_mb_dev = normalize(X_mb_dev, norm="l2")
        X_mb_te = normalize(X_mb_te, norm="l2")

    X_tr = np.hstack([X_mb_tr, X_sel_tr * alpha])
    X_dev = np.hstack([X_mb_dev, X_sel_dev * alpha])
    X_te = np.hstack([X_mb_te, X_sel_te * alpha])

    if normalizar_l2:
        X_tr = normalize(X_tr, norm="l2")
        X_dev = normalize(X_dev, norm="l2")
        X_te = normalize(X_te, norm="l2")

    return X_tr, X_dev, X_te


def cargar_particiones_y_tfidf(
    semillas: Sequence[int],
    k_tfidf: int = 100,
    clase_critica: str = "Attempt",
    objetivo_chi2: str = "binario",
    incluir_x384: bool = False,
    precalcular_tfidf: bool = True,
    ruta_csv: str = RUTA_CSV_DEFECTO,
) -> Tuple[Dict[int, Dict[str, Any]], np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Carga particiones estratificadas Train/Dev/Test para una lista de semillas,
    los textos limpios del corpus y, opcionalmente, precalcula matrices TF-IDF + chi2
    ajustadas estrictamente en Train por semilla.
    """
    df_raw = pd.read_csv(ruta_csv)
    textos_raw = np.array(df_raw["Post"].fillna("").apply(limpiar_texto_reddit).tolist())

    particiones: Dict[int, Dict[str, Any]] = {}
    tfidf_por_semilla: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for s in semillas:
        p = cargar_particion_oficial_equipo(ruta_csv=ruta_csv, seed=s)
        entry: Dict[str, Any] = {
            "tr": (p["idx"][0], p["y"][0]),
            "dev": (p["idx"][1], p["y"][1]),
            "te": (p["idx"][2], p["y"][2]),
        }
        if incluir_x384:
            entry["X384"] = p["X_384"]
        particiones[s] = entry

        if precalcular_tfidf:
            idx_tr_s, y_tr_s = entry["tr"]
            idx_dev_s, _ = entry["dev"]
            idx_te_s, _ = entry["te"]
            tfidf_por_semilla[s] = construir_espacio_hibrido(
                None, None, None,
                textos_raw[idx_tr_s], textos_raw[idx_dev_s], textos_raw[idx_te_s],
                y_tr_s,
                clase_critica=clase_critica,
                k_tfidf=k_tfidf,
                solo_tfidf=True,
                objetivo_chi2=objetivo_chi2,
            )

    return particiones, textos_raw, tfidf_por_semilla


def seleccionar_c_en_dev(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    grid_c: Sequence[float] = GRID_C_LOGARITMICO,
    class_weight: Optional[Any] = "balanced",
    seed: int = 42,
    clase_critica: str = "Attempt",
) -> Tuple[float, float]:
    """
    Selecciona el parámetro de regularización C* sobre una escala logarítmica base 10
    evaluando exclusivamente sobre el conjunto de validación Dev (cero data leakage).

    Función objetivo en Dev:
        J_dev(C) = Macro_F1_dev(C)
    Se optimiza Macro-F1 global sobre las N_dev=75 muestras (con Quadratic Weighted Kappa
    como desempate ordinal) para evitar sobreajuste a la varianza de una sola clase
    minoritaria en Dev (n=7).

    Precondiciones:
        - X_tr y X_dev están previamente ajustados/transformados sin fuga de Test.

    Postcondiciones:
        - Devuelve (C_optimo, mejor_score_dev).

    Complejidad:
        - Temporal: O(|grid_c| * N_tr * D).
    """
    mejor_c = float(grid_c[0])
    mejor_obj = (-1.0, -1.0)

    for c_val in grid_c:
        clf = LinearSVC(
            C=float(c_val),
            class_weight=class_weight,
            random_state=seed,
            max_iter=5000,
            dual=False,
        )
        clf.fit(X_tr, y_tr)
        y_pred_dev = clf.predict(X_dev)
        macro_f1 = float(f1_score(y_dev, y_pred_dev, average="macro", zero_division=0))
        kappa_dev = float(cohen_kappa_score(y_dev, y_pred_dev, labels=CLASES_ORDEN_CSSRS, weights="quadratic"))
        obj = (macro_f1, kappa_dev)

        if obj > mejor_obj:
            mejor_obj = obj
            mejor_c = float(c_val)

    return mejor_c, mejor_obj[0]


def calibrar_umbral_en_dev(
    scores_dev: np.ndarray,
    y_dev: np.ndarray,
    y_pred_dev_base: Optional[np.ndarray] = None,
    prec_min: float = 0.20,
    clase_critica: str = "Attempt",
    n_pasos: int = 100,
) -> Tuple[float, float]:
    """
    Calibra el umbral de decisión post-hoc theta* exclusivamente sobre Dev:
        theta* = argmax_theta F1_dev(Attempt; theta)   s.a.   Precision_dev(Attempt; theta) >= prec_min
    La cota `prec_min = 0.20` (> 2x prevalencia base del 9%) evita saturar el triaje
    con falsos positivos masivos y previene el colapso de la Accuracy global.

    Precondiciones:
        - len(scores_dev) == len(y_dev) > 0.

    Postcondiciones:
        - Devuelve (theta_optimo, f1_dev_optimo) congelado para aplicar en Test.

    Complejidad:
        - Temporal: O(n_pasos * N_dev).
    """
    y_dev_arr = np.asarray(y_dev)
    s_min, s_max = float(np.min(scores_dev)), float(np.max(scores_dev))
    if np.isclose(s_min, s_max):
        return s_min, float(f1_score(y_dev_arr, np.full_like(y_dev_arr, clase_critica), labels=[clase_critica], average="macro", zero_division=0))

    umbrales = np.linspace(s_min, s_max, n_pasos)
    mejor_th: Optional[float] = None
    mejor_f1 = -1.0

    for th in umbrales:
        if y_pred_dev_base is not None:
            cand = y_pred_dev_base.copy()
            cand[scores_dev >= th] = clase_critica
            pr = float(precision_score(y_dev_arr, cand, labels=[clase_critica], average="macro", zero_division=0))
            f1_d = float(f1_score(y_dev_arr, cand, labels=[clase_critica], average="macro", zero_division=0))
        else:
            pred_bin = (scores_dev >= th).astype(int)
            y_bin = (y_dev_arr == clase_critica).astype(int)
            pr = float(precision_score(y_bin, pred_bin, zero_division=0))
            f1_d = float(f1_score(y_bin, pred_bin, zero_division=0))

        if pr >= prec_min and f1_d > mejor_f1:
            mejor_f1 = f1_d
            mejor_th = float(th)

    if mejor_th is None:
        # Si ningún umbral supera prec_min en Dev, no forzar reclasificación agresiva
        return s_max + 1e-6, 0.0

    return mejor_th, mejor_f1


def resumir_metricas_multirun(
    lista_metricas: List[Dict[str, Any]],
    clases_orden: Sequence[str] = CLASES_ORDEN_CSSRS,
) -> Dict[str, Any]:
    """
    Consolida media, desviación estándar empírica y valor de la semilla oficial (índice 0)
    a partir de una lista de diccionarios de métricas por ejecución.
    """
    claves_escalares = [
        "Accuracy",
        "Balanced_Acc",
        "Macro_F1",
        "Weighted_F1",
        "F1_Attempt",
        "Recall_Attempt",
        "Precision_Attempt",
        "Quadratic_Kappa",
        "ROC_AUC",
        "PR_AUC",
        "fit_time_s",
        "infer_latency_ms_doc",
        "peak_ram_mb",
    ]
    resumen: Dict[str, Any] = {}
    for clave in claves_escalares:
        if clave in lista_metricas[0]:
            valores = [float(m[clave]) for m in lista_metricas]
            resumen[f"{clave}_seed42"] = valores[0]
            resumen[f"{clave}_mean"] = float(np.mean(valores))
            resumen[f"{clave}_std"] = float(np.std(valores))

    if "F1_por_clase" in lista_metricas[0]:
        f1_clases_mean = {}
        f1_clases_std = {}
        for c in clases_orden:
            vals_c = [float(m["F1_por_clase"][c]) for m in lista_metricas]
            f1_clases_mean[c] = float(np.mean(vals_c))
            f1_clases_std[c] = float(np.std(vals_c))
        resumen["F1_por_clase_mean"] = f1_clases_mean
        resumen["F1_por_clase_std"] = f1_clases_std

    return resumen
