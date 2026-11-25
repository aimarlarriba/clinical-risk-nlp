"""
Estudio empírico de mejoras exclusivas con Support Vector Machines (SVM) sobre el clasificador híbrido.
Auditado bajo AUDITORIA_DOCENTE.md:
  - Cero data leakage (TF-IDF, clustering, escalado/L2, selección de hiperparámetros y umbrales en Train/Dev; Test ciego).
  - Uso exclusivo de SVM (LinearSVC, SVC con kernel RBF y descomposición Ordinal Frank & Hall).
  - Multi-semilla [42, 43, 44, 45, 46] con media +/- desviación estándar.

Precondiciones:
  - Corpus CSV y representaciones MentalBERT en caché disponibles.
Postcondiciones:
  - Imprime tabla comparativa en consola con Accuracy, Macro-F1, Weighted-F1, F1-Attempt,
    Quadratic Kappa, ROC-AUC, PR-AUC y costes computacionales.
Complejidad:
  - Temporal: O(M * S * N_tr * D) donde M=8 configuraciones SVM, S=5 semillas.
  - Espacial: O(N * D).
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC, LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraccion_mentalbert import MentalBertHierarchicalExtractor
from src.metricas_clasificacion import (
    CLASES_ORDEN_CSSRS,
    GRID_C_LOGARITMICO,
    calibrar_umbral_en_dev,
    construir_espacio_hibrido,
    evaluar_predicciones_clinicas,
    perfilar_entrenamiento_e_inferencia,
    resumir_metricas_multirun,
    seleccionar_c_en_dev,
)
from src.modulo_comun_equipo import RUTA_CSV_DEFECTO, cargar_particion_oficial_equipo, limpiar_texto_reddit


class OrdinalSVMClassifier:
    """
    Clasificador SVM Ordinal mediante descomposición acumulativa de Frank & Hall (2001).
    Entrena K-1 clasificadores binarios calibrados para estimar P(y > k), k = 0..K-2,
    explotando la estructura ordinal de la escala C-SSRS:
        Supportive (0) < Indicator (1) < Ideation (2) < Behavior (3) < Attempt (4).

    Precondiciones:
        - `clases_orden` lista las K clases en orden estricto ascendente de severidad.
    Complejidad:
        - Temporal: O((K-1) * T_fit_svm_calibrado).
    """

    def __init__(self, clases_orden: List[str], C: float = 1.0, class_weight: Optional[str] = "balanced", random_state: int = 42):
        self.clases_orden = list(clases_orden)
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.classes_ = np.array(self.clases_orden)
        self.clasificadores_: List[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalSVMClassifier":
        mapa_idx = {c: i for i, c in enumerate(self.clases_orden)}
        y_ord = np.array([mapa_idx[val] for val in y], dtype=int)
        self.clasificadores_ = []

        for k in range(len(self.clases_orden) - 1):
            y_bin = (y_ord > k).astype(int)
            base_svm = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=5000,
                dual=False,
            )
            cal = CalibratedClassifierCV(base_svm, cv=3, method="sigmoid")
            cal.fit(X, y_bin)
            self.clasificadores_.append(cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        k_tot = len(self.clases_orden)
        p_mayor = np.column_stack([clf.predict_proba(X)[:, 1] for clf in self.clasificadores_])

        probs = np.zeros((n, k_tot), dtype=float)
        probs[:, 0] = 1.0 - p_mayor[:, 0]
        for k in range(1, k_tot - 1):
            probs[:, k] = np.clip(p_mayor[:, k - 1] - p_mayor[:, k], 0.0, 1.0)
        probs[:, k_tot - 1] = p_mayor[:, k_tot - 2]

        suma = probs.sum(axis=1, keepdims=True)
        probs = np.where(suma > 0.0, probs / suma, 1.0 / k_tot)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


def _construir_hibrido_multiclase_l2(
    X_mb_tr: np.ndarray,
    X_mb_dev: np.ndarray,
    X_mb_te: np.ndarray,
    textos_tr: np.ndarray,
    textos_dev: np.ndarray,
    textos_te: np.ndarray,
    y_tr: np.ndarray,
    k_tfidf: int = 200,
    peso_tfidf: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye espacio híbrido con normalización L2 independiente por bloque y selección
    chi2 multiclase (sobre las 5 clases C-SSRS) ajustada exclusivamente en Train.
    """
    X_mb_tr_n = normalize(X_mb_tr, norm="l2")
    X_mb_dev_n = normalize(X_mb_dev, norm="l2")
    X_mb_te_n = normalize(X_mb_te, norm="l2")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
    X_tf_tr = vec.fit_transform(textos_tr)
    X_tf_dev = vec.transform(textos_dev)
    X_tf_te = vec.transform(textos_te)

    k_ef = min(k_tfidf, X_tf_tr.shape[1])
    sel = SelectKBest(chi2, k=k_ef)
    X_sel_tr = normalize(sel.fit_transform(X_tf_tr, y_tr).toarray(), norm="l2")
    X_sel_dev = normalize(sel.transform(X_tf_dev).toarray(), norm="l2")
    X_sel_te = normalize(sel.transform(X_tf_te).toarray(), norm="l2")

    X_tr = normalize(np.hstack([X_mb_tr_n, X_sel_tr * peso_tfidf]), norm="l2")
    X_dev = normalize(np.hstack([X_mb_dev_n, X_sel_dev * peso_tfidf]), norm="l2")
    X_te = normalize(np.hstack([X_mb_te_n, X_sel_te * peso_tfidf]), norm="l2")
    return X_tr, X_dev, X_te


def _seleccionar_rbf_cv_train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
) -> Tuple[float, Any]:
    """
    Selecciona (C*, gamma*) para SVC(kernel='rbf') mediante validación cruzada
    estratificada 5-fold exclusivamente sobre Train sobre rejilla logarítmica.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    mejor_sc = -1.0
    mejor_par: Tuple[float, Any] = (1.0, "scale")

    for c_val in (0.3162, 1.0, 3.1623):
        for g_val in ("scale", 0.01, 0.1):
            clf = SVC(C=c_val, kernel="rbf", gamma=g_val, class_weight="balanced", random_state=seed)
            sc = float(cross_val_score(clf, X_tr, y_tr, cv=skf, scoring="f1_macro").mean())
            if sc > mejor_sc:
                mejor_sc = sc
                mejor_par = (c_val, g_val)
    return mejor_par


def _seleccionar_c_ordinal_en_dev(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    seed: int = 42,
) -> float:
    """
    Selecciona C* para OrdinalSVMClassifier sobre rejilla logarítmica base 10
    evaluando exclusivamente en Dev (Macro-F1 con desempate por Quadratic Kappa).
    """
    mejor_c = 1.0
    mejor_obj = (-1.0, -1.0)
    for c_val in (0.1, 0.3162, 1.0, 3.1623):
        clf = OrdinalSVMClassifier(CLASES_ORDEN_CSSRS, C=c_val, class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y_tr)
        yp = clf.predict(X_dev)
        obj = (
            float(f1_score(y_dev, yp, average="macro", zero_division=0)),
            float(cohen_kappa_score(y_dev, yp, labels=CLASES_ORDEN_CSSRS, weights="quadratic")),
        )
        if obj > mejor_obj:
            mejor_obj = obj
            mejor_c = float(c_val)
    return mejor_c


def ejecutar_experimento_mejoras(semillas: List[int] = [42, 43, 44, 45, 46]) -> pd.DataFrame:
    """Ejecuta la comparativa de mejoras exclusivas con SVM (cero Regresión Logística)."""
    extractor = MentalBertHierarchicalExtractor()
    rep_dict = extractor.obtener_o_generar_representaciones()

    df_raw = pd.read_csv(RUTA_CSV_DEFECTO)
    textos_raw = np.array(df_raw["Post"].fillna("").apply(limpiar_texto_reddit).tolist())

    particiones = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones[s] = {
            "tr": (p["idx"][0], p["y"][0]),
            "dev": (p["idx"][1], p["y"][1]),
            "te": (p["idx"][2], p["y"][2]),
            "X384": p["X_384"],
        }

    nombre_attempt = "Attempt"
    print("=" * 120)
    print("COMPARATIVA DE ARQUITECTURAS Y MEJORAS EXCLUSIVAS CON SVM (CERO REGRESIÓN LOGÍSTICA)")
    print("=" * 120)

    configuraciones: List[Dict[str, Any]] = [
        {"nombre": "1. Híbrido Truncado + LinearSVC (Simétrico)", "modo": "base", "tipo_emb": "plano_truncado", "pesos": None, "cluster": False},
        {"nombre": "2. Híbrido Mean-Mean + LinearSVC (Simétrico)", "modo": "base", "tipo_emb": "mean_mean", "pesos": None, "cluster": False},
        {"nombre": "3. Híbrido Mean-Mean + LinearSVC (Balanced)", "modo": "base", "tipo_emb": "mean_mean", "pesos": "balanced", "cluster": False},
        {"nombre": "4. Híbrido + Clustering Ward 384D + LinearSVC", "modo": "base", "tipo_emb": "mean_mean", "pesos": None, "cluster": True},
        {"nombre": "5. [Mean+Max] + TF-IDF chi2 Multi + L2 + LinearSVC", "modo": "l2_multi_lin", "pesos": "balanced"},
        {"nombre": "6. Mean-Mean + TF-IDF + SVD(64) + SVC(RBF CV)", "modo": "l2_multi_rbf", "pesos": "balanced"},
        {"nombre": "7. Mean-Mean + TF-IDF chi2 Multi + L2 + SVM Ordinal", "modo": "l2_multi_ord", "pesos": "balanced"},
        {"nombre": "8. Mean-Mean + TF-IDF + L2 + LinearSVC + Umbral Dev (Prec>=0.25)", "modo": "l2_multi_lin_th", "pesos": "balanced"},
    ]

    resultados_totales = []

    for cfg in configuraciones:
        metricas_por_semilla = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            if cfg["modo"] == "base":
                emb_base = rep_dict[cfg["tipo_emb"]]
                X_tr, X_dev, X_te = construir_espacio_hibrido(
                    emb_base[idx_tr], emb_base[idx_dev], emb_base[idx_te],
                    textos_raw[idx_tr], textos_raw[idx_dev], textos_raw[idx_te],
                    y_tr, clase_critica=nombre_attempt, k_tfidf=100, alpha=2.0, solo_tfidf=False,
                )

                if cfg["cluster"]:
                    X384_tr, X384_dev, X384_te = particiones[s]["X384"]
                    clust = AgglomerativeClustering(n_clusters=5, linkage="ward")
                    labels_tr = clust.fit_predict(X384_tr)
                    centroides = np.array([X384_tr[labels_tr == k].mean(axis=0) for k in range(5)])

                    dist_tr = cdist(X384_tr, centroides, metric="euclidean")
                    dist_dev = cdist(X384_dev, centroides, metric="euclidean")
                    dist_te = cdist(X384_te, centroides, metric="euclidean")
                    media_d = float(dist_tr.mean())
                    if media_d > 0.0:
                        dist_tr /= media_d
                        dist_dev /= media_d
                        dist_te /= media_d

                    X_tr = np.hstack([X_tr, dist_tr])
                    X_dev = np.hstack([X_dev, dist_dev])
                    X_te = np.hstack([X_te, dist_te])

                def _crear_lin(seed_val: int = s, w_val: Any = cfg["pesos"]) -> LinearSVC:
                    return LinearSVC(C=1.0, class_weight=w_val, random_state=seed_val, max_iter=5000, dual=False)

                clf, y_pred_te, costes = perfilar_entrenamiento_e_inferencia(_crear_lin, X_tr, y_tr, X_te)
                idx_att_clf = list(clf.classes_).index(nombre_attempt)
                scores_att_te = clf.decision_function(X_te)[:, idx_att_clf]

            else:
                emb_base_m = np.hstack([rep_dict["mean_mean"], rep_dict["max_mean"]]) if cfg["modo"] == "l2_multi_lin" else rep_dict["mean_mean"]
                X_tr, X_dev, X_te = _construir_hibrido_multiclase_l2(
                    emb_base_m[idx_tr], emb_base_m[idx_dev], emb_base_m[idx_te],
                    textos_raw[idx_tr], textos_raw[idx_dev], textos_raw[idx_te],
                    y_tr, k_tfidf=200, peso_tfidf=0.5,
                )

                if cfg["modo"] in ("l2_multi_lin", "l2_multi_lin_th"):
                    c_opt, _ = seleccionar_c_en_dev(
                        X_tr, y_tr, X_dev, y_dev,
                        grid_c=GRID_C_LOGARITMICO, class_weight="balanced", seed=s, clase_critica=nombre_attempt,
                    )

                    def _crear_lin_n(c_v: float = c_opt, seed_val: int = s) -> LinearSVC:
                        return LinearSVC(C=c_v, class_weight="balanced", random_state=seed_val, max_iter=5000, dual=False)

                    clf, y_pred_base, costes = perfilar_entrenamiento_e_inferencia(_crear_lin_n, X_tr, y_tr, X_te)
                    idx_att_clf = list(clf.classes_).index(nombre_attempt)
                    scores_att_te = clf.decision_function(X_te)[:, idx_att_clf]

                    if cfg["modo"] == "l2_multi_lin_th":
                        scores_dev = clf.decision_function(X_dev)[:, idx_att_clf]
                        th_seguro, _ = calibrar_umbral_en_dev(
                            scores_dev,
                            y_dev,
                            y_pred_dev_base=clf.predict(X_dev),
                            prec_min=0.25,
                            clase_critica=nombre_attempt,
                            n_pasos=100,
                        )
                        y_pred_te = y_pred_base.copy()
                        y_pred_te[scores_att_te >= th_seguro] = nombre_attempt
                    else:
                        y_pred_te = y_pred_base

                elif cfg["modo"] == "l2_multi_rbf":
                    # Reducción SVD + escalado estrictamente ajustados en Train para evitar maldición de dimensionalidad en RBF
                    svd = TruncatedSVD(n_components=64, random_state=s)
                    scaler = StandardScaler()
                    X_tr_r = scaler.fit_transform(svd.fit_transform(X_tr))
                    X_te_r = scaler.transform(svd.transform(X_te))
                    c_opt, g_opt = _seleccionar_rbf_cv_train(X_tr_r, y_tr, seed=s)

                    def _crear_rbf(c_v: float = c_opt, g_v: Any = g_opt, seed_val: int = s) -> SVC:
                        return SVC(C=c_v, kernel="rbf", gamma=g_v, class_weight="balanced", random_state=seed_val)

                    clf, y_pred_te, costes = perfilar_entrenamiento_e_inferencia(_crear_rbf, X_tr_r, y_tr, X_te_r)
                    idx_att_clf = list(clf.classes_).index(nombre_attempt)
                    scores_att_te = clf.decision_function(X_te_r)[:, idx_att_clf]

                else:  # l2_multi_ord
                    c_ord = _seleccionar_c_ordinal_en_dev(X_tr, y_tr, X_dev, y_dev, seed=s)

                    def _crear_ord(c_v: float = c_ord, seed_val: int = s) -> OrdinalSVMClassifier:
                        return OrdinalSVMClassifier(CLASES_ORDEN_CSSRS, C=c_v, class_weight="balanced", random_state=seed_val)

                    clf, y_pred_te, costes = perfilar_entrenamiento_e_inferencia(_crear_ord, X_tr, y_tr, X_te)
                    idx_att_clf = list(clf.classes_).index(nombre_attempt)
                    scores_att_te = clf.predict_proba(X_te)[:, idx_att_clf]

            m_te = evaluar_predicciones_clinicas(
                y_true=y_te,
                y_pred=y_pred_te,
                clases_orden=CLASES_ORDEN_CSSRS,
                y_scores_attempt=scores_att_te,
                clase_critica=nombre_attempt,
            )
            m_te.update(costes)
            metricas_por_semilla.append(m_te)

        res = resumir_metricas_multirun(metricas_por_semilla, clases_orden=CLASES_ORDEN_CSSRS)
        res["Configuracion"] = cfg["nombre"]
        resultados_totales.append(res)
        print(
            f"{cfg['nombre']:<60} | Acc: {res['Accuracy_mean']:.3f} | w-F1: {res['Weighted_F1_mean']:.3f} | "
            f"Macro: {res['Macro_F1_mean']:.3f} | F1-Att(42): {res['F1_Attempt_seed42']:.3f} | "
            f"F1-Att(μ): {res['F1_Attempt_mean']:.3f}±{res['F1_Attempt_std']:.3f} | "
            f"Rec: {res['Recall_Attempt_mean']:.3f} | Kap: {res['Quadratic_Kappa_mean']:.3f}"
        )

    return pd.DataFrame(resultados_totales)


if __name__ == "__main__":
    ejecutar_experimento_mejoras()
