"""
Evaluación de optimizaciones exclusivas para Support Vector Machines (LinearSVC).
Auditado bajo AUDITORIA_DOCENTE.md:
  1. Búsqueda de regularización C sobre escala logarítmica base 10 en Dev:
       GRID_C_LOGARITMICO = [10^-2, 10^-1.5, 10^-1, 10^-0.5, 10^0, 10^0.5, 10^1]
     Evaluada estrictamente en Dev para evitar selección arbitraria y data leakage.
  2. Comparación de class_weight: None vs 'balanced'.
  3. Calibración de umbral de margen theta* sobre Dev en decision_function.
  4. Normalización L2 post-concatenación en el espacio híbrido.

Precondiciones:
  - Corpus CSV y representaciones MentalBERT en caché disponibles.
Postcondiciones:
  - Imprime métricas multirun (5 semillas) con Accuracy, Weighted-F1, Macro-F1,
    F1-Attempt, Kappa y perfilado de coste computacional.
Complejidad:
  - Temporal: O(E * S * |GRID_C| * N_tr * D) en la variante con búsqueda logarítmica en Dev.
  - Espacial: O(N * D).
"""

import sys
from pathlib import Path
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extraccion_mentalbert import MentalBertHierarchicalExtractor
from src.metricas_clasificacion import (
    CLASES_ORDEN_CSSRS,
    GRID_C_LOGARITMICO,
    calibrar_umbral_en_dev,
    cargar_particiones_y_tfidf,
    evaluar_predicciones_clinicas,
    perfilar_entrenamiento_e_inferencia,
    resumir_metricas_multirun,
    seleccionar_c_en_dev,
)


def ejecutar_experimento_svm_opt(semillas: List[int] = [42, 43, 44, 45, 46]) -> pd.DataFrame:
    """Ejecuta el estudio de ablación y optimización de SVM sobre espacio híbrido."""
    extractor = MentalBertHierarchicalExtractor()
    rep_dict = extractor.obtener_o_generar_representaciones()

    nombre_att = "Attempt"
    print("=" * 115)
    print("OPTIMIZACIÓN EXCLUSIVA PARA SUPPORT VECTOR MACHINES (LINEARSVC)")
    print("=" * 115)

    # (Nombre, emb_type, modo_c, class_weight, l2_norm, threshold_dev)
    experimentos = [
        ("1. Híbrido Truncado + C=1.0 + Simétrico", "plano_truncado", "fixed_1.0", None, False, False),
        ("2. Híbrido Truncado + C* log-grid Dev + Simétrico", "plano_truncado", "select_dev_log", None, False, False),
        ("3. Híbrido Truncado + C=1.0 + L2-norm híbrida", "plano_truncado", "fixed_1.0", None, True, False),
        ("4. Híbrido Truncado + C=1.0 + Umbral Dev (theta*)", "plano_truncado", "fixed_1.0", None, False, True),
        ("5. Híbrido Jerárquico Mean-Mean + C=1.0 + Simétrico", "mean_mean", "fixed_1.0", None, False, False),
        ("6. Híbrido Jerárquico Mean-Mean + C* log-grid Dev + Balanced", "mean_mean", "select_dev_log", "balanced", False, False),
        ("7. Híbrido Jerárquico Mean-Mean + C=1.0 + Balanced", "mean_mean", "fixed_1.0", "balanced", False, False),
        ("8. Híbrido Jerárquico Mean-Mean + C=1.0 + Umbral Dev (theta*)", "mean_mean", "fixed_1.0", None, False, True),
    ]

    filas = []
    particiones, _, tfidf_por_semilla = cargar_particiones_y_tfidf(
        semillas=semillas,
        k_tfidf=100,
        clase_critica=nombre_att,
        objetivo_chi2="binario",
        precalcular_tfidf=True,
    )

    for exp_nom, emb_type, modo_c, cw, use_l2, use_th in experimentos:
        metricas_por_semilla = []
        c_seleccionados = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            emb_base = rep_dict[emb_type]
            X_tf_tr, X_tf_dev, X_tf_te = tfidf_por_semilla[s]
            X_tr = np.hstack([emb_base[idx_tr], X_tf_tr * 2.0])
            X_dev = np.hstack([emb_base[idx_dev], X_tf_dev * 2.0])
            X_te = np.hstack([emb_base[idx_te], X_tf_te * 2.0])

            if use_l2:
                X_tr = normalize(X_tr, norm="l2")
                X_dev = normalize(X_dev, norm="l2")
                X_te = normalize(X_te, norm="l2")

            if modo_c == "select_dev_log":
                c_val, _ = seleccionar_c_en_dev(
                    X_tr, y_tr, X_dev, y_dev,
                    grid_c=GRID_C_LOGARITMICO, class_weight=cw, seed=s, clase_critica=nombre_att,
                )
            else:
                c_val = 1.0
            c_seleccionados.append(c_val)

            def _crear_svm(c_param: float = c_val, cw_val: Any = cw, seed_val: int = s) -> LinearSVC:
                return LinearSVC(C=c_param, class_weight=cw_val, random_state=seed_val, max_iter=5000, dual=False)

            clf, y_pred_base, costes = perfilar_entrenamiento_e_inferencia(_crear_svm, X_tr, y_tr, X_te)
            idx_att_clf = list(clf.classes_).index(nombre_att)
            dec_te = clf.decision_function(X_te)
            scores_att_te = dec_te[:, idx_att_clf]

            if use_th:
                margin_att_dev = clf.decision_function(X_dev)[:, idx_att_clf]
                best_th, _ = calibrar_umbral_en_dev(margin_att_dev, y_dev, clase_critica=nombre_att, n_pasos=100)
                y_pred = y_pred_base.copy()
                y_pred[scores_att_te >= best_th] = nombre_att
            else:
                y_pred = y_pred_base

            m_te = evaluar_predicciones_clinicas(
                y_true=y_te,
                y_pred=y_pred,
                clases_orden=CLASES_ORDEN_CSSRS,
                y_scores_attempt=scores_att_te,
                clase_critica=nombre_att,
            )
            m_te.update(costes)
            metricas_por_semilla.append(m_te)

        res = resumir_metricas_multirun(metricas_por_semilla, clases_orden=CLASES_ORDEN_CSSRS)
        res["Experimento"] = exp_nom
        res["C_mean"] = float(np.mean(c_seleccionados))
        filas.append(res)

        print(
            f"{exp_nom:<61} (C={res['C_mean']:.2f}) | Acc: {res['Accuracy_mean']:.3f} | "
            f"w-F1: {res['Weighted_F1_mean']:.3f} | F1(s42): {res['F1_Attempt_seed42']:.3f} | "
            f"F1(μ): {res['F1_Attempt_mean']:.3f}±{res['F1_Attempt_std']:.3f} | "
            f"Rec: {res['Recall_Attempt_mean']:.3f} | Macro: {res['Macro_F1_mean']:.3f} | "
            f"Kap: {res['Quadratic_Kappa_mean']:.3f}"
        )

    return pd.DataFrame(filas)


if __name__ == "__main__":
    ejecutar_experimento_svm_opt()
