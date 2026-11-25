"""
Experimento 2: Respuesta empírica a la RQ2 (Auditado contra AUDITORIA_DOCENTE.md).
«¿Qué mecanismo supervisado resulta más eficaz y clínicamente viable para mitigar el colapso del Recall
ante el desbalanceo severo (9% en Attempt): la reponderación de la función de coste lineal mediante penalización
asimétrica (r = C_FN / C_FP) o la complementariedad ortogonal de una arquitectura híbrida neuro-simbólica
(MentalBERT + TF-IDF de términos críticos con Support Vector Machines)?»

Marco experimental estricto:
  1. Justificación analítica del espacio de búsqueda de costes asimétricos:
     El ratio natural de desbalanceo en Train es (350 - 31) / 31 ~= 10.29.
     El rango r in [1.0, 20.0] barre sub-compensación (r < 10), compensación natural (r ~= 10)
     y sobre-penalización clínica (r > 10) en la pérdida squared hinge ponderada:
       min_{w,b} 0.5 ||w||_2^2 + C * sum_k w_k * sum_{i: y_i=k} max(0, 1 - y_i (w^T x_i + b))^2
  2. Separación estricta Train / Dev / Test (Cero Data Leakage):
     - Vectorizador TF-IDF y selector chi2 (K=100) se ajustan exclusivamente en Train (n=350).
     - La selección de r* en Fase 1 y de C* (rejilla logarítmica base 10) y theta* en Fase 2
       se realiza exclusivamente sobre Dev (n=75).
     - Test (n=75) permanece congelado como evaluación ciega final.
  3. Estabilidad y costes: 5 particiones estratificadas ([42, 43, 44, 45, 46]), reportando
     semilla oficial 42, media +/- desviación estándar, latencia (ms/doc) y pico de RAM (MB).

Precondiciones:
  - Corpus CSV y caché MentalBERT disponibles.
Postcondiciones:
  - Serializa resultados en data_processed/resultados_rq2_costes_pareto.json.
  - Genera gráficas a 300 DPI en data_processed/graficas_clasificacion/.
Complejidad:
  - Temporal: O(S * (|R| * N_tr * D + M * N_tr * (D + K))) donde S=5 semillas, |R|=9 ratios, M=9 modelos.
  - Espacial: O(N * (D + K)).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.modulo_comun_equipo import RUTA_CSV_DEFECTO, cargar_particion_oficial_equipo, limpiar_texto_reddit
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

DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"
DIR_DATA_PROCESSED = PROJECT_ROOT / "data_processed"


def ejecutar_experimento_rq2(
    semillas: List[int] = [42, 43, 44, 45, 46],
    ratios_coste: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
) -> Dict[str, Any]:
    """
    Ejecuta el barrido de costes asimétricos (Fase 1) y la comparativa de modelos
    híbridos neuro-simbólicos con optimización en Dev (Fase 2).
    """
    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    print("=" * 95)
    print("EJECUTANDO EXPERIMENTO RQ2: ASIMETRÍA DE COSTES, PARETO Y MODELO HÍBRIDO SVM")
    print("=" * 95)

    nombre_attempt = "Attempt"
    print(f"[*] Clases ordinales C-SSRS: {CLASES_ORDEN_CSSRS} | Clase crítica: '{nombre_attempt}'")
    print(f"[*] Pre-cargando particiones estratificadas Train/Dev/Test para semillas {semillas}...")

    particiones = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones[s] = {
            "tr": (p["idx"][0], p["y"][0]),
            "dev": (p["idx"][1], p["y"][1]),
            "te": (p["idx"][2], p["y"][2]),
        }

    extractor = MentalBertHierarchicalExtractor()
    rep_dict = extractor.obtener_o_generar_representaciones()

    X_mean_mean = rep_dict["mean_mean"]
    curva_costes = []

    print("\n--- FASE 1: BARRIDO DE LA FUNCIÓN DE COSTE ASIMÉTRICA EN MENTALBERT MEAN-MEAN ---")
    for r in ratios_coste:
        metricas_test_r = []
        f1_dev_r = []
        macro_dev_r = []

        pesos_clase = {c: 1.0 for c in CLASES_ORDEN_CSSRS}
        pesos_clase[nombre_attempt] = float(r)

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            X_tr = X_mean_mean[idx_tr]
            X_dev = X_mean_mean[idx_dev]
            X_te = X_mean_mean[idx_te]

            def _crear_svm_coste(seed_val: int = s, w_dict: Dict[str, float] = pesos_clase) -> LinearSVC:
                return LinearSVC(C=1.0, class_weight=w_dict, random_state=seed_val, max_iter=5000, dual=False)

            clf, y_pred_te, costes = perfilar_entrenamiento_e_inferencia(_crear_svm_coste, X_tr, y_tr, X_te)

            # Evaluación en Dev exclusivamente para selección de r* sin fuga de Test
            y_pred_dev = clf.predict(X_dev)
            m_dev = evaluar_predicciones_clinicas(y_dev, y_pred_dev, clases_orden=CLASES_ORDEN_CSSRS)
            f1_dev_r.append(m_dev["F1_Attempt"])
            macro_dev_r.append(m_dev["Macro_F1"])

            idx_att_clf = list(clf.classes_).index(nombre_attempt)
            scores_att_te = clf.decision_function(X_te)[:, idx_att_clf]
            m_te = evaluar_predicciones_clinicas(
                y_true=y_te,
                y_pred=y_pred_te,
                clases_orden=CLASES_ORDEN_CSSRS,
                y_scores_attempt=scores_att_te,
                clase_critica=nombre_attempt,
            )
            m_te.update(costes)
            metricas_test_r.append(m_te)

        res_r = resumir_metricas_multirun(metricas_test_r, clases_orden=CLASES_ORDEN_CSSRS)
        res_r["Ratio_Coste"] = float(r)
        res_r["F1_Attempt_dev_mean"] = float(np.mean(f1_dev_r))
        res_r["Macro_F1_dev_mean"] = float(np.mean(macro_dev_r))
        curva_costes.append(res_r)

        print(
            f"  [Coste r={r:>4.1f}] -> Dev F1(Att): {res_r['F1_Attempt_dev_mean']:.3f} | "
            f"Test Acc: {res_r['Accuracy_mean']:.3f} | Rec(Att): {res_r['Recall_Attempt_mean']:.3f}±{res_r['Recall_Attempt_std']:.3f} | "
            f"Prec(Att): {res_r['Precision_Attempt_mean']:.3f} | F1(Att): {res_r['F1_Attempt_mean']:.3f} | "
            f"Macro-F1: {res_r['Macro_F1_mean']:.3f} | Kappa: {res_r['Quadratic_Kappa_mean']:.3f}"
        )

    df_curva = pd.DataFrame(curva_costes)
    # Selección formal de r* en Dev (cero data leakage sobre Test)
    idx_optimo_dev = df_curva["F1_Attempt_dev_mean"].idxmax()
    punto_optimo_pareto = df_curva.loc[idx_optimo_dev]
    r_optimo = float(punto_optimo_pareto["Ratio_Coste"])
    print(f"\n[*] Punto de operación seleccionado en Dev (sin fuga de Test): r*={r_optimo:.1f}")
    print(
        f"    En Test ciego -> Acc={punto_optimo_pareto['Accuracy_mean']:.3f} | "
        f"Recall(Attempt)={punto_optimo_pareto['Recall_Attempt_mean']:.3f} | "
        f"F1(Attempt)={punto_optimo_pareto['F1_Attempt_mean']:.3f} | "
        f"Macro-F1={punto_optimo_pareto['Macro_F1_mean']:.3f}"
    )

    print("\n--- FASE 2: ESTUDIO DE ENRIQUECIMIENTO NEURO-SIMBÓLICO Y OPTIMIZACIÓN SVM ---")
    df_raw = pd.read_csv(RUTA_CSV_DEFECTO)
    textos_raw = np.array(df_raw["Post"].fillna("").apply(limpiar_texto_reddit).tolist())

    # Precalcular TF-IDF + chi2 exactamente una vez por semilla (idéntico resultado, 6x más rápido)
    tfidf_por_semilla = {}
    for s in semillas:
        idx_tr_s, y_tr_s = particiones[s]["tr"]
        idx_dev_s, _ = particiones[s]["dev"]
        idx_te_s, _ = particiones[s]["te"]
        tfidf_por_semilla[s] = construir_espacio_hibrido(
            None, None, None,
            textos_raw[idx_tr_s], textos_raw[idx_dev_s], textos_raw[idx_te_s],
            y_tr_s, clase_critica=nombre_attempt, k_tfidf=100, solo_tfidf=True,
        )

    # Modos: (Nombre descriptivo, tipo_base, modo_c, modo_pesos, use_dev_threshold)
    # modo_c: "fixed_1.0" o "select_dev_log" (escala logarítmica base 10 en Dev)
    modelos_comparativa = [
        ("1. Plano Truncado (512)", "plano", "fixed_1.0", "Simétrico (r=1.0)", False),
        ("1. Plano Truncado (512)", "plano", "fixed_1.0", "Heurístico Balanced", False),
        ("2. Jerárquico Mean-Mean", "mean_mean", "fixed_1.0", "Simétrico (r=1.0)", False),
        ("2. Jerárquico Mean-Mean", "mean_mean", "fixed_1.0", "Heurístico Balanced", False),
        ("3. Solo TF-IDF Crítico", "tfidf_solo", "fixed_1.0", "Simétrico (r=1.0)", False),
        ("3. Solo TF-IDF Crítico", "tfidf_solo", "fixed_1.0", "Heurístico Balanced", False),
        ("4. Híbrido Propuesto (Oficial)", "hibrido_plano", "fixed_1.0", "Simétrico (r=1.0)", False),
        ("4. Híbrido Propuesto (Robusto)", "hibrido_mean_max_l2", "select_dev_log", "Heurístico Balanced", False),
        ("4. Híbrido Propuesto (Triaje Dev)", "hibrido_plano", "fixed_1.0", "Calibrado Dev theta* (Prec>=0.20)", True),
    ]

    filas_comparativa = []

    for desc_mod, tipo_mod, modo_c, modo_pesos, use_dev_th in modelos_comparativa:
        metricas_por_semilla = []
        c_seleccionados = []
        thetas_seleccionados = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            if tipo_mod == "plano":
                X_tr = rep_dict["plano_truncado"][idx_tr]
                X_dev = rep_dict["plano_truncado"][idx_dev]
                X_te = rep_dict["plano_truncado"][idx_te]
            elif tipo_mod == "mean_mean":
                X_tr = rep_dict["mean_mean"][idx_tr]
                X_dev = rep_dict["mean_mean"][idx_dev]
                X_te = rep_dict["mean_mean"][idx_te]
            elif tipo_mod == "tfidf_solo":
                X_tr, X_dev, X_te = tfidf_por_semilla[s]
            elif tipo_mod == "hibrido_plano":
                X_tf_tr, X_tf_dev, X_tf_te = tfidf_por_semilla[s]
                X_tr = np.hstack([rep_dict["plano_truncado"][idx_tr], X_tf_tr * 2.0])
                X_dev = np.hstack([rep_dict["plano_truncado"][idx_dev], X_tf_dev * 2.0])
                X_te = np.hstack([rep_dict["plano_truncado"][idx_te], X_tf_te * 2.0])
            elif tipo_mod == "hibrido_mean_max_l2":
                X_tf_tr, X_tf_dev, X_tf_te = tfidf_por_semilla[s]
                X_mm_tr = normalize(rep_dict["mean_mean"][idx_tr], norm="l2")
                X_mm_dev = normalize(rep_dict["mean_mean"][idx_dev], norm="l2")
                X_mm_te = normalize(rep_dict["mean_mean"][idx_te], norm="l2")
                X_mx_tr = normalize(rep_dict["max_mean"][idx_tr], norm="l2")
                X_mx_dev = normalize(rep_dict["max_mean"][idx_dev], norm="l2")
                X_mx_te = normalize(rep_dict["max_mean"][idx_te], norm="l2")
                X_tr = np.hstack([X_mm_tr, X_mx_tr, normalize(X_tf_tr, norm="l2")])
                X_dev = np.hstack([X_mm_dev, X_mx_dev, normalize(X_tf_dev, norm="l2")])
                X_te = np.hstack([X_mm_te, X_mx_te, normalize(X_tf_te, norm="l2")])
            else:
                raise ValueError(f"Tipo de modelo no reconocido: {tipo_mod}")

            cw = "balanced" if "Balanced" in modo_pesos else None

            if modo_c == "select_dev_log":
                c_param, _ = seleccionar_c_en_dev(
                    X_tr, y_tr, X_dev, y_dev,
                    grid_c=GRID_C_LOGARITMICO, class_weight=cw, seed=s, clase_critica=nombre_attempt,
                )
            else:
                c_param = 1.0
            c_seleccionados.append(c_param)

            def _crear_svm_mod(c_val: float = c_param, cw_val: Any = cw, seed_val: int = s) -> LinearSVC:
                return LinearSVC(C=c_val, class_weight=cw_val, random_state=seed_val, max_iter=5000, dual=False)

            clf, y_pred_base, costes = perfilar_entrenamiento_e_inferencia(_crear_svm_mod, X_tr, y_tr, X_te)
            idx_att_clf = list(clf.classes_).index(nombre_attempt)
            dec_te = clf.decision_function(X_te)
            scores_att_te = dec_te[:, idx_att_clf]

            if use_dev_th:
                dec_dev = clf.decision_function(X_dev)
                margin_att_dev = dec_dev[:, idx_att_clf]
                y_pred_dev_base = clf.predict(X_dev)
                best_th, _ = calibrar_umbral_en_dev(
                    margin_att_dev,
                    y_dev,
                    y_pred_dev_base=y_pred_dev_base,
                    prec_min=0.20,
                    clase_critica=nombre_attempt,
                    n_pasos=100,
                )
                thetas_seleccionados.append(best_th)

                y_pred = y_pred_base.copy()
                mask_forzar = scores_att_te >= best_th
                y_pred[mask_forzar] = nombre_attempt
            else:
                y_pred = y_pred_base

            m_te = evaluar_predicciones_clinicas(
                y_true=y_te,
                y_pred=y_pred,
                clases_orden=CLASES_ORDEN_CSSRS,
                y_scores_attempt=scores_att_te,
                clase_critica=nombre_attempt,
            )
            m_te.update(costes)
            metricas_por_semilla.append(m_te)

        res_comp = resumir_metricas_multirun(metricas_por_semilla, clases_orden=CLASES_ORDEN_CSSRS)
        res_comp["Modelo"] = desc_mod
        res_comp["Pesos"] = modo_pesos
        res_comp["C_seed42"] = float(c_seleccionados[0])
        res_comp["C_mean"] = float(np.mean(c_seleccionados))
        if thetas_seleccionados:
            res_comp["theta_seed42"] = float(thetas_seleccionados[0])
            res_comp["theta_mean"] = float(np.mean(thetas_seleccionados))

        filas_comparativa.append(res_comp)
        print(
            f"  {desc_mod:<35} [{modo_pesos:<21}] (C={res_comp['C_mean']:.2f}) -> "
            f"Acc: {res_comp['Accuracy_mean']:.3f} | w-F1: {res_comp['Weighted_F1_mean']:.3f} | "
            f"F1(s42): {res_comp['F1_Attempt_seed42']:.3f} | F1(μ): {res_comp['F1_Attempt_mean']:.3f}±{res_comp['F1_Attempt_std']:.3f} | "
            f"Rec: {res_comp['Recall_Attempt_mean']:.3f} | Prec: {res_comp['Precision_Attempt_mean']:.3f} | "
            f"Macro: {res_comp['Macro_F1_mean']:.3f} | Kappa: {res_comp['Quadratic_Kappa_mean']:.3f}"
        )

    df_comparativa = pd.DataFrame(filas_comparativa)

    # Generación de Figuras Científicas (300 DPI)
    print("\n[*] Generando figuras científicas de la RQ2 a 300 DPI...")
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax1.plot(df_curva["Ratio_Coste"], df_curva["Recall_Attempt_mean"], marker="o", linewidth=2.2, color="#d95f02", label="Recall Attempt (Sensibilidad)")
    ax1.plot(df_curva["Ratio_Coste"], df_curva["F1_Attempt_mean"], marker="s", linewidth=2.2, color="#2ca02c", label="F1-Score Attempt")
    ax1.plot(df_curva["Ratio_Coste"], df_curva["Macro_F1_mean"], marker="^", linewidth=2.0, linestyle="--", color="#2b5c8f", label="Macro-F1 (Global)")
    ax1.plot(df_curva["Ratio_Coste"], df_curva["Accuracy_mean"], marker="d", linewidth=1.8, linestyle="-.", color="#7570b3", label="Accuracy (Global)")

    ax1.scatter([r_optimo], [punto_optimo_pareto["F1_Attempt_mean"]], color="#e41a1c", s=140, zorder=5, edgecolor="black", label=f"r* seleccionado en Dev (r={r_optimo:.1f})")
    ax1.axhline(0.85, color="gray", linestyle=":", linewidth=1.0, label="Referencia Triaje (Recall >= 0.85)")
    ax1.set_xlabel("Ratio de Penalización Asimétrica en Función de Coste ($r = C_{FN} / C_{FP}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Puntuación de Métricas [0.0 - 1.0]", fontsize=11, fontweight="bold")
    ax1.set_title("RQ2: Curva de Pareto y Trade-Off Coste-Sensibilidad (Evaluación en Test)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(ratios_coste)
    ax1.legend(loc="center right", frameon=True, facecolor="white", fontsize=9.0)
    plt.tight_layout()

    out_pareto = DIR_GRAFICAS / "fig_rq2_pareto_frontier.png"
    plt.savefig(out_pareto, dpi=300)
    plt.close()
    print(f"  [OK] Frontera de Pareto guardada en: {out_pareto.name}")

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    df_plot_bars = df_comparativa[df_comparativa["Pesos"].isin(["Simétrico (r=1.0)", "Heurístico Balanced"])].copy()
    sns.barplot(
        data=df_plot_bars,
        x="Modelo",
        y="F1_Attempt_mean",
        hue="Pesos",
        palette=["#2b5c8f", "#d95f02"],
        ax=ax,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_title("RQ2: Impacto del Enriquecimiento Neuro-Simbólico (MentalBERT + TF-IDF Crítico) en F1-Attempt", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("F1-Score en Clase Crítica 'Attempt' (μ)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Arquitectura de Representación", fontsize=11, fontweight="bold")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Plano (512)", "Mean-Mean", "Solo TF-IDF", "Híbrido Oficial", "Híbrido Robusto"], fontsize=9.5, fontweight="bold")
    ax.set_ylim(0.0, 0.40)
    ax.legend(title="Configuración de Costes", title_fontsize=10, loc="upper right")
    plt.tight_layout()

    out_bars = DIR_GRAFICAS / "fig_rq2_cost_gain_bars.png"
    plt.savefig(out_bars, dpi=300)
    plt.close()
    print(f"  [OK] Gráfica comparativa guardada en: {out_bars.name}")

    out_json = DIR_DATA_PROCESSED / "resultados_rq2_costes_pareto.json"
    res_completos = {
        "r_optimo_dev": r_optimo,
        "punto_optimo": punto_optimo_pareto.to_dict(),
        "curva_costes": df_curva.to_dict(orient="records"),
        "comparativa_modelos": df_comparativa.to_dict(orient="records"),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res_completos, f, indent=2, ensure_ascii=False)
    print(f"[*] Resultados numéricos completos serializados en: {out_json}")

    return res_completos


if __name__ == "__main__":
    ejecutar_experimento_rq2()
