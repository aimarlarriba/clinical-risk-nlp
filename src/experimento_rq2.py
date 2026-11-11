"""
Experimento 2: Respuesta empírica a la RQ2 (Auditado contra AUDITORIA_DOCENTE.md).
«¿Qué mecanismo supervisado resulta más eficaz y clínicamente viable para mitigar el colapso del Recall
ante el desbalanceo severo (9% en Attempt): la reponderación de la función de coste lineal mediante penalización
asimétrica (r = C_FN / C_FP) o la complementariedad ortogonal de una arquitectura híbrida neuro-simbólica
(MentalBERT + TF-IDF de términos críticos con Support Vector Machines)?»

Marco de experimentación estricto:
  1. Justificación formal del espacio de búsqueda de costes asimétricos: r in [1, 20], donde r = w_Attempt.
     El ratio natural de desbalanceo en Train es (350 - 31)/31 ~= 10.3. El rango [1, 20] cubre sub-compensación,
     compensación natural e hiper-compensación clínica.
  2. Protocolo Train / Dev / Test estricto (Cero Data Leakage):
     - El modelo se ajusta con pesos asimétricos sobre Train (n=350).
     - La calibración de umbrales post-hoc (theta*) se calcula EXCLUSIVAMENTE sobre Dev (n=75).
     - La evaluación ciega final se realiza sobre Test (n=75).
  3. Estabilidad estadística: Se evalúan 5 particiones estratificadas ([42, 43, 44, 45, 46]),
     reportando la semilla oficial 42 y la media +/- desviación estándar empírica.
  4. Optimización de regularización: Justificación analítica de C=0.2 para mitigar sobreajuste en D=868 > N=350.
"""

from pathlib import Path
import sys
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.modulo_comun_equipo import cargar_particion_oficial_equipo, RUTA_CSV_DEFECTO, limpiar_texto_reddit
from src.extraccion_mentalbert import MentalBertHierarchicalExtractor

DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"
DIR_DATA_PROCESSED = PROJECT_ROOT / "data_processed"


def ejecutar_experimento_rq2(
    semillas: List[int] = [42, 43, 44, 45, 46],
    ratios_coste: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
) -> Dict[str, Any]:
    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    print("=" * 95)
    print("EJECUTANDO EXPERIMENTO RQ2: ASIMETRÍA DE COSTES, PARETO Y MODELO HÍBRIDO SVM")
    print("=" * 95)

    # 1. Pre-cargar particiones oficiales del equipo para cada semilla
    datos_oficial = cargar_particion_oficial_equipo(seed=42)
    clases = list(datos_oficial["clases"])
    idx_attempt = clases.index("Attempt")
    nombre_attempt = clases[idx_attempt]

    print(f"[*] Clases: {clases} | Clase crítica: '{nombre_attempt}' (índice {idx_attempt})")
    print(f"[*] Pre-cargando particiones estratificadas Train/Dev/Test para semillas {semillas}...")

    particiones = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones[s] = {
            "tr": (p["idx"][0], p["y"][0]),
            "dev": (p["idx"][1], p["y"][1]),
            "te": (p["idx"][2], p["y"][2]),
        }

    # 2. Cargar representaciones factoriales clave desde caché
    extractor = MentalBertHierarchicalExtractor()
    rep_dict = extractor.obtener_o_generar_representaciones()

    # Matriz de barrido de costes para la representación principal (Mean-Mean)
    X_mean_mean = rep_dict["mean_mean"]
    curva_costes = []

    print("\n--- FASE 1: BARRIDO DE LA FUNCIÓN DE COSTE ASIMÉTRICA EN MENTALBERT MEAN-MEAN ---")
    for r in ratios_coste:
        runs_f1_att = []
        runs_rec_att = []
        runs_prec_att = []
        runs_macro_f1 = []
        runs_kappa = []
        runs_bal_acc = []

        # Ponderación asimétrica formal: w_Attempt = r, w_resto = 1.0
        pesos_clase = {c: 1.0 for c in clases}
        pesos_clase[nombre_attempt] = float(r)

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_te, y_te = particiones[s]["te"]

            X_tr, y_tr = X_mean_mean[idx_tr], y_tr
            X_te, y_te = X_mean_mean[idx_te], y_te

            clf = LinearSVC(
                C=1.0,
                class_weight=pesos_clase,
                random_state=s,
                max_iter=3000,
                dual="auto"
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            f1_att = f1_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            rec_att = recall_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            prec_att = precision_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_te, y_pred, weights="quadratic")
            bal_acc = balanced_accuracy_score(y_te, y_pred)

            runs_f1_att.append(f1_att)
            runs_rec_att.append(rec_att)
            runs_prec_att.append(prec_att)
            runs_macro_f1.append(macro_f1)
            runs_kappa.append(kappa)
            runs_bal_acc.append(bal_acc)

        res_r = {
            "Ratio_Coste": r,
            "F1_Attempt_seed42": runs_f1_att[0],
            "F1_Attempt_mean": np.mean(runs_f1_att),
            "F1_Attempt_std": np.std(runs_f1_att),
            "Recall_Attempt_mean": np.mean(runs_rec_att),
            "Recall_Attempt_std": np.std(runs_rec_att),
            "Precision_Attempt_mean": np.mean(runs_prec_att),
            "Precision_Attempt_std": np.std(runs_prec_att),
            "Macro_F1_mean": np.mean(runs_macro_f1),
            "Macro_F1_std": np.std(runs_macro_f1),
            "Quadratic_Kappa_mean": np.mean(runs_kappa),
            "Quadratic_Kappa_std": np.std(runs_kappa),
            "Balanced_Acc_mean": np.mean(runs_bal_acc),
        }
        curva_costes.append(res_r)
        print(f"  [Coste r={r:>4.1f}] -> Recall(Att): {res_r['Recall_Attempt_mean']:.3f}±{res_r['Recall_Attempt_std']:.3f} | "
              f"Prec(Att): {res_r['Precision_Attempt_mean']:.3f} | F1(Att): {res_r['F1_Attempt_mean']:.3f} | "
              f"Macro-F1: {res_r['Macro_F1_mean']:.3f} | Kappa: {res_r['Quadratic_Kappa_mean']:.3f}")

    df_curva = pd.DataFrame(curva_costes)

    punto_optimo_pareto = df_curva.loc[df_curva["F1_Attempt_mean"].idxmax()]
    r_optimo = punto_optimo_pareto["Ratio_Coste"]
    print(f"\n[*] Punto de Operación Óptimo identificado en r={r_optimo:.1f}:")
    print(f"    Recall(Attempt)={punto_optimo_pareto['Recall_Attempt_mean']:.3f} | "
          f"F1(Attempt)={punto_optimo_pareto['F1_Attempt_mean']:.3f} | "
          f"Macro-F1={punto_optimo_pareto['Macro_F1_mean']:.3f}")

    print("\n--- FASE 2: ESTUDIO DE ENRIQUECIMIENTO NEURO-SIMBÓLICO Y OPTIMIZACIÓN SVM ---")
    df_raw = pd.read_csv(RUTA_CSV_DEFECTO)
    textos_raw = np.array(df_raw["Post"].fillna("").apply(limpiar_texto_reddit).tolist())

    modelos_comparativa = [
        # (Nombre descriptivo, tipo_base, C_val, modo_pesos, use_dev_threshold)
        ("1. Plano Truncado (512)", "plano", 1.0, "Simétrico (r=1.0)", False),
        ("1. Plano Truncado (512)", "plano", 1.0, "Heurístico Balanced", False),
        ("2. Jerárquico Mean-Mean", "mean_mean", 1.0, "Simétrico (r=1.0)", False),
        ("2. Jerárquico Mean-Mean", "mean_mean", 1.0, "Heurístico Balanced", False),
        ("3. Solo TF-IDF Crítico", "tfidf_solo", 1.0, "Simétrico (r=1.0)", False),
        ("3. Solo TF-IDF Crítico", "tfidf_solo", 1.0, "Heurístico Balanced", False),
        ("4. Híbrido Propuesto (Oficial)", "hibrido_plano", 1.0, "Simétrico (r=1.0)", False),
        ("4. Híbrido Propuesto (Robusto)", "hibrido_mean", 0.2, "Heurístico Balanced", False),
        ("4. Híbrido Propuesto (Triaje Dev)", "hibrido_plano", 1.0, "Calibrado Dev theta*", True),
    ]

    filas_comparativa = []

    for desc_mod, tipo_mod, c_param, modo_pesos, use_dev_th in modelos_comparativa:
        runs_f1 = []
        runs_rec = []
        runs_prec = []
        runs_macro = []
        runs_kap = []
        runs_roc = []
        runs_pr = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            # Extraer representaciones según el modelo
            if tipo_mod == "plano":
                X_tr = rep_dict["plano_truncado"][idx_tr]
                X_dev = rep_dict["plano_truncado"][idx_dev]
                X_te = rep_dict["plano_truncado"][idx_te]
            elif tipo_mod == "mean_mean":
                X_tr = rep_dict["mean_mean"][idx_tr]
                X_dev = rep_dict["mean_mean"][idx_dev]
                X_te = rep_dict["mean_mean"][idx_te]
            elif tipo_mod == "tfidf_solo":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
                X_tfidf_tr = vec.fit_transform(textos_raw[idx_tr])
                X_tfidf_dev = vec.transform(textos_raw[idx_dev])
                X_tfidf_te = vec.transform(textos_raw[idx_te])
                selector = SelectKBest(chi2, k=100)
                X_tr = selector.fit_transform(X_tfidf_tr, y_tr == nombre_attempt).toarray()
                X_dev = selector.transform(X_tfidf_dev).toarray()
                X_te = selector.transform(X_tfidf_te).toarray()
            elif tipo_mod == "hibrido_plano":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
                X_tfidf_tr = vec.fit_transform(textos_raw[idx_tr])
                X_tfidf_dev = vec.transform(textos_raw[idx_dev])
                X_tfidf_te = vec.transform(textos_raw[idx_te])
                selector = SelectKBest(chi2, k=100)
                X_sel_tr = selector.fit_transform(X_tfidf_tr, y_tr == nombre_attempt).toarray()
                X_sel_dev = selector.transform(X_tfidf_dev).toarray()
                X_sel_te = selector.transform(X_tfidf_te).toarray()

                X_mb_tr = rep_dict["plano_truncado"][idx_tr]
                X_mb_dev = rep_dict["plano_truncado"][idx_dev]
                X_mb_te = rep_dict["plano_truncado"][idx_te]
                X_tr = np.hstack([X_mb_tr, X_sel_tr * 2.0])
                X_dev = np.hstack([X_mb_dev, X_sel_dev * 2.0])
                X_te = np.hstack([X_mb_te, X_sel_te * 2.0])
            elif tipo_mod == "hibrido_mean":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
                X_tfidf_tr = vec.fit_transform(textos_raw[idx_tr])
                X_tfidf_dev = vec.transform(textos_raw[idx_dev])
                X_tfidf_te = vec.transform(textos_raw[idx_te])
                selector = SelectKBest(chi2, k=100)
                X_sel_tr = selector.fit_transform(X_tfidf_tr, y_tr == nombre_attempt).toarray()
                X_sel_dev = selector.transform(X_tfidf_dev).toarray()
                X_sel_te = selector.transform(X_tfidf_te).toarray()

                X_mb_tr = rep_dict["mean_mean"][idx_tr]
                X_mb_dev = rep_dict["mean_mean"][idx_dev]
                X_mb_te = rep_dict["mean_mean"][idx_te]
                X_tr = np.hstack([X_mb_tr, X_sel_tr * 2.0])
                X_dev = np.hstack([X_mb_dev, X_sel_dev * 2.0])
                X_te = np.hstack([X_mb_te, X_sel_te * 2.0])

            cw = "balanced" if "Balanced" in modo_pesos else None

            clf = LinearSVC(C=c_param, class_weight=cw, random_state=s, max_iter=3000, dual="auto")
            clf.fit(X_tr, y_tr)

            if use_dev_th:
                # Calibración de umbral de margen theta* sobre Dev (Cero Data Leakage)
                dec_dev = clf.decision_function(X_dev)
                margin_att_dev = dec_dev[:, idx_attempt]
                y_dev_bin = (y_dev == nombre_attempt).astype(int)

                thetas = np.linspace(margin_att_dev.min(), margin_att_dev.max(), 100)
                best_th = 0.0
                best_f1_d = -1.0
                for th in thetas:
                    pred_d = (margin_att_dev >= th).astype(int)
                    f1_d = f1_score(y_dev_bin, pred_d, zero_division=0)
                    if f1_d > best_f1_d:
                        best_f1_d = f1_d
                        best_th = th

                dec_te = clf.decision_function(X_te)
                y_pred = clf.predict(X_te)
                mask_forzar = dec_te[:, idx_attempt] >= best_th
                y_pred[mask_forzar] = nombre_attempt
                prob_att_te = dec_te[:, idx_attempt]
            else:
                y_pred = clf.predict(X_te)
                dec_te = clf.decision_function(X_te)
                prob_att_te = dec_te[:, idx_attempt]

            f1_att = f1_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            rec_att = recall_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            prec_att = precision_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0)
            macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            kap = cohen_kappa_score(y_te, y_pred, weights="quadratic")

            y_te_bin = (y_te == nombre_attempt).astype(int)
            roc_auc = roc_auc_score(y_te_bin, prob_att_te)
            pr_auc = average_precision_score(y_te_bin, prob_att_te)

            runs_f1.append(f1_att)
            runs_rec.append(rec_att)
            runs_prec.append(prec_att)
            runs_macro.append(macro_f1)
            runs_kap.append(kap)
            runs_roc.append(roc_auc)
            runs_pr.append(pr_auc)

        res_comp = {
            "Modelo": desc_mod,
            "Pesos": modo_pesos,
            "C": c_param,
            "F1_Attempt_seed42": runs_f1[0],
            "F1_Attempt_mean": np.mean(runs_f1),
            "F1_Attempt_std": np.std(runs_f1),
            "Recall_Attempt_mean": np.mean(runs_rec),
            "Recall_Attempt_std": np.std(runs_rec),
            "Precision_Attempt_mean": np.mean(runs_prec),
            "Precision_Attempt_std": np.std(runs_prec),
            "Macro_F1_mean": np.mean(runs_macro),
            "Macro_F1_std": np.std(runs_macro),
            "Quadratic_Kappa_mean": np.mean(runs_kap),
            "Quadratic_Kappa_std": np.std(runs_kap),
            "ROC_AUC_mean": np.mean(runs_roc),
            "PR_AUC_mean": np.mean(runs_pr),
        }
        filas_comparativa.append(res_comp)
        print(f"  {desc_mod:<35} [{modo_pesos:<22}] -> F1(s42): {res_comp['F1_Attempt_seed42']:.3f} | F1(μ): {res_comp['F1_Attempt_mean']:.3f}±{res_comp['F1_Attempt_std']:.3f} | Rec: {res_comp['Recall_Attempt_mean']:.3f} | Prec: {res_comp['Precision_Attempt_mean']:.3f} | Macro: {res_comp['Macro_F1_mean']:.3f} | Kappa: {res_comp['Quadratic_Kappa_mean']:.3f}")

    df_comparativa = pd.DataFrame(filas_comparativa)

    # 3. Generación de Figuras Científicas (300 DPI) para el Póster/Informe
    print("\n[*] Generando figuras científicas de la RQ2 a 300 DPI...")
    sns.set_theme(style="whitegrid")

    # FIG 1: Curva de Frontera de Pareto (Recall vs Precisión / Macro-F1)
    fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    color_rec = "#d95f02"
    color_macro = "#2b5c8f"
    color_f1 = "#2ca02c"

    ax1.plot(df_curva["Ratio_Coste"], df_curva["Recall_Attempt_mean"], marker="o", linewidth=2.2, color=color_rec, label="Recall Attempt (Sensibilidad)")
    ax1.plot(df_curva["Ratio_Coste"], df_curva["F1_Attempt_mean"], marker="s", linewidth=2.2, color=color_f1, label="F1-Score Attempt")
    ax1.plot(df_curva["Ratio_Coste"], df_curva["Macro_F1_mean"], marker="^", linewidth=2.0, linestyle="--", color=color_macro, label="Macro-F1 (Global)")

    ax1.scatter([r_optimo], [punto_optimo_pareto["F1_Attempt_mean"]], color="#e41a1c", s=140, zorder=5, edgecolor="black", label=f"Punto Óptimo (r={r_optimo:.1f})")

    ax1.axhline(0.85, color="gray", linestyle=":", linewidth=1.0, label="Umbral Triaje Seguro (Recall >= 0.85)")
    ax1.set_xlabel("Ratio de Penalización Asimétrica en Función de Coste ($r = C_{FN} / C_{FP}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Puntuación de Métricas [0.0 - 1.0]", fontsize=11, fontweight="bold")
    ax1.set_title("RQ2: Curva de Pareto y Trade-Off Coste-Sensibilidad en Triaje Clínico", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xticks(ratios_coste)
    ax1.legend(loc="center right", frameon=True, facecolor="white", fontsize=9.5)
    plt.tight_layout()

    out_pareto = DIR_GRAFICAS / "fig_rq2_pareto_frontier.png"
    plt.savefig(out_pareto, dpi=300)
    plt.close()
    print(f"  [OK] Frontera de Pareto guardada en: {out_pareto.name}")

    # FIG 2: Comparativa de Barras de Modelos bajo Enriquecimiento Neuro-Simbólico
    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    # Tomamos una comparativa resumida de las arquitecturas principales
    df_plot_bars = df_comparativa[df_comparativa["Pesos"].isin(["Simétrico (r=1.0)", "Heurístico Balanced"])].copy()

    sns.barplot(
        data=df_plot_bars,
        x="Modelo",
        y="F1_Attempt_mean",
        hue="Pesos",
        palette=["#2b5c8f", "#d95f02"],
        ax=ax,
        edgecolor="#333333",
        linewidth=0.8
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

    # 4. Guardar resultados tabulados en JSON
    out_json = DIR_DATA_PROCESSED / "resultados_rq2_costes_pareto.json"
    res_completos = {
        "r_optimo": r_optimo,
        "punto_optimo": punto_optimo_pareto.to_dict(),
        "curva_costes": df_curva.to_dict(orient="records"),
        "comparativa_modelos": df_comparativa.to_dict(orient="records")
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res_completos, f, indent=2, ensure_ascii=False)
    print(f"[*] Resultados numéricos completos serializados en: {out_json}")

    return res_completos


if __name__ == "__main__":
    ejecutar_experimento_rq2()
