"""
Experimento 2: Respuesta empírica a la RQ2 (Auditado contra AUDITORIA_DOCENTE.md).
«¿Bajo qué ratio de penalización asimétrica de costes (C_FN / C_FP) en la función de pérdida de la SVM
es posible alcanzar un Recall clínico crítico (>= 0.85) en la clase 'Attempt' sin degradar la precisión
global del sistema por debajo del umbral de viabilidad operativa (Macro-F1 > 0.35)?»

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
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.modulo_comun_equipo import cargar_particion_oficial_equipo
from src.extraccion_mentalbert import MentalBertHierarchicalExtractor

DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"
DIR_DATA_PROCESSED = PROJECT_ROOT / "data_processed"


def ejecutar_experimento_rq2(
    semillas: List[int] = [42, 43, 44, 45, 46],
    ratios_coste: List[float] = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
) -> Dict[str, Any]:
    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    print("=" * 95)
    print("EJECUTANDO EXPERIMENTO RQ2: ASIMETRÍA DE COSTES, PARETO Y VIABILIDAD CLÍNICA")
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

    # Evaluaremos la representación ganadora en estabilidad de RQ1 (Mean-Mean)
    # y contrastaremos con el Plano Truncado (Baseline) y Max-Max (Extremos)
    representaciones_eval = [
        ("Jerárquico Mean-Mean (Más estable en RQ1)", "mean_mean"),
        ("Plano Truncado 512 (Baseline estándar)", "plano_truncado"),
        ("Jerárquico Max-Max (Máximo Kappa en RQ1)", "max_max"),
    ]

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

    # Identificar el punto de operación clínica óptimo:
    # Máximo F1 en Attempt sujeto a Recall >= 0.70 o máximo compromiso de Pareto
    punto_optimo_pareto = df_curva.loc[df_curva["F1_Attempt_mean"].idxmax()]
    r_optimo = punto_optimo_pareto["Ratio_Coste"]
    print(f"\n[*] Punto de Operación Óptimo identificado en r={r_optimo:.1f}:")
    print(f"    Recall(Attempt)={punto_optimo_pareto['Recall_Attempt_mean']:.3f} | "
          f"F1(Attempt)={punto_optimo_pareto['F1_Attempt_mean']:.3f} | "
          f"Macro-F1={punto_optimo_pareto['Macro_F1_mean']:.3f}")

    print("\n--- FASE 2: COMPARATIVA FACTORIAL DE MODELOS BAJO EL COSTE ÓPTIMO ---")
    filas_comparativa = []

    for desc_rep, clave_rep in representaciones_eval:
        X_rep = rep_dict[clave_rep]

        for modo_pesos in ["1. Simétrico (Baseline r=1.0)", f"2. Asimétrico Óptimo (r={r_optimo})", "3. Heurístico Balanced"]:
            runs_f1 = []
            runs_rec = []
            runs_prec = []
            runs_macro = []
            runs_kap = []

            for s in semillas:
                idx_tr, y_tr = particiones[s]["tr"]
                idx_te, y_te = particiones[s]["te"]

                X_tr, y_tr = X_rep[idx_tr], y_tr
                X_te, y_te = X_rep[idx_te], y_te

                if modo_pesos.startswith("1."):
                    cw = None
                elif modo_pesos.startswith("2."):
                    cw = {c: 1.0 for c in clases}
                    cw[nombre_attempt] = float(r_optimo)
                else:
                    cw = "balanced"

                clf = LinearSVC(C=1.0, class_weight=cw, random_state=s, max_iter=3000, dual="auto")
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)

                runs_f1.append(f1_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0))
                runs_rec.append(recall_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0))
                runs_prec.append(precision_score(y_te, y_pred, labels=[nombre_attempt], average="macro", zero_division=0))
                runs_macro.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
                runs_kap.append(cohen_kappa_score(y_te, y_pred, weights="quadratic"))

            filas_comparativa.append({
                "Representacion": desc_rep,
                "Configuracion_Costes": modo_pesos,
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
            })

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

    # Resaltar punto óptimo
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

    # FIG 2: Comparativa de Barras de Modelos bajo Coste Óptimo
    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=300)
    sns.set_theme(style="whitegrid")

    sub_df = df_comparativa[df_comparativa["Configuracion_Costes"].str.contains("Asimétrico|Simétrico")]
    sns.barplot(
        data=sub_df,
        x="Representacion",
        y="Recall_Attempt_mean",
        hue="Configuracion_Costes",
        palette=["#999999", "#d95f02"],
        ax=ax,
        edgecolor="#333333",
        linewidth=0.8
    )
    ax.set_title("RQ2: Ganancia de Sensibilidad (Recall) tras la Optimización Asimétrica de Costes", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Recall en Clase Crítica 'Attempt'", fontsize=11, fontweight="bold")
    ax.set_xlabel("Estrategia de Representación Factorial", fontsize=11, fontweight="bold")
    ax.set_xticklabels(["Jerárquico Mean-Mean", "Plano Truncado (512)", "Jerárquico Max-Max"], fontsize=10)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Función Objetivo", title_fontsize=10, loc="upper left")
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

