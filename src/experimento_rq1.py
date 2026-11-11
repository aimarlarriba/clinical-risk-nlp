"""
Experimento 1: Respuesta empírica a la RQ1.
«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto completo
vs. Post-level Max-Pooling) a la capacidad de un clasificador SVM para identificar publicaciones aisladas
de crisis aguda (Attempt) sin diluir la señal en publicaciones previas inocuas?»

Compara 5 estrategias de agregación sobre MentalBERT (768D) con Linear SVM:
  1. Plano Truncado (Mean-Pooling texto concatenado tradicional, 512 tokens)
  2. Jerárquico Mean-Mean (Mean fragmentos -> Mean usuario)
  3. Jerárquico Max-Mean  (Max fragmentos  -> Mean usuario)
  4. Jerárquico Mean-Max  (Mean fragmentos -> Max usuario)
  5. Jerárquico Max-Max   (Max fragmentos  -> Max usuario)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

from src.modulo_comun_equipo import cargar_particion_oficial_equipo
from src.extraccion_mentalbert import MentalBertHierarchicalExtractor

DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"


def ejecutar_experimento_rq1(semillas: List[int] = [42, 43, 44, 45, 46]) -> pd.DataFrame:
    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("EJECUTANDO EXPERIMENTO RQ1: IMPACTO DE LA AGREGACIÓN TEMPORAL (MENTALBERT + SVM)")
    print("=" * 80)

    # 1. Cargar partición común oficial del equipo (70/15/15, seed=42, stratify=y)
    # y pre-cargar las 5 particiones estratificadas para el análisis de varianza estadística
    datos_oficial = cargar_particion_oficial_equipo(seed=42)
    clases = list(datos_oficial["clases"])
    idx_attempt = clases.index("Attempt")

    print(f"[*] Clases oficiales: {clases} (Clase crítica 'Attempt' indexada en {idx_attempt})")
    print(f"[*] Pre-cargando particiones estratificadas para semillas {semillas}...", flush=True)

    particiones_semillas = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones_semillas[s] = {
            "idx_tr": p["idx"][0],
            "idx_te": p["idx"][2],
            "y_tr": p["y"][0],
            "y_te": p["y"][2],
        }
    print(f"[*] Partición oficial Train={len(particiones_semillas[42]['y_tr'])} | Test={len(particiones_semillas[42]['y_te'])}")

    # 2. Cargar las 5 representaciones factoriales de MentalBERT desde caché
    extractor = MentalBertHierarchicalExtractor()
    representaciones_dict = extractor.obtener_o_generar_representaciones()

    estrategias = [
        ("1. Plano Truncado (512 tokens)", "plano_truncado"),
        ("2. Jerárquico Mean-Mean", "mean_mean"),
        ("3. Jerárquico Max-Mean",  "max_mean"),
        ("4. Jerárquico Mean-Max",  "mean_max"),
        ("5. Jerárquico Max-Max",   "max_max"),
    ]

    filas_resumen = []
    detalles_runs = {}

    for idx_est, (nombre_est, clave_dict) in enumerate(estrategias, 1):
        print(f"\n[Estrategia {idx_est}/5] Evaluando '{nombre_est}' (clave: {clave_dict})...", flush=True)
        X_all = representaciones_dict[clave_dict]

        runs_f1_att = []
        runs_rec_att = []
        runs_prec_att = []
        runs_macro_f1 = []
        runs_bal_acc = []
        runs_kappa = []

        # Evaluación sobre las semillas: s=42 es la oficial; el resto evalúa estabilidad de partición
        for s in semillas:
            p_s = particiones_semillas[s]
            X_tr_s = X_all[p_s["idx_tr"]]
            X_te_s = X_all[p_s["idx_te"]]
            y_tr_s = p_s["y_tr"]
            y_te_s = p_s["y_te"]

            # LinearSVC estándar (L2-penalized Squared Hinge Loss, C=1.0)
            clf = LinearSVC(C=1.0, random_state=s, max_iter=2000, dual="auto")
            clf.fit(X_tr_s, y_tr_s)
            y_pred = clf.predict(X_te_s)

            f1_att = f1_score(y_te_s, y_pred, labels=[clases[idx_attempt]], average="macro", zero_division=0)
            rec_att = recall_score(y_te_s, y_pred, labels=[clases[idx_attempt]], average="macro", zero_division=0)
            prec_att = precision_score(y_te_s, y_pred, labels=[clases[idx_attempt]], average="macro", zero_division=0)
            macro_f1 = f1_score(y_te_s, y_pred, average="macro", zero_division=0)
            bal_acc = balanced_accuracy_score(y_te_s, y_pred)
            kappa = cohen_kappa_score(y_te_s, y_pred, weights="quadratic")

            print(f"    -> Seed {s}: F1-Attempt={f1_att:.3f} | Recall={rec_att:.3f} | Prec={prec_att:.3f} | Macro-F1={macro_f1:.3f} | Quad-Kappa={kappa:.3f}", flush=True)

            runs_f1_att.append(f1_att)
            runs_rec_att.append(rec_att)
            runs_prec_att.append(prec_att)
            runs_macro_f1.append(macro_f1)
            runs_bal_acc.append(bal_acc)
            runs_kappa.append(kappa)

        filas_resumen.append({
            "Estrategia": nombre_est,
            "Clave": clave_dict,
            "F1_Attempt_seed42": runs_f1_att[0],
            "F1_Attempt_mean": np.mean(runs_f1_att),
            "F1_Attempt_std": np.std(runs_f1_att),
            "Recall_Attempt_mean": np.mean(runs_rec_att),
            "Recall_Attempt_std": np.std(runs_rec_att),
            "Precision_Attempt_mean": np.mean(runs_prec_att),
            "Precision_Attempt_std": np.std(runs_prec_att),
            "Macro_F1_seed42": runs_macro_f1[0],
            "Macro_F1_mean": np.mean(runs_macro_f1),
            "Macro_F1_std": np.std(runs_macro_f1),
            "Balanced_Acc_mean": np.mean(runs_bal_acc),
            "Balanced_Acc_std": np.std(runs_bal_acc),
            "Quadratic_Kappa_mean": np.mean(runs_kappa),
            "Quadratic_Kappa_std": np.std(runs_kappa),
        })

    df_rq1 = pd.DataFrame(filas_resumen)

    print("\n" + "=" * 95)
    print("RESULTADOS EXPERIMENTALES RQ1 EN TEST (Seed Oficial = 42 | Estabilidad: 5 Runs)")
    print("=" * 95)
    for _, r in df_rq1.iterrows():
        print(f"{r['Estrategia']:<32} | F1-Attempt (42): {r['F1_Attempt_seed42']:.3f} | F1-Att (μ±σ): {r['F1_Attempt_mean']:.3f}±{r['F1_Attempt_std']:.3f} | Rec-Att: {r['Recall_Attempt_mean']:.3f} | Macro-F1: {r['Macro_F1_mean']:.3f}±{r['Macro_F1_std']:.3f}")

    # 3. Generación de Gráficas Científicas para el Póster (300 DPI)
    print("\n[*] Generando figuras científicas Seaborn para el póster...")

    # Fig 1: Comparativa de Barras F1-Attempt vs Macro-F1
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    nombres_cortos = ["Plano (512)", "Mean-Mean", "Max-Mean", "Mean-Max", "Max-Max"]
    x = np.arange(len(nombres_cortos))
    width = 0.35

    plt.bar(x - width/2, df_rq1["F1_Attempt_mean"], width=width, yerr=df_rq1["F1_Attempt_std"],
            capsize=5, label="F1 Attempt (Clase Crítica)", color="#d95f02", edgecolor="#555555", linewidth=0.8)
    plt.bar(x + width/2, df_rq1["Macro_F1_mean"], width=width, yerr=df_rq1["Macro_F1_std"],
            capsize=5, label="Macro-F1 (Global)", color="#2b5c8f", edgecolor="#555555", linewidth=0.8)

    plt.xticks(x, nombres_cortos, fontsize=10, fontweight="bold")
    plt.ylabel("Puntuación [0.0 - 1.0]", fontsize=11, fontweight="bold")
    plt.title("RQ1: Impacto de la Agregación Temporal en la Detección de 'Attempt' (Media ± Desv. Típica)",
              fontsize=11, fontweight="bold", pad=12)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=True, facecolor="white", loc="upper left", fontsize=10)
    plt.tight_layout()

    out_bar = DIR_GRAFICAS / "fig_rq1_pooling_bars.png"
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"  [OK] Gráfica de barras: {out_bar.name}")

    # Fig 2: Heatmap 2x2 Factorial (Intra-Post vs Inter-Post Temporal)
    plt.figure(figsize=(6, 4.8), dpi=300)
    sns.set_theme(style="white")

    matriz_2x2 = np.array([
        [df_rq1.loc[df_rq1['Clave'] == 'mean_mean', 'F1_Attempt_mean'].values[0],
         df_rq1.loc[df_rq1['Clave'] == 'mean_max', 'F1_Attempt_mean'].values[0]],
        [df_rq1.loc[df_rq1['Clave'] == 'max_mean', 'F1_Attempt_mean'].values[0],
         df_rq1.loc[df_rq1['Clave'] == 'max_max', 'F1_Attempt_mean'].values[0]]
    ])

    ax = sns.heatmap(
        matriz_2x2,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=["Mean Inter-Post", "Max Inter-Post (Temporal)"],
        yticklabels=["Mean Intra-Post", "Max Intra-Post"],
        cbar_kws={"label": "F1-Score en Attempt"},
        linewidths=1.0,
        linecolor="#cccccc",
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("RQ1: Matriz Factorial 2x2 de F1-Score en 'Attempt'", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Nivel 2: Agregación Temporal de Posts", fontsize=10, fontweight="bold", labelpad=8)
    plt.ylabel("Nivel 1: Agregación de Fragmentos", fontsize=10, fontweight="bold", labelpad=8)
    plt.tight_layout()

    out_heat = DIR_GRAFICAS / "fig_rq1_heatmap_2x2.png"
    plt.savefig(out_heat, dpi=300)
    plt.close()
    print(f"  [OK] Mapa de calor 2x2: {out_heat.name}")

    # 4. Guardar resultados tabulados en JSON
    out_json = PROJECT_ROOT / "data_processed" / "resultados_rq1_pooling.json"
    df_rq1.to_json(out_json, orient="records", indent=2)
    print(f"[*] Resultados numéricos serializados en: {out_json}")

    return df_rq1


if __name__ == "__main__":
    ejecutar_experimento_rq1()

