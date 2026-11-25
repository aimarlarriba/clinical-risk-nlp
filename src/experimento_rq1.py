"""
Experimento 1: Respuesta empírica a la RQ1 (Auditado bajo AUDITORIA_DOCENTE.md).
«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto completo
vs. Post-level Max-Pooling) a la capacidad de un clasificador SVM para identificar publicaciones aisladas
de crisis aguda (Attempt) sin diluir la señal en publicaciones previas inocuas?»

Compara 5 estrategias de agregación sobre MentalBERT (768D) con LinearSVC (C=1.0, L2 squared hinge):
  1. Plano Truncado (Mean-Pooling texto concatenado tradicional, 512 tokens)
  2. Jerárquico Mean-Mean (Mean fragmentos -> Mean usuario)
  3. Jerárquico Max-Mean  (Max fragmentos  -> Mean usuario)
  4. Jerárquico Mean-Max  (Mean fragmentos -> Max usuario)
  5. Jerárquico Max-Max   (Max fragmentos  -> Max usuario)

Precondiciones:
  - Corpus CSV accesible y representaciones MentalBERT en caché o extraíbles.
Postcondiciones:
  - Serializa métricas multirun (incluyendo Accuracy, Macro-F1, Weighted-F1, F1 por clase,
    Quadratic Kappa y costes computacionales) en data_processed/resultados_rq1_pooling.json.
  - Genera figuras a 300 DPI en data_processed/graficas_clasificacion/.
Complejidad:
  - Temporal: O(E * S * N_tr * D) donde E=5 estrategias, S=5 semillas, N_tr=350, D=768.
  - Espacial: O(N * D).
"""

import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.modulo_comun_equipo import cargar_particion_oficial_equipo
from src.extraccion_mentalbert import MentalBertHierarchicalExtractor
from src.metricas_clasificacion import (
    CLASES_ORDEN_CSSRS,
    evaluar_predicciones_clinicas,
    perfilar_entrenamiento_e_inferencia,
    resumir_metricas_multirun,
)

DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"


def ejecutar_experimento_rq1(semillas: List[int] = [42, 43, 44, 45, 46]) -> pd.DataFrame:
    """
    Ejecuta la comparativa factorial de agregación temporal en MentalBERT + LinearSVC
    sobre 5 particiones estratificadas reproducibles.
    """
    DIR_GRAFICAS.mkdir(parents=True, exist_ok=True)
    print("=" * 95)
    print("EJECUTANDO EXPERIMENTO RQ1: IMPACTO DE LA AGREGACIÓN TEMPORAL (MENTALBERT + LINEARSVC)")
    print("=" * 95)

    particiones_semillas = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones_semillas[s] = {
            "idx_tr": p["idx"][0],
            "idx_te": p["idx"][2],
            "y_tr": p["y"][0],
            "y_te": p["y"][2],
        }

    print(
        f"[*] Clases ordinales C-SSRS: {CLASES_ORDEN_CSSRS} | "
        f"Train={len(particiones_semillas[42]['y_tr'])}, Test={len(particiones_semillas[42]['y_te'])}"
    )

    extractor = MentalBertHierarchicalExtractor()
    representaciones_dict = extractor.obtener_o_generar_representaciones()

    estrategias = [
        ("1. Plano Truncado (512 tokens)", "plano_truncado"),
        ("2. Jerárquico Mean-Mean", "mean_mean"),
        ("3. Jerárquico Max-Mean", "max_mean"),
        ("4. Jerárquico Mean-Max", "mean_max"),
        ("5. Jerárquico Max-Max", "max_max"),
    ]

    filas_resumen = []

    for idx_est, (nombre_est, clave_dict) in enumerate(estrategias, 1):
        print(f"\n[Estrategia {idx_est}/5] Evaluando '{nombre_est}' (clave: {clave_dict})...", flush=True)
        X_all = representaciones_dict[clave_dict]
        metricas_por_semilla = []

        for s in semillas:
            p_s = particiones_semillas[s]
            X_tr_s = X_all[p_s["idx_tr"]]
            X_te_s = X_all[p_s["idx_te"]]
            y_tr_s = p_s["y_tr"]
            y_te_s = p_s["y_te"]

            def _crear_svm(seed_val: int = s) -> LinearSVC:
                return LinearSVC(C=1.0, random_state=seed_val, max_iter=5000, dual=False)

            clf, y_pred, costes = perfilar_entrenamiento_e_inferencia(_crear_svm, X_tr_s, y_tr_s, X_te_s)
            idx_att_clf = list(clf.classes_).index("Attempt")
            scores_att = clf.decision_function(X_te_s)[:, idx_att_clf]

            metricas = evaluar_predicciones_clinicas(
                y_true=y_te_s,
                y_pred=y_pred,
                clases_orden=CLASES_ORDEN_CSSRS,
                y_scores_attempt=scores_att,
                clase_critica="Attempt",
            )
            metricas.update(costes)
            metricas_por_semilla.append(metricas)

            print(
                f"    -> Seed {s}: Acc={metricas['Accuracy']:.3f} | Macro-F1={metricas['Macro_F1']:.3f} | "
                f"w-F1={metricas['Weighted_F1']:.3f} | F1-Att={metricas['F1_Attempt']:.3f} | "
                f"Kappa={metricas['Quadratic_Kappa']:.3f} | Lat={costes['infer_latency_ms_doc']:.3f} ms/doc",
                flush=True,
            )

        resumen = resumir_metricas_multirun(metricas_por_semilla, clases_orden=CLASES_ORDEN_CSSRS)
        resumen["Estrategia"] = nombre_est
        resumen["Clave"] = clave_dict
        filas_resumen.append(resumen)

    df_rq1 = pd.DataFrame(filas_resumen)

    print("\n" + "=" * 115)
    print("RESULTADOS EXPERIMENTALES RQ1 EN TEST (Seed Oficial = 42 | Estabilidad: 5 Semillas [42..46])")
    print("=" * 115)
    for _, r in df_rq1.iterrows():
        print(
            f"{r['Estrategia']:<30} | Acc: {r['Accuracy_mean']:.3f}±{r['Accuracy_std']:.3f} | "
            f"Macro-F1: {r['Macro_F1_mean']:.3f}±{r['Macro_F1_std']:.3f} | "
            f"w-F1: {r['Weighted_F1_mean']:.3f} | "
            f"F1-Att (42): {r['F1_Attempt_seed42']:.3f} | "
            f"F1-Att (μ±σ): {r['F1_Attempt_mean']:.3f}±{r['F1_Attempt_std']:.3f} | "
            f"Kappa: {r['Quadratic_Kappa_mean']:.3f}"
        )

    # Generación de Figuras Científicas (300 DPI)
    print("\n[*] Generando figuras científicas Seaborn para RQ1 (300 DPI)...")
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")

    nombres_cortos = ["Plano (512)", "Mean-Mean", "Max-Mean", "Mean-Max", "Max-Max"]
    x = np.arange(len(nombres_cortos))
    width = 0.27

    plt.bar(
        x - width,
        df_rq1["F1_Attempt_mean"],
        width=width,
        yerr=df_rq1["F1_Attempt_std"],
        capsize=4,
        label="F1 Attempt (Clase Crítica)",
        color="#d95f02",
        edgecolor="#555555",
        linewidth=0.8,
    )
    plt.bar(
        x,
        df_rq1["Macro_F1_mean"],
        width=width,
        yerr=df_rq1["Macro_F1_std"],
        capsize=4,
        label="Macro-F1 (Global)",
        color="#2b5c8f",
        edgecolor="#555555",
        linewidth=0.8,
    )
    plt.bar(
        x + width,
        df_rq1["Accuracy_mean"],
        width=width,
        yerr=df_rq1["Accuracy_std"],
        capsize=4,
        label="Accuracy (Global)",
        color="#1b9e77",
        edgecolor="#555555",
        linewidth=0.8,
    )

    plt.xticks(x, nombres_cortos, fontsize=10, fontweight="bold")
    plt.ylabel("Puntuación [0.0 - 1.0]", fontsize=11, fontweight="bold")
    plt.title(
        "RQ1: Impacto de la Agregación Temporal en Detección C-SSRS (5 Particiones, μ ± σ)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9.5)
    plt.tight_layout()

    out_bar = DIR_GRAFICAS / "fig_rq1_pooling_bars.png"
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"  [OK] Gráfica de barras: {out_bar.name}")

    plt.figure(figsize=(6, 4.8), dpi=300)
    sns.set_theme(style="white")

    matriz_2x2 = np.array([
        [
            df_rq1.loc[df_rq1["Clave"] == "mean_mean", "F1_Attempt_mean"].values[0],
            df_rq1.loc[df_rq1["Clave"] == "mean_max", "F1_Attempt_mean"].values[0],
        ],
        [
            df_rq1.loc[df_rq1["Clave"] == "max_mean", "F1_Attempt_mean"].values[0],
            df_rq1.loc[df_rq1["Clave"] == "max_max", "F1_Attempt_mean"].values[0],
        ],
    ])

    sns.heatmap(
        matriz_2x2,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=["Mean Inter-Post", "Max Inter-Post (Temporal)"],
        yticklabels=["Mean Intra-Post", "Max Intra-Post"],
        cbar_kws={"label": "F1-Score en Attempt"},
        linewidths=1.0,
        linecolor="#cccccc",
        annot_kws={"size": 13, "weight": "bold"},
    )
    plt.title("RQ1: Matriz Factorial 2x2 de F1-Score en 'Attempt'", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Nivel 2: Agregación Temporal de Posts", fontsize=10, fontweight="bold", labelpad=8)
    plt.ylabel("Nivel 1: Agregación de Fragmentos", fontsize=10, fontweight="bold", labelpad=8)
    plt.tight_layout()

    out_heat = DIR_GRAFICAS / "fig_rq1_heatmap_2x2.png"
    plt.savefig(out_heat, dpi=300)
    plt.close()
    print(f"  [OK] Mapa de calor 2x2: {out_heat.name}")

    out_json = PROJECT_ROOT / "data_processed" / "resultados_rq1_pooling.json"
    df_rq1.to_json(out_json, orient="records", indent=2)
    print(f"[*] Resultados numéricos serializados en: {out_json}")

    return df_rq1


if __name__ == "__main__":
    ejecutar_experimento_rq1()
