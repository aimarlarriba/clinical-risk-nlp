"""
clinical-risk-nl: Orquestador Principal del Pipeline de Clasificación Supervisada
========================================================================================
Asignatura: Minería de Datos / Aprendizaje Supervisado (UPV/EHU)
Tarea: Clasificación supervisada de texto sobre el Corpus Clínico Reddit C-SSRS (N=500).

Este script ejecuta y coordina los experimentos del proyecto bajo el protocolo AUDITORIA_DOCENTE.md:
1. Evaluación del Baseline común del equipo (Clustering Jerárquico Ward sobre MiniLM 384D).
2. Experimento RQ1: Evaluación factorial de agregación temporal jerárquica vs. truncamiento 512.
3. Experimento RQ2: Asimetría de costes, frontera de Pareto y arquitectura híbrida neuro-simbólica con SVM.
4. Consolidación de la Tabla Maestra de alternativas con Accuracy, Macro-F1, Weighted-F1,
   F1-Attempt, F1 detallado por clase, Kappa cuadrático y perfilado de costes.

Uso:
    python main.py                    # Ejecuta baseline, RQ1, RQ2 y resumen ejecutivo
    python main.py --baseline         # Ejecuta solo la evaluación del Baseline común
    python main.py --rq1              # Ejecuta solo el experimento RQ1 (Pooling temporal)
    python main.py --rq2              # Ejecuta solo el experimento RQ2 (Costes y Modelo Híbrido)
    python main.py --mejoras          # Ejecuta estudio complementario SVM (RBF, Ordinal, L2, clustering)
    python main.py --svm-opt          # Ejecuta ablación de optimizaciones SVM
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.metricas_clasificacion import (
    CLASES_ORDEN_CSSRS,
    evaluar_predicciones_clinicas,
    perfilar_entrenamiento_e_inferencia,
    resumir_metricas_multirun,
)
from src.modulo_comun_equipo import (
    BaselineClusteringJerarquico384D,
    cargar_particion_oficial_equipo,
)


def ejecutar_evaluacion_baseline(semillas: List[int] = [42, 43, 44, 45, 46]) -> Dict[str, Any]:
    """
    Evalúa el clasificador Baseline común del equipo sobre la partición oficial (seed=42,
    determinista con StDev=0.0000 entre ejecuciones repetidas) y sobre 5 particiones
    estratificadas [42..46] para cuantificar su sensibilidad al muestreo.

    Precondiciones:
        - CSV oficial y embeddings 384D accesibles mediante `cargar_particion_oficial_equipo`.
    Postcondiciones:
        - Serializa las métricas y costes en `data_processed/resultados_baseline.json`.
    Complejidad:
        - Temporal: O(S * N_tr^2) por el enlace Ward aglomerativo en Train.
        - Espacial: O(N_tr^2).
    """
    print("\n" + "=" * 95)
    print(" [FASE 1] EVALUACIÓN DEL CLASIFICADOR BASELINE COMÚN DEL EQUIPO")
    print("=" * 95)
    print("[*] Algoritmo: Clustering Jerárquico Aglomerativo (Ward, k=5) sobre MiniLM-L6-v2 (384D)")
    print("[*] Partición oficial: 70% Train (n=350) / 15% Dev (n=75) / 15% Test (n=75), seed=42")

    metricas_por_semilla = []
    for s in semillas:
        datos = cargar_particion_oficial_equipo(seed=s)
        X_tr, _, X_te = datos["X_384"]
        y_tr, _, y_te = datos["y"]

        def _crear_baseline() -> BaselineClusteringJerarquico384D:
            return BaselineClusteringJerarquico384D(n_clusters=5, linkage="ward")

        baseline, y_pred_te, costes = perfilar_entrenamiento_e_inferencia(_crear_baseline, X_tr, y_tr, X_te)
        idx_att = list(baseline.clases_).index("Attempt")
        scores_att = baseline.predict_proba(X_te)[:, idx_att]

        m_te = evaluar_predicciones_clinicas(
            y_true=y_te,
            y_pred=y_pred_te,
            clases_orden=CLASES_ORDEN_CSSRS,
            y_scores_attempt=scores_att,
            clase_critica="Attempt",
        )
        m_te.update(costes)
        metricas_por_semilla.append(m_te)

    resumen = resumir_metricas_multirun(metricas_por_semilla, clases_orden=CLASES_ORDEN_CSSRS)
    m42 = metricas_por_semilla[0]

    print(
        f"  [OK] Semilla oficial 42 (determinista, StDev=0.0000 entre repeticiones) | "
        f"Ajuste: {m42['fit_time_s']:.4f}s | Latencia: {m42['infer_latency_ms_doc']:.4f} ms/doc | "
        f"RAM pico: {m42['peak_ram_mb']:.2f} MB"
    )
    print("  [RESULTADOS BASELINE EN TEST OFICIAL (seed=42)]:")
    print(f"    - Accuracy (Test):          {m42['Accuracy']:.4f}")
    print(f"    - Macro-F1 (Test):          {m42['Macro_F1']:.4f}")
    print(f"    - Weighted-F1 (Test):       {m42['Weighted_F1']:.4f}")
    print(f"    - F1-Score (Attempt):       {m42['F1_Attempt']:.4f}")
    print(f"    - Recall / Prec (Attempt):  {m42['Recall_Attempt']:.4f} / {m42['Precision_Attempt']:.4f}")
    print(f"    - Quadratic Weighted Kappa: {m42['Quadratic_Kappa']:.4f}")
    print(
        f"  [ESTABILIDAD ENTRE 5 PARTICIONES {semillas}]: "
        f"Acc={resumen['Accuracy_mean']:.4f}±{resumen['Accuracy_std']:.4f} | "
        f"Macro-F1={resumen['Macro_F1_mean']:.4f}±{resumen['Macro_F1_std']:.4f}"
    )

    out_dir = PROJECT_ROOT / "data_processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "resultados_baseline.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    return resumen


def ejecutar_rq1():
    """Ejecuta el experimento factorial de la RQ1."""
    print("\n" + "=" * 95)
    print(" [FASE 2] EXPERIMENTO RQ1: AGREGACIÓN TEMPORAL JERÁRQUICA EN MENTALBERT")
    print("=" * 95)
    from src.experimento_rq1 import ejecutar_experimento_rq1
    return ejecutar_experimento_rq1()


def ejecutar_rq2():
    """Ejecuta el estudio de asimetría de costes y arquitectura híbrida de la RQ2."""
    print("\n" + "=" * 95)
    print(" [FASE 3] EXPERIMENTO RQ2: ASIMETRÍA DE COSTES Y MODELO HÍBRIDO CON SVM")
    print("=" * 95)
    from src.experimento_rq2 import ejecutar_experimento_rq2
    return ejecutar_experimento_rq2()


def ejecutar_mejoras():
    """Ejecuta el estudio complementario de mejoras exclusivas con SVM (L2, RBF, Ordinal y clustering)."""
    print("\n" + "=" * 95)
    print(" [EXTENSIÓN] ESTUDIO DE MEJORAS EXCLUSIVAS CON SVM (RBF, ORDINAL, L2 Y CLUSTERING)")
    print("=" * 95)
    from src.experimento_mejoras import ejecutar_experimento_mejoras
    return ejecutar_experimento_mejoras()


def ejecutar_svm_opt():
    """Ejecuta el estudio de ablación de regularización y umbrales para SVM."""
    print("\n" + "=" * 95)
    print(" [EXTENSIÓN] ABLACIÓN DE OPTIMIZACIÓN EXCLUSIVA PARA LINEARSVC")
    print("=" * 95)
    from src.experimento_svm_opt import ejecutar_experimento_svm_opt
    return ejecutar_experimento_svm_opt()


def mostrar_resumen_ejecutivo():
    """
    Muestra la tabla maestra consolidada a partir de los artefactos serializados
    en `data_processed/resultados_baseline.json` y `resultados_rq2_costes_pareto.json`.
    """
    ruta_base = PROJECT_ROOT / "data_processed" / "resultados_baseline.json"
    ruta_rq2 = PROJECT_ROOT / "data_processed" / "resultados_rq2_costes_pareto.json"

    print("\n" + "=" * 126)
    print(" [SÍNTESIS FINAL] TABLA MAESTRA COMPARATIVA DE ALTERNATIVAS EN TEST (N=75, 5 SEMILLAS [42..46])")
    print("=" * 126)
    print(
        f"{'Alternativa':<30} | {'Acc(s42)':<8} | {'Acc(μ±σ)':<13} | {'w-F1':<6} | "
        f"{'Macro-F1':<8} | {'F1-Att(42)':<10} | {'F1-Att(μ±σ)':<13} | {'Kappa':<6} | {'Lat(ms)':<7}"
    )
    print("-" * 126)

    if ruta_base.exists():
        with open(ruta_base, "r", encoding="utf-8") as f:
            b = json.load(f)
        print(
            f"{'1. Baseline (MiniLM+Ward k=5)':<30} | "
            f"{b['Accuracy_seed42']:<8.3f} | "
            f"{b['Accuracy_mean']:.3f}±{b['Accuracy_std']:.3f}   | "
            f"{b['Weighted_F1_mean']:<6.3f} | "
            f"{b['Macro_F1_mean']:<8.3f} | "
            f"{b['F1_Attempt_seed42']:<10.3f} | "
            f"{b['F1_Attempt_mean']:.3f}±{b['F1_Attempt_std']:.3f}   | "
            f"{b['Quadratic_Kappa_mean']:<6.3f} | "
            f"{b['infer_latency_ms_doc_mean']:<7.3f}"
        )

    if ruta_rq2.exists():
        with open(ruta_rq2, "r", encoding="utf-8") as f:
            data = json.load(f)
        modelos = data.get("comparativa_modelos", [])
        mapeo = {
            "plano_sim": "2. MentalBERT Plano (512)",
            "tfidf_bal": "3. Solo TF-IDF Crítico chi2",
            "hibrido_oficial": "4a. Híbrido (Oficial C=1.0)",
            "hibrido_robusto": "4b. Híbrido (Robusto C* Dev)",
            "hibrido_triaje": "4c. Híbrido (Triaje Dev θ*)",
        }
        for m in modelos:
            mod_id = m.get("Modelo_ID")
            if mod_id in mapeo:
                etiqueta = mapeo[mod_id]
            else:
                continue

            print(
                f"{etiqueta:<30} | "
                f"{m['Accuracy_seed42']:<8.3f} | "
                f"{m['Accuracy_mean']:.3f}±{m['Accuracy_std']:.3f}   | "
                f"{m['Weighted_F1_mean']:<6.3f} | "
                f"{m['Macro_F1_mean']:<8.3f} | "
                f"{m['F1_Attempt_seed42']:<10.3f} | "
                f"{m['F1_Attempt_mean']:.3f}±{m['F1_Attempt_std']:.3f}   | "
                f"{m['Quadratic_Kappa_mean']:<6.3f} | "
                f"{m['infer_latency_ms_doc_mean']:<7.3f}"
            )
    print("-" * 126)


def main():
    """Punto de entrada CLI del pipeline experimental y tabla maestra comparativa."""
    parser = argparse.ArgumentParser(
        description="Pipeline de Clasificación Supervisada en Texto Clínico (Corpus Reddit C-SSRS)."
    )
    parser.add_argument("--baseline", action="store_true", help="Ejecuta solo la evaluación del baseline común.")
    parser.add_argument("--rq1", action="store_true", help="Ejecuta solo el experimento RQ1.")
    parser.add_argument("--rq2", action="store_true", help="Ejecuta solo el experimento RQ2.")
    parser.add_argument("--mejoras", action="store_true", help="Ejecuta el estudio complementario de mejoras.")
    parser.add_argument("--svm-opt", action="store_true", help="Ejecuta la ablación de optimización SVM.")
    args = parser.parse_args()

    if args.baseline or args.rq1 or args.rq2 or args.mejoras or args.svm_opt:
        if args.baseline:
            ejecutar_evaluacion_baseline()
        if args.rq1:
            ejecutar_rq1()
        if args.rq2:
            ejecutar_rq2()
        if args.mejoras:
            ejecutar_mejoras()
        if args.svm_opt:
            ejecutar_svm_opt()
        return

    print("\n" + "=" * 95)
    print(" INICIANDO PIPELINE INTEGRAL: CLASIFICACIÓN SUPERVISADA DE TEXTO (C-SSRS)")
    print("=" * 95)
    t_inicio = time.time()

    ejecutar_evaluacion_baseline()
    ejecutar_rq1()
    ejecutar_rq2()
    mostrar_resumen_ejecutivo()

    t_total = time.time() - t_inicio
    print(f"\n[OK] Pipeline integral completado en {t_total:.1f} segundos.")
    print("[*] Resultados exportados en 'data_processed/' y figuras en 'data_processed/graficas_clasificacion/'.")


if __name__ == "__main__":
    main()
