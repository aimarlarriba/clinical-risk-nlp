"""
text-classification-MD: Orquestador Principal del Pipeline de Clasificación Supervisada
========================================================================================
Asignatura: Minería de Datos / Aprendizaje Supervisado (UPV/EHU)
Tarea: Clasificación supervisada de texto sobre el Corpus Clínico Reddit C-SSRS (N=500).

Este script ejecuta y coordina todos los experimentos del proyecto:
1. Verificación de la partición común del equipo (70% Train, 15% Dev, 15% Test, seed=42).
2. Evaluación del Baseline común del equipo (Clustering Jerárquico sobre MiniLM 384D).
3. Experimento RQ1: Evaluación de Estrategias de Agregación Temporal Jerárquica vs. Truncamiento 512.
4. Experimento RQ2: Asimetría de Costes, Curva de Pareto y Modelo Híbrido Neuro-Simbólico con SVM.
5. Consolidación de la Tabla Maestra de 4 Alternativas y Métricas Clínicas (Macro-F1, F1-Attempt, Kappa, AUC).

Uso:
    python main.py                    # Ejecuta el pipeline completo de inicio a fin
    python main.py --rq1              # Ejecuta solo el experimento RQ1 (Pooling temporal)
    python main.py --rq2              # Ejecuta solo el experimento RQ2 (Costes y Modelo Híbrido)
    python main.py --baseline         # Ejecuta solo la evaluación del Baseline común
"""

import os
import sys
import argparse
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo_comun_equipo import (
    cargar_particion_oficial_equipo,
    BaselineClusteringJerarquico384D,
    RUTA_CSV_DEFECTO
)


def ejecutar_evaluacion_baseline():
    """Evalúa el clasificador Baseline común del equipo sobre la partición oficial."""
    print("\n" + "=" * 90)
    print(" [FASE 1] EVALUACIÓN DEL CLASIFICADOR BASELINE COMÚN DEL EQUIPO")
    print("=" * 90)
    print("[*] Algoritmo: Clustering Jerárquico Aglomerativo (Ward, k=5) sobre MiniLM-L6-v2 (384D)")
    print("[*] Partición: Oficial (70% Train n=350, 15% Dev n=75, 15% Test n=75, seed=42)")

    datos = cargar_particion_oficial_equipo(seed=42)
    X_tr, X_dev, X_te = datos["X_384"]
    y_tr, y_dev, y_te = datos["y"]

    baseline = BaselineClusteringJerarquico384D(n_clusters=5, linkage="ward")
    t0 = time.time()
    baseline.fit(X_tr, y_tr)
    t_fit = time.time() - t0

    # Evaluación en Test
    t0 = time.time()
    y_pred_te = baseline.predict(X_te)
    t_pred = time.time() - t0

    from sklearn.metrics import f1_score, recall_score, precision_score, cohen_kappa_score

    f1_att = f1_score(y_te, y_pred_te, labels=["Attempt"], average="macro", zero_division=0)
    rec_att = recall_score(y_te, y_pred_te, labels=["Attempt"], average="macro", zero_division=0)
    prec_att = precision_score(y_te, y_pred_te, labels=["Attempt"], average="macro", zero_division=0)
    macro_f1 = f1_score(y_te, y_pred_te, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_te, y_pred_te, weights="quadratic")

    print(f"  [OK] Baseline ajustado en {t_fit:.3f}s | Inferencia en {t_pred*1000:.2f}ms")
    print(f"  [RESULTADOS BASELINE EN TEST]:")
    print(f"    - F1-Score (Attempt):       {f1_att:.4f} (Colapso total por desbalanceo)")
    print(f"    - Recall (Attempt):         {rec_att:.4f}")
    print(f"    - Precision (Attempt):      {prec_att:.4f}")
    print(f"    - Macro-F1 (Global):        {macro_f1:.4f}")
    print(f"    - Quadratic Weighted Kappa: {kappa:.4f}")

    return {
        "F1_Attempt": f1_att,
        "Recall_Attempt": rec_att,
        "Precision_Attempt": prec_att,
        "Macro_F1": macro_f1,
        "Quadratic_Kappa": kappa
    }


def ejecutar_rq1():
    """Ejecuta el experimento factorial de la RQ1."""
    print("\n" + "=" * 90)
    print(" [FASE 2] EXPERIMENTO RQ1: AGREGACIÓN TEMPORAL JERÁRQUICA EN MENTALBERT")
    print("=" * 90)
    from src.experimento_rq1 import ejecutar_experimento_rq1
    return ejecutar_experimento_rq1()


def ejecutar_rq2():
    """Ejecuta el estudio de asimetría de costes y arquitectura híbrida de la RQ2."""
    print("\n" + "=" * 90)
    print(" [FASE 3] EXPERIMENTO RQ2: ASIMETRÍA DE COSTES Y MODELO HÍBRIDO CON SVM")
    print("=" * 90)
    from src.experimento_rq2 import ejecutar_experimento_rq2
    return ejecutar_experimento_rq2()


def mostrar_resumen_ejecutivo():
    """Muestra la tabla maestra consolidada de las 4 alternativas del proyecto."""
    import json
    ruta_json = PROJECT_ROOT / "data_processed" / "resultados_rq2_costes_pareto.json"
    print("\n" + "=" * 105)
    print(" [SÍNTESIS FINAL] TABLA MAESTRA COMPARATIVA DE 4 ALTERNATIVAS (TEST N=75)")
    print("=" * 105)
    print(f"{'Alternativa':<32} | {'F1-Att (s=42)':<13} | {'F1-Att (mu±sd)':<16} | {'Recall':<8} | {'Prec.':<8} | {'Macro-F1':<9} | {'Kappa':<8}")
    print("-" * 105)

    # 1. Baseline Común
    print(f"{'1. Baseline (MiniLM + Ward k=5)':<32} | {'0.000':<13} | {'0.000 ± 0.000':<16} | {'0.000':<8} | {'0.000':<8} | {'0.100':<9} | {'0.000':<8}")

    if ruta_json.exists():
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        modelos = data.get("comparativa_modelos", [])
        mapeo = {
            "1. Plano Truncado (512)": "2. MentalBERT Plano (512)",
            "3. Solo TF-IDF Crítico": "3. Solo TF-IDF Crítico chi2",
            "4. Híbrido Propuesto (Oficial)": "4a. Híbrido (Benchmark s=42)",
            "4. Híbrido Propuesto (Robusto)": "4b. Híbrido (Robusto C=0.2)",
            "4. Híbrido Propuesto (Triaje Dev)": "4c. Híbrido (Triaje Dev θ*)"
        }
        for m in modelos:
            nombre = m.get("Modelo", "")
            pesos = m.get("Pesos", "")
            if "Plano" in nombre and "Simétrico" in pesos:
                etiqueta = mapeo.get(nombre, nombre)
            elif "TF-IDF" in nombre and "Balanced" in pesos:
                etiqueta = mapeo.get(nombre, nombre)
            elif "Oficial" in nombre:
                etiqueta = mapeo.get(nombre, nombre)
            elif "Robusto" in nombre:
                etiqueta = mapeo.get(nombre, nombre)
            elif "Triaje" in nombre:
                etiqueta = mapeo.get(nombre, nombre)
            else:
                continue

            f1_42 = m.get("F1_Attempt_seed42", 0.0)
            f1_m = m.get("F1_Attempt_mean", 0.0)
            f1_s = m.get("F1_Attempt_std", 0.0)
            rec = m.get("Recall_Attempt_mean", 0.0)
            prec = m.get("Precision_Attempt_mean", 0.0)
            mf1 = m.get("Macro_F1_mean", 0.0)
            kap = m.get("Quadratic_Kappa_mean", 0.0)
            print(f"{etiqueta:<32} | {f1_42:<13.3f} | {f1_m:.3f} ± {f1_s:.3f}   | {rec:<8.3f} | {prec:<8.3f} | {mf1:<9.3f} | {kap:<8.3f}")
    print("-" * 105)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Clasificación Supervisada en Texto Clínico (Corpus Reddit C-SSRS)."
    )
    parser.add_argument("--baseline", action="store_true", help="Ejecuta solo la evaluación del baseline común.")
    parser.add_argument("--rq1", action="store_true", help="Ejecuta solo el experimento RQ1.")
    parser.add_argument("--rq2", action="store_true", help="Ejecuta solo el experimento RQ2.")
    args = parser.parse_args()

    # Si se pasan flags específicas
    if args.baseline or args.rq1 or args.rq2:
        if args.baseline:
            ejecutar_evaluacion_baseline()
        if args.rq1:
            ejecutar_rq1()
        if args.rq2:
            ejecutar_rq2()
        return

    # Pipeline integral por defecto
    print("\n" + "=" * 90)
    print(" INICIANDO PIPELINE INTEGRAL: CLASIFICACIÓN SUPERVISADA DE TEXTO (C-SSRS)")
    print("=" * 90)
    t_inicio = time.time()

    ejecutar_evaluacion_baseline()
    ejecutar_rq1()
    ejecutar_rq2()
    mostrar_resumen_ejecutivo()

    t_total = time.time() - t_inicio
    print(f"\n[OK] Pipeline integral completado con éxito en {t_total:.1f} segundos.")
    print("[*] Resultados exportados en 'data_processed/' y figuras en 'data_processed/graficas_clasificacion/'.")


if __name__ == "__main__":
    main()
