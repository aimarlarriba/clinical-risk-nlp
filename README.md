# Evaluación de MentalBERT y Support Vector Machines en Texto Clínico (C-SSRS)

Este repositorio contiene la arquitectura experimental, pipeline de extracción de representaciones y evaluación empírica para la detección de riesgo de suicidio en redes sociales (Corpus Reddit C-SSRS, $N=500$ usuarios).

---

## Preguntas de Investigación (Research Questions)

### 💎 RQ1 (Dimensión Temporal: Truncamiento vs. Agregación Jerárquica):
> *«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto completo vs. Post-level Max-Pooling) a la capacidad de un clasificador SVM para identificar publicaciones aisladas de crisis aguda (`Attempt`) sin diluir la señal en publicaciones previas inocuas?»*
* **Hallazgo Clave:** La representación `Jerárquico Mean-Mean` es la más estable entre particiones independientes ($\text{Macro-F1} = 0.327 \pm 0.035$, $\text{F1-Attempt} = 0.196 \pm 0.119$, $\text{Precisión} = 0.407$). Las variantes con Max-Pooling (`Max-Mean`) maximizan la consistencia ordinal de la escala C-SSRS ($\text{Kappa} = 0.344 \pm 0.101$). Sin embargo, la SVM simétrica colapsa a Recall $< 0.15$ debido al desbalanceo severo (9% de prevalencia).

### 💎 RQ2 (Dimensión de Optimización y Representación: Asimetría de Costes vs. Modelo Híbrido):
> *«¿Qué mecanismo supervisado resulta más eficaz y clínicamente viable para mitigar el colapso del Recall ante el desbalanceo severo (9% en `Attempt`): la intervención de la función de coste lineal mediante penalización asimétrica ($r = C_{\text{FN}} / C_{\text{FP}}$) o la complementariedad ortogonal de una arquitectura híbrida neuro-simbólica (MentalBERT + TF-IDF de términos críticos)?»*
* **Hallazgo Clave:** La reponderación lineal de costes satura el Recall en $\approx 0.171$ a expensas de desplomar la precisión del $40.7\%$ al $8.2\%$. Por el contrario, la arquitectura híbrida neuro-simbólica (MentalBERT + TF-IDF Crítico con $\chi^2$ ajustado en Train) rompe esta limitación:
  - En la semilla oficial (Seed 42), alcanza **F1 = 0.600 con un 100% de precisión** (cero falsos positivos).
  - En evaluación multillamada con regularización $C=0.2$ y pesos balanceados, alcanza el **máximo Quadratic Weighted Kappa del estudio ($0.402 \pm 0.081$)** y Macro-F1 de $0.349$.
  - Mediante calibración post-hoc de umbral $\theta^*$ en Dev, alcanza un **$51.4\%$ de Recall en Test** para triaje clínico sin data leakage.

---

## Reproducibilidad de los Experimentos

Todos los experimentos se ejecutan bajo la partición oficial del equipo (70/15/15, $N=500$, `seed=42`) y se validan con 5 particiones estratificadas ($[42, 43, 44, 45, 46]$):

```bash
# 1. Ejecutar experimento RQ1 (Pooling temporal y figuras 300 DPI)
python src/experimento_rq1.py

# 2. Ejecutar experimento RQ2 (Barrido de costes, frontera de Pareto y figuras 300 DPI)
python src/experimento_rq2.py
```

Las figuras científicas a 300 DPI se guardan automáticamente en `data_processed/graficas_clasificacion/` y el informe detallado está disponible en `INFORME_CIENTIFICO_C-SSRS.md`.

