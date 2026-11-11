# Evaluación de MentalBERT y Support Vector Machines en Texto Clínico (C-SSRS)

Este repositorio contiene la arquitectura experimental, pipeline de extracción de representaciones y evaluación empírica para la detección de riesgo de suicidio en redes sociales (Corpus Reddit C-SSRS, $N=500$ usuarios).

---

## Preguntas de Investigación (Research Questions)

### 💎 RQ1 (Dimensión Temporal: Truncamiento vs. Agregación Jerárquica):
> *«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto completo vs. Post-level Max-Pooling) a la capacidad de un clasificador SVM para identificar publicaciones aisladas de crisis aguda (`Attempt`) sin diluir la señal en publicaciones previas inocuas?»*
* **Hallazgo Clave:** La representación `Jerárquico Mean-Mean` es la más estable entre particiones independientes ($\text{Macro-F1} = 0.327 \pm 0.035$, $\text{F1-Attempt} = 0.196 \pm 0.119$, $\text{Precisión} = 0.407$). Las variantes con Max-Pooling (`Max-Mean`) maximizan la consistencia ordinal de la escala C-SSRS ($\text{Kappa} = 0.344 \pm 0.101$). Sin embargo, la SVM simétrica colapsa a Recall $< 0.15$ debido al desbalanceo severo (9% de prevalencia).

### 💎 RQ2 (Dimensión de Optimización: Asimetría de Costes y Viabilidad Clínica):
> *«¿Bajo qué ratio de penalización asimétrica de costes ($r = C_{\text{FN}} / C_{\text{FP}}$) en la función de pérdida squared hinge de la SVM es posible optimizar el compromiso de Pareto en la clase minoritaria crítica `Attempt`, y en qué medida la reponderación de costes mitiga el colapso del Recall observado en la RQ1 sin degradar la viabilidad operativa global del sistema?»*
* **Hallazgo Clave:** El barrido en el espacio analítico $r \in [1, 20]$ satura el Recall en $\approx 0.171$ a expensas de degradar la precisión clínica de $0.407$ a $0.082$. Se identifica el punto de operación óptimo en $r = 1.0$ (o $r = 2.0$), concluyendo que una SVM puramente lineal en MentalBERT no puede superar la cota de Recall del 20% sin generar un volumen prohibitivo de falsas alarmas para el triaje hospitalario.

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

