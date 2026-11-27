# Clasificación Supervisada de Riesgo Clínico en Texto (Reddit C-SSRS)

Repositorio experimental para la evaluación comparativa de representaciones jerárquicas basadas en **MentalBERT** (Ji et al., 2021), enriquecimiento léxico neuro-simbólico ($\text{TF-IDF} + \chi^2$) y clasificadores **Support Vector Machines (SVM)** sobre el corpus clínico Reddit C-SSRS ($N = 500$ usuarios, Gaur et al., 2019).

---

## 1. Marco Experimental y Acuerdos del Equipo

* **Corpus ($N=500$):** `Ideation` ($171$, $34.2\%$), `Supportive` ($108$, $21.6\%$), `Indicator` ($99$, $19.8\%$), `Behavior` ($77$, $15.4\%$) y `Attempt` ($45$, $9.0\%$).
* **Partición oficial estratificada (70 / 15 / 15):** Train ($n=350$, $31$ `Attempt`), Dev ($n=75$, $7$ `Attempt`), Test ($n=75$, $7$ `Attempt`), generada con `random_state=42` mediante `src/modulo_comun_equipo.py`.
* **Representación y Baseline común del equipo:** Embeddings densos `all-MiniLM-L6-v2` ($384\text{D}$) con Clustering Jerárquico Aglomerativo (`Ward`, $k=5$, determinista con $\sigma = 0.000$ entre repeticiones sobre una misma partición).
* **Protocolo de validación:** Todas las métricas se reportan en Test ciego tanto para la semilla oficial (`seed=42`) como en media y desviación estándar ($\mu \pm \sigma$) sobre 5 particiones estratificadas ($[42, 43, 44, 45, 46]$), midiendo tiempo de ajuste, latencia de inferencia (ms/doc) y pico de memoria RAM (MB).

---

## 2. Preguntas de Investigación (Research Questions)

### RQ1 (Agregación Temporal Jerárquica vs. Truncamiento)
> *«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto concatenado truncado a 512 tokens frente a agregación jerárquica en dos niveles con Mean/Max-Pooling) a la capacidad de una SVM lineal para identificar crisis agudas (`Attempt`) sin diluir la señal clínica?»*

* **Evidencia empírica:** Bajo la partición fijada ($N=500$), la estrategia `Jerárquico Mean-Mean` obtiene la mayor estabilidad inter-partición en `Macro-F1` ($0.327 \pm 0.035$), `F1-Attempt` ($0.196 \pm 0.119$) y `Precision-Attempt` ($0.407 \pm 0.339$), frente a la elevada varianza del truncamiento a 512 tokens (`F1-Attempt` $= 0.132 \pm 0.182$). La variante `Jerárquico Max-Mean` alcanza el mayor `Quadratic Weighted Kappa` de RQ1 ($0.344 \pm 0.101$). Sin embargo, bajo pérdida simétrica ($C=1.0$), todas las variantes muestran un `Recall-Attempt` inferior a $0.15$.

### RQ2 (Penalización Asimétrica de Costes vs. Arquitectura Híbrida Neuro-Simbólica)
> *«¿Qué mecanismo supervisado resulta más eficaz para mitigar el colapso del Recall ante el desbalanceo del $9\%$ en `Attempt`: la reponderación asimétrica de la función de coste ($r = C_{\text{FN}} / C_{\text{FP}}$) o una arquitectura híbrida neuro-simbólica (MentalBERT + TF-IDF de términos críticos $\chi^2$) combinada con calibración en Dev?»*

* **Evidencia empírica:** El barrido de costes $r \in [1.0, 20.0]$ sobre `Mean-Mean` muestra saturación del `Recall-Attempt` en $0.171 \pm 0.107$ acompañada de un descenso de la precisión de $0.407$ a $0.082$ ($r^* = 2.0$ en Dev). Por el contrario, la arquitectura híbrida neuro-simbólica ($\text{MentalBERT} + \alpha \cdot \text{TF-IDF}_{\chi^2}$ ajustado estrictamente en Train) logra:
  * **Híbrido Oficial ($C=1.0$, simétrico):** `F1-Attempt` $= 0.600$ (Precisión $= 1.000$, Recall $= 0.429$) en la semilla oficial `42`, y `F1-Attempt` $= 0.201 \pm 0.219$ en 5 particiones.
  * **Híbrido Robusto (`[Mean-Mean || Max-Mean || TF-IDF]` $\ell_2$, $C^*$ seleccionado en Dev sobre escala logarítmica, `balanced`):** Mayor acuerdo ordinal del estudio (`Quadratic Kappa` $= 0.442 \pm 0.117$), mayor `Accuracy` ($0.371 \pm 0.048$), `Weighted-F1` ($0.350 \pm 0.045$) y mínima dispersión en `F1-Attempt` ($0.214 \pm 0.058$).
  * **Híbrido Triaje Dev ($\theta^*$ calibrado en Dev con $\text{Prec}_{\text{dev}} \ge 0.20$):** `F1-Attempt` medio máximo de $0.225 \pm 0.202$ (`F1-Attempt` $= 0.600$ en `seed=42`, `Recall` $= 0.229 \pm 0.146$) sin fuga de datos desde Test.

---

## 3. Estructura del Proyecto y Ejecución

```text
text-classification-MD/
├── main.py                              # Orquestador CLI principal (Baseline, RQ1, RQ2, mejoras, tabla maestra)
├── pipeline.py                          # Carga reproducible 70/15/15 y resumen estadístico del corpus
├── ejecutar_pipeline.{ps1,sh,bat}       # Lanzadores multiplataforma
├── requirements.txt                     # Dependencias Python
├── INFORME_CIENTIFICO_C-SSRS.md         # Memoria experimental formalizada
├── data/                                # Corpus CSV y caché 384D
│   ├── 500_Reddit_users_posts_labels.csv
│   └── embeddings_384d_comun.npy
├── data_processed/                      # Embeddings MentalBERT cacheados, JSON de resultados y gráficas 300 DPI
└── src/
    ├── config.py                        # Rutas y mapeo ontológico ordinal C-SSRS
    ├── modulo_comun_equipo.py           # Partición oficial 70/15/15 (seed=42) y Baseline Ward 384D
    ├── extraccion_mentalbert.py         # Extractor jerárquico en 2 niveles (MentalBERT 768D)
    ├── metricas_clasificacion.py        # Métricas clínicas, espacio híbrido inductivo, selección en Dev y profiling
    ├── experimento_rq1.py               # Experimento factorial RQ1 y generación de figuras 300 DPI
    ├── experimento_rq2.py               # Barrido de costes, frontera de Pareto y comparativa RQ2
    ├── experimento_mejoras.py           # Extensiones SVM (L2, RBF con SVD-64, Ordinal Frank & Hall, clustering)
    └── experimento_svm_opt.py           # Ablación de regularización C, normalización L2 y umbral theta*
```

### Comandos de Ejecución

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (Baseline + RQ1 + RQ2 + Tabla Maestra + Figuras 300 DPI)
python main.py

# Ejecutar módulos individuales o estudios complementarios
python main.py --baseline
python main.py --rq1
python main.py --rq2
python main.py --mejoras
python main.py --svm-opt
```

> **Nota sobre ejecución y caché:** Colocar `500_Reddit_users_posts_labels.csv` en `data/` antes de ejecutar `main.py`. Si `data_processed/embeddings_mentalbert_*.npy` ya están presentes, el pipeline completo termina en ~90 segundos; sin ellos, la primera extracción de MentalBERT en CPU requiere ~25–30 minutos con guardado incremental por lotes.
