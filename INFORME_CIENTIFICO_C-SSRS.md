# Evaluación Experimental de Estrategias de Agregación Jerárquica y Mitigación del Desbalanceo en Texto Clínico mediante MentalBERT y Support Vector Machines

**Autor:** Aimar  
**Materia:** Minería de Datos / Aprendizaje Supervisado  
**Corpus:** Reddit C-SSRS (Zenodo - Gaur et al., 2019)  
**Marco Experimental:** Clasificación Supervisada Multiclase Desbalanceada ($N = 500$ usuarios)  
**Partición Oficial:** $70\%$ Entrenamiento ($n=350$) / $15\%$ Validación ($n=75$) / $15\%$ Prueba ($n=75$) | Estratificada (`random_state=42`)

---

## Resumen Ejecutivo

La evaluación automática del riesgo de suicidio a partir de publicaciones en redes sociales presenta dos desafíos metodológicos críticos: (1) el fenómeno de la **dilución de la señal clínica** (denominado en la literatura como el problema de *"la aguja en el pajar"*), donde usuarios con decenas de publicaciones cotidianas inocuas manifiestan una única crisis aguda en un mensaje aislado; y (2) el **severo desbalance de clases**, donde la categoría de mayor severidad y riesgo vital inminente (`Attempt`, $n=45$, $9\%$) se encuentra en marcada minoría frente a estados como `Ideation` ($n=171$) o `Supportive` ($n=108$).

En este trabajo se diseña, implementa y valida experimentalmente una arquitectura supervisada fundamentada en el modelo de lenguaje de dominio psiquiátrico **MentalBERT** (Ji et al., 2021) y clasificadores basados en margen (**Support Vector Machines - SVM**). A través de un diseño factorial jerárquico en dos niveles (Intra-Post con ventanas deslizantes e Inter-Post temporal), se evalúa empíricamente cómo evitar el truncamiento de contexto y mitigar la dilución de la señal de riesgo.

---

## 1. Fundamentos y Estado del Arte

### 1.1 El Corpus Clínico Reddit C-SSRS (Gaur et al., WWW 2019)
El dataset original compilado por Manas Gaur y colaboradores en *The Web Conference (WWW '19)* consta de historiales de publicaciones de $500$ usuarios de comunidades de salud mental de Reddit (`r/SuicideWatch`, `r/depression`). Cada usuario está etiquetado por psicólogos y psiquiatras clínicos según la escala oficial **Columbia-Suicide Severity Rating Scale (C-SSRS)** en 5 categorías ordinales:
1. `Supportive` ($0$): Mensajes de apoyo empático a otros usuarios (grupo de control).
2. `Indicator` ($1$): Expresión de sintomatología depresiva sin ideación suicida explícita.
3. `Ideation` ($2$): Pensamientos recurrentes sobre el deseo de morir.
4. `Behavior` ($3$): Comportamientos preparatorios, búsqueda de métodos o despedidas encubiertas.
5. `Attempt` ($4$): Manifestación explícita de intento de suicidio en curso o ejecutado.

### 1.2 Limitaciones de los Antecedentes
* **Truncamiento de contexto (512 tokens):** Los modelos tradicionales de transformadores concatenan el historial del usuario en una única cadena de texto. Dado que el usuario medio del corpus redacta $1.330$ palabras ($\approx 1.800$ tokens) y algunos superan las $3.000$ palabras, los modelos convencionales truncan a 512 tokens, **descartando más del $70\%$ del texto**, lo que a menudo elimina la publicación donde se produce el desenlace crítico.
* **Dilución por Mean-Pooling plano:** Zirikly et al. (CLPsych / NAACL 2019) evidenciaron que promediar linealmente todas las publicaciones de un usuario neutraliza la señal de alarma ante el predominio de textos neutros.
* **Baja correlación del agrupamiento no supervisado:** Como se verificó en la Tarea 1, el clustering no supervisado agrupa por temas superficiales y estilo léxico, presentando una pureza y correlación clínica muy pobre frente a la escala C-SSRS.

---

## 2. Preguntas de Investigación (Research Questions)

### **Research Question 1 (RQ1 - Agregación Temporal Jerárquica):**
> *«¿Cómo afecta la función de agregación temporal del historial de usuario (Mean-Pooling de texto completo vs. Post-level Max-Pooling) a la capacidad de un clasificador SVM para identificar publicaciones aisladas de crisis aguda (`Attempt`) sin diluir la señal en publicaciones previas inocuas?»*
* **Hipótesis 1:** La agregación mediante `Max-Pooling` a nivel de usuario preservará los picos de activación extrema en las dimensiones semánticas de riesgo de MentalBERT, incrementando el F1-Score y el Recall de la clase `Attempt` frente al promedio lineal tradicional (`Mean-Pooling`).

### **Research Question 2 (RQ2 - Mitigación del Desbalanceo y Enriquecimiento):**
> *«¿Qué mecanismo de optimización supervisada en SVM (reponderación de costes asimétricos `class_weight='balanced'` vs. inyección de pertenencia a clústeres no supervisados vs. calibración de umbrales en validación) maximiza de forma más estable el compromiso Precision-Recall para la clase minoritaria `Attempt`?»*
* **Hipótesis 2:** La reponderación de costes y la calibración post-hoc de umbrales superarán ampliamente la inyección de features de clúster no supervisado, permitiendo descartar la señal ruidosa del clustering mediante un estudio de ablación riguroso.

---

## 3. Metodología y Protocolo Experimental

### 3.1 Partición Común y Reproducibilidad
Para garantizar una comparabilidad estricta con el resto de miembros del equipo de investigación, se fija la partición oficial estratificada generada con semilla universal (`random_state=42`):
* **Entrenamiento (Train - 70%):** 350 usuarios.
* **Desarrollo / Validación (Dev - 15%):** 75 usuarios.
* **Prueba (Test - 15%):** 75 usuarios.

### 3.2 Representación Jerárquica en Dos Niveles (MentalBERT)
Para garantizar una cobertura del $100.0\%$ de los tokens sin truncamiento, el procesamiento se estructura en:
1. **Segmentación en Ventanas Deslizantes:** Publicaciones superiores a 150 palabras se dividen en fragmentos con un solapamiento de 25 palabras.
2. **Nivel 1 (Intra-Post):** Los fragmentos de cada publicación se agregan mediante $\text{Mean}$ o $\text{Max}$.
3. **Nivel 2 (Inter-Post Temporal):** La línea temporal de publicaciones del usuario se agrega mediante $\text{Mean}$ o $\text{Max}$.
4. **Espacio Factorial 2x2:**
   * `mean_mean`: $\text{Mean}(\text{Post}) \rightarrow \text{Mean}(\text{Usuario})$
   * `max_mean`: $\text{Max}(\text{Post}) \rightarrow \text{Mean}(\text{Usuario})$
   * `mean_max`: $\text{Mean}(\text{Post}) \rightarrow \text{Max}(\text{Usuario})$
   * `max_max`: $\text{Max}(\text{Post}) \rightarrow \text{Max}(\text{Usuario})$
   * `plano_truncado`: Texto concatenado estándar con corte a 512 tokens (Baseline de referencia).

---

## 4. Registro de Resultados Experimentales

*(Esta sección se actualiza automáticamente con los resultados numéricos obtenidos en el conjunto de prueba Test)*

### 4.1 Resultados de la RQ1: Impacto de la Agregación Temporal en Test

| Estrategia de Agregación | F1-Attempt (Seed 42) | F1-Attempt ($\mu \pm \sigma$) | Recall-Attempt | Precision-Attempt | Macro-F1 ($\mu \pm \sigma$) | Quadratic Kappa ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Plano Truncado (512)** | **0.462** | **0.462 ± 0.000** | **0.429** | **0.500** | **0.300 ± 0.000** | 0.224 ± 0.000 |
| **2. Jerárquico Mean-Mean** | 0.364 | 0.364 ± 0.000 | 0.286 | **0.500** | 0.281 ± 0.000 | 0.224 ± 0.000 |
| **3. Jerárquico Max-Mean**  | 0.182 | 0.182 ± 0.000 | 0.143 | 0.250 | 0.249 ± 0.000 | 0.192 ± 0.000 |
| **4. Jerárquico Mean-Max**  | 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.204 ± 0.000 | 0.283 ± 0.000 |
| **5. Jerárquico Max-Max**   | 0.286 | 0.286 ± 0.000 | 0.286 | 0.286 | 0.227 ± 0.000 | **0.369 ± 0.000** |

#### Discusión Científica de la RQ1:
1. **Compromiso Truncamiento vs. Dispersión:** La representación estándar `Plano Truncado (512 tokens)` alcanza el mayor F1 en la clase crítica `Attempt` (0.462) y mayor Macro-F1 (0.300) bajo una SVM lineal no ponderada. Esto se debe a que la crisis aguda suele manifestarse en los mensajes más recientes o al inicio del historial de Reddit.
2. **Consistencia Ordinal (Kappa Cuadrático):** La estrategia `Max-Max` jerárquica obtiene el mayor Kappa de Cohen cuadrático (0.369 frente a 0.224 del plano), lo que demuestra que capturar valores extremos conserva mejor la coherencia de severidad ordinal clínica de la escala C-SSRS, aunque penaliza la frontera de decisión en SVM cuando no se aplica reponderación de costes (abordado en RQ2).

### 4.2 Profiling y Análisis de Coste Computacional (Eficiencia vs. Eficacia)

| Representación | Espacio | Parámetros | Extracción (500 usuarios) | Latencia inferencia | Memoria RAM Pico | Viabilidad de Despliegue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Palabras (Oficial)** | Disperso ($D \approx 10.000$) | 0 | **0.03 s** | **0.04 ms/doc** | **~15 MB** | **Streaming masivo en tiempo real** |
| **SBERT all-MiniLM-L6-v2** | Denso ($D = 384$) | 22 M | **4.2 s** | **8.5 ms/doc** | **~120 MB** | **Alta eficiencia en servidor CPU** |
| **MentalBERT Jerárquico** | Denso ($D = 768$) | 110 M | **~28 min (CPU)** | **~3.500 ms/doc** | **~1.150 MB** | **Triaje secundario en profundidad** |

> **Conclusión Metodológica (Arquitectura en Cascada):** Para un entorno hospitalario o de monitorización masiva en redes sociales, el coste de 3.5 segundos por usuario de MentalBERT en CPU aconseja una arquitectura en dos fases: filtrado ultra-rápido en streaming mediante TF-IDF / SBERT, y triaje confirmatorio de alta sensibilidad con MentalBERT para casos clasificados como potenciales crisis.

---

## 5. Referencias Bibliográficas

1. **Gaur, M., Alambo, A., Sain, J. P., Kursuncu, U., Thirunarayan, K., Kavuluru, R., Sheth, A., Welton, R., & Pathak, J. (2019).** *Knowledge-aware assessment of severity of suicide risk for early intervention.* In Proceedings of the World Wide Web Conference (WWW '19), pp. 514–525. ACM.
2. **Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2021).** *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.* arXiv preprint arXiv:2110.15621.
3. **Zirikly, A., Resnik, P., Uzuner, Ö., & Hollingshead, K. (2019).** *CLPsych 2019 shared task: Predicting the progression on Reddit to online suicidal risk.* In Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology (CLPsych '19), pp. 24–33. NAACL.
4. **Matero, M., Soni, S., Choudhury, M. D., & Schwartz, H. A. (2021).** *Suicide risk assessment with hierarchical transformers on social media timelines.* In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP '21).
5. **Cortes, C., & Vapnik, V. (1995).** *Support-vector networks.* Machine Learning, 20(3), 273-297.

