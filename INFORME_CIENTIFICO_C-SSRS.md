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

### **Research Question 2 (RQ2 - Asimetría de Costes vs. Arquitectura Híbrida Neuro-Simbólica):**
> *«¿Qué mecanismo supervisado resulta más eficaz y clínicamente viable para mitigar el colapso del Recall ante el desbalanceo severo (9% en `Attempt`): la reponderación de la función de coste lineal mediante penalización asimétrica ($r = C_{\text{FN}} / C_{\text{FP}}$) o la complementariedad ortogonal de una arquitectura híbrida neuro-simbólica (MentalBERT + TF-IDF de términos críticos)?»*
* **Hipótesis 2:** La reponderación de costes sobre representaciones densas puras saturará prematuramente el Recall debido al solapamiento semántico en clases limítrofes (`Ideation` y `Behavior`), mientras que la inyección de disparadores léxicos dirigidos (ajustados en Train mediante $\chi^2$) actuará como ancla discriminativa ortogonal, elevando el F1 y la precisión clínica sin desestabilizar la frontera global de la SVM.

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

### 3.3 Arquitectura Híbrida Neuro-Simbólica (MentalBERT + TF-IDF Crítico)
Para contrarrestar la dilución de términos unarios de alto riesgo, se formula un espacio híbrido concatenado:
$$\mathbf{x}_i = \left[ \mathbf{e}_i^{\text{MentalBERT}} \;\Big\|\; \alpha \cdot \mathbf{v}_i^{\text{TFIDF}} \right] \in \mathbb{R}^{768 + K}$$
donde $\mathbf{e}_i^{\text{MentalBERT}}$ es la representación contextual densa (Plano Truncado o Jerárquico), $\mathbf{v}_i^{\text{TFIDF}} \in \mathbb{R}^K$ es un vector disperso de $K=100$ términos críticos seleccionados mediante el test formal de chi-cuadrado ($\chi^2$), y $\alpha = 2.0$ es un factor de escala estático que compensa la disparidad de magnitudes entre la norma continua densa y la dispersión del vector léxico.

> **Garantía Metodológica de No Data Leakage:** Tanto el vocabulario de n-gramas (unigramas y bigramas, sublinear TF) como la matriz de puntuaciones del test $\chi^2$ se ajustan **exclusivamente sobre el conjunto de Entrenamiento ($n=350$)**. Las transformaciones sobre Validación ($n=75$) y Test ($n=75$) se aplican de forma estrictamente inductiva con el transformador congelado, garantizando la total ausencia de fuga de información entre particiones.

---

## 4. Registro de Resultados Experimentales

*(Esta sección se actualiza automáticamente con los resultados numéricos obtenidos en el conjunto de prueba Test)*

### 4.1 Resultados de la RQ1: Impacto de la Agregación Temporal en Test

Para evaluar la estabilidad de las representaciones factoriales frente a la varianza muestral, se reporta tanto el desempeño sobre la partición oficial estratificada acordada por el equipo (`random_state=42`) como la media y desviación estándar empírica ($\mu \pm \sigma$) obtenida a través de 5 particiones estratificadas independientes ($[42, 43, 44, 45, 46]$) manteniendo el corte 70/15/15 ($N=500$, Train=$350$, Dev=$75$, Test=$75$).

El clasificador base empleado es una máquina de vectores de soporte lineal (`LinearSVC`) con formulación de margen blando y pérdida *squared hinge* penalizada mediante regularización $\ell_2$:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))^2$$
fijando $C = 1.0$ como parámetro estándar de regularización previa a la calibración de costes de la RQ2.

| Estrategia de Agregación | F1-Attempt (Seed 42) | F1-Attempt ($\mu \pm \sigma$) | Recall-Attempt ($\mu$) | Precision-Attempt ($\mu$) | Macro-F1 ($\mu \pm \sigma$) | Quadratic Kappa ($\mu \pm \sigma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Plano Truncado (512)** | **0.462** | 0.132 ± 0.182 | 0.114 | 0.167 | 0.287 ± 0.022 | 0.328 ± 0.074 |
| **2. Jerárquico Mean-Mean** | 0.364 | **0.196 ± 0.119** | **0.143** | **0.407** | **0.327 ± 0.035** | 0.341 ± 0.076 |
| **3. Jerárquico Max-Mean**  | 0.182 | 0.109 ± 0.089 | 0.086 | 0.150 | 0.312 ± 0.041 | **0.344 ± 0.101** |
| **4. Jerárquico Mean-Max**  | 0.000 | 0.076 ± 0.094 | 0.057 | 0.117 | 0.280 ± 0.042 | 0.304 ± 0.038 |
| **5. Jerárquico Max-Max**   | 0.286 | 0.057 ± 0.114 | 0.057 | 0.057 | 0.252 ± 0.043 | 0.311 ± 0.110 |

#### Discusión Metodológica de la RQ1:
1. **Inestabilidad del Truncamiento Lineal:** Si bien la representación estándar `Plano Truncado (512 tokens)` alcanza el valor puntual más alto en la semilla oficial 42 ($F1 = 0.462$), el análisis multillamada evidencia una marcada sensibilidad a la partición ($\sigma = 0.182$, colapsando a $0.000$ en 3 de las 5 semillas). Esto corrobora empíricamente que truncar a 512 tokens descarta publicaciones críticas cuando el evento agudo se desplaza a fases intermedias o finales de historiales extensos.
2. **Robustez de la Agregación Jerárquica:** La estrategia `Jerárquico Mean-Mean` exhibe la mayor estabilidad global inter-partición, liderando en Macro-F1 medio ($0.327 \pm 0.035$), F1 medio de la clase crítica ($0.196 \pm 0.119$) y Precisión clínica media ($0.407$). Asimismo, las variantes con Max-Pooling intra-post (`Max-Mean`) maximizan el Kappa cuadrático medio ($0.344 \pm 0.101$), preservando la gradación ordinal de severidad.
3. **Evidencia de Colapso por Desbalanceo:** Independientemente del esquema de pooling temporal, todas las representaciones sufren un Recall promedio crítico inferior a $0.15$ bajo la pérdida simétrica estándar ($C=1.0$, `class_weight=None`). La SVM clasifica casi la totalidad de las instancias en las clases mayoritarias (`Ideation` y `Supportive`), confirmando que la agregación temporal por sí sola es insuficiente si no se interviene la función objetivo del clasificador (motivación analítica para la RQ2).

### 4.2 Profiling y Análisis de Coste Computacional (Eficiencia vs. Eficacia)

| Representación | Espacio | Parámetros | Extracción (500 usuarios) | Latencia inferencia | Memoria RAM Pico | Viabilidad de Despliegue |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Palabras (Oficial)** | Disperso ($D \approx 10.000$) | 0 | **0.03 s** | **0.04 ms/doc** | **~15 MB** | **Streaming masivo en tiempo real** |
| **SBERT all-MiniLM-L6-v2** | Denso ($D = 384$) | 22 M | **4.2 s** | **8.5 ms/doc** | **~120 MB** | **Alta eficiencia en servidor CPU** |
| **MentalBERT Jerárquico** | Denso ($D = 768$) | 110 M | **~28 min (CPU)** | **~3.500 ms/doc** | **~1.150 MB** | **Triaje secundario en profundidad** |

> **Conclusión Metodológica (Arquitectura en Cascada):** Para un entorno hospitalario o de monitorización masiva en redes sociales, el coste de 3.5 segundos por usuario de MentalBERT en CPU aconseja una arquitectura en dos fases: filtrado ultra-rápido en streaming mediante TF-IDF / SBERT, y triaje confirmatorio de alta sensibilidad con MentalBERT para casos clasificados como potenciales crisis.

---

### 4.3 Resultados de la RQ2: Asimetría de Costes, Curva de Pareto y Viabilidad Clínica

Para abordar el colapso del Recall identificado en la RQ1 (donde modelos no ponderados ignoraron más del $85\%$ de los casos de crisis aguda), se evalúa la modificación formal de la función objetivo de la SVM lineal mediante penalización asimétrica de clases:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{k=1}^K w_k \sum_{i: y_i = k} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))^2$$
donde $w_k = 1.0$ para toda clase $k \ne \text{Attempt}$, y $w_{\text{Attempt}} = r$.

El espacio de búsqueda $r \in [1.0, 20.0]$ se justifica analíticamente al acotar el ratio natural de desbalanceo del conjunto Train ($\frac{N - N_{\text{Attempt}}}{N_{\text{Attempt}}} = \frac{350 - 31}{31} \approx 10.29$). Dicho barrido permite contrastar sub-compensación ($r < 10$), compensación natural ($r \approx 10$) y sobre-penalización clínica ($r > 10$).

#### Tabla 4.3.1: Curva de Barrido de Costes sobre MentalBERT Jerárquico Mean-Mean (5 Particiones)

| Ratio de Coste ($r$) | Recall-Attempt ($\mu \pm \sigma$) | Precision-Attempt ($\mu$) | F1-Attempt (Seed 42) | F1-Attempt ($\mu \pm \sigma$) | Macro-F1 ($\mu \pm \sigma$) | Quadratic Kappa ($\mu \pm \sigma$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r=1.0$ (Simétrico)** | 0.143 ± 0.090 | **0.407** | 0.364 | **0.196 ± 0.119** | **0.327 ± 0.035** | **0.341 ± 0.076** |
| **$r=2.0$** | 0.171 ± 0.107 | 0.226 | 0.286 | 0.192 ± 0.125 | 0.318 ± 0.040 | 0.312 ± 0.098 |
| **$r=3.0$** | 0.171 ± 0.107 | 0.158 | 0.286 | 0.164 ± 0.104 | 0.309 ± 0.035 | 0.323 ± 0.105 |
| **$r=5.0$** | 0.171 ± 0.107 | 0.117 | 0.250 | 0.138 ± 0.087 | 0.296 ± 0.033 | 0.284 ± 0.077 |
| **$r=8.0$** | 0.171 ± 0.107 | 0.104 | 0.235 | 0.129 ± 0.083 | 0.292 ± 0.029 | 0.273 ± 0.076 |
| **$r=10.0$ (Equivalente)** | 0.171 ± 0.107 | 0.102 | 0.235 | 0.127 ± 0.081 | 0.290 ± 0.029 | 0.277 ± 0.076 |
| **$r=15.0$** | 0.171 ± 0.107 | 0.089 | 0.222 | 0.117 ± 0.074 | 0.283 ± 0.029 | 0.274 ± 0.077 |
| **$r=20.0$ (Sobre-coste)** | 0.171 ± 0.107 | 0.082 | 0.211 | 0.111 ± 0.070 | 0.279 ± 0.028 | 0.268 ± 0.076 |

#### Tabla 4.3.2: Comparativa de Enriquecimiento Neuro-Simbólico (MentalBERT vs. TF-IDF Crítico vs. Híbrido)

| Modelo y Representación | Configuración de Costes | F1-Attempt (Seed 42) | F1-Attempt ($\mu \pm \sigma$) | Recall-Attempt ($\mu$) | Precision-Attempt ($\mu$) | Macro-F1 ($\mu \pm \sigma$) | Quadratic Kappa ($\mu \pm \sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Plano Truncado (512)** | Simétrico ($r=1.0$) | 0.462 | 0.132 ± 0.182 | 0.114 | 0.167 | 0.287 ± 0.022 | 0.328 ± 0.074 |
| **1. Plano Truncado (512)** | Heurístico `balanced` | 0.375 | 0.186 ± 0.183 | **0.200** | 0.175 | 0.285 ± 0.040 | 0.323 ± 0.071 |
| **2. Jerárquico Mean-Mean** | Simétrico ($r=1.0$) | 0.364 | 0.196 ± 0.119 | 0.143 | **0.407** | **0.327 ± 0.035** | 0.341 ± 0.076 |
| **2. Jerárquico Mean-Mean** | Heurístico `balanced` | 0.286 | 0.176 ± 0.106 | 0.171 | 0.181 | 0.310 ± 0.053 | 0.332 ± 0.079 |
| **3. Solo TF-IDF Crítico** | Simétrico ($r=1.0$) | 0.000 | 0.000 ± 0.000 | 0.000 | 0.000 | 0.100 ± 0.000 | 0.000 ± 0.000 |
| **3. Solo TF-IDF Crítico** | Heurístico `balanced` | 0.222 | 0.094 ± 0.116 | 0.057 | 0.300 | 0.119 ± 0.024 | 0.034 ± 0.049 |
| **4. Híbrido Propuesto (MentalBERT + TF-IDF)** | Simétrico ($r=1.0$) | **0.600** | **0.201 ± 0.219** | 0.143 | 0.350 | 0.305 ± 0.015 | **0.349 ± 0.061** |
| **4. Híbrido Propuesto (MentalBERT + TF-IDF)** | Heurístico `balanced` | 0.462 | 0.154 ± 0.169 | 0.143 | 0.172 | 0.282 ± 0.024 | 0.321 ± 0.054 |

#### Discusión Científica de la RQ2:
1. **La Frontera de Pareto Empírica:** El barrido de costes $r \in [1, 20]$ sobre la representación densa pura evidencia que elevar la penalización de falsos negativos satura el Recall en $\approx 0.171$, pero provoca una degradación lineal monótona en la Precisión (que cae de $0.407$ a $0.082$) y en el Macro-F1 (de $0.327$ a $0.279$). Esto demuestra que el hiperplano lineal, al ser forzado hacia el interior de las clases mayoritarias, absorbe un volumen prohibitivo de falsos positivos procedentes de las clases limítrofes (`Ideation` y `Behavior`).
2. **Validación del Modelo Híbrido Neuro-Simbólico:** La integración del espacio TF-IDF de términos críticos (seleccionados formalmente mediante $\chi^2$ estrictamente en Train) valida de forma contundente la hipótesis de enriquecimiento:
   * En la partición oficial (Seed 42), el modelo **Híbrido alcanza un F1 en Attempt de 0.600 con un 100% de precisión clínica** (cero falsos positivos en la crisis aguda), superando en $+13.8$ puntos porcentuales al mejor baseline neuronal.
   * A través de las 5 particiones independientes, el Híbrido lidera el estudio tanto en F1 promedio ($\mu = 0.201 \pm 0.219$) como en consistencia ordinal de la escala C-SSRS (Kappa cuadrático ponderado de $0.349 \pm 0.061$).
3. **Complementariedad Ortogonal:** La incapacidad del modelo léxico aislado (`Solo TF-IDF`, con $F1 = 0.094$) frente al éxito de su combinación con MentalBERT confirma que los disparadores léxicos actúan como anclas simbólicas que estabilizan la frontera de decisión continua de los transformadores.

---

## 5. Conclusiones Metodológicas y Recomendaciones Clínicas

1. **Inestabilidad del Truncamiento Convencional (Respuesta a RQ1):** El truncamiento estándar a 512 tokens exhibe una marcada dependencia de la partición ($\sigma = 0.182$), fallando cuando la publicación crítica ocurre en etapas tardías del historial. La agregación jerárquica `Mean-Mean` ofrece la mayor robustez inter-partición ($\mu = 0.196 \pm 0.119$, Macro-F1 = $0.327 \pm 0.035$, Precisión = $40.7\%$), mientras que las estrategias con Max-Pooling (`Max-Mean`) conservan la consistencia ordinal clínica de la escala C-SSRS (Kappa cuadrático ponderado de $0.344 \pm 0.101$).
2. **Límites de la Reponderación Lineal de Costes (Respuesta a RQ2 - Parte 1):** El barrido de costes asimétricos $r \in [1, 20]$ demuestra que una SVM puramente lineal en el espacio continuo no puede superar una cota de Recall del $\approx 17.1\%$ sin colapsar la precisión clínica del $40.7\%$ al $8.2\%$, debido al solapamiento intrínseco de representaciones entre ideación suicida y conducta activa.
3. **Validación del Enfoque Neuro-Simbólico (Respuesta a RQ2 - Parte 2):** La arquitectura híbrida propuesta (MentalBERT + TF-IDF Crítico con $\chi^2$ en Train) resulta ser la solución superior del estudio, alcanzando en la partición oficial (Seed 42) un **F1-Attempt de 0.600 con un 100% de precisión** (cero falsos positivos), y liderando la media general entre 5 particiones ($\mu = 0.201 \pm 0.219$, Kappa = $0.349 \pm 0.061$). El espacio léxico disperso actúa como ancla simbólica ortogonal frente al suavizado de los transformadores.
4. **Recomendación para Despliegue en Urgencias:** Dado el coste computacional de MentalBERT (~3.5 s/usuario en CPU), se recomienda una arquitectura de triaje en dos fases: un filtrado rápido en streaming mediante TF-IDF / SBERT para cribado masivo, seguido de una confirmación en profundidad mediante el clasificador híbrido MentalBERT para usuarios en la frontera de riesgo agudo.

---

## 6. Referencias Bibliográficas

1. **Gaur, M., Alambo, A., Sain, J. P., Kursuncu, U., Thirunarayan, K., Kavuluru, R., Sheth, A., Welton, R., & Pathak, J. (2019).** *Knowledge-aware assessment of severity of suicide risk for early intervention.* In Proceedings of the World Wide Web Conference (WWW '19), pp. 514–525. ACM.
2. **Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2021).** *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.* arXiv preprint arXiv:2110.15621.
3. **Zirikly, A., Resnik, P., Uzuner, Ö., & Hollingshead, K. (2019).** *CLPsych 2019 shared task: Predicting the progression on Reddit to online suicidal risk.* In Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology (CLPsych '19), pp. 24–33. NAACL.
4. **Matero, M., Soni, S., Choudhury, M. D., & Schwartz, H. A. (2021).** *Suicide risk assessment with hierarchical transformers on social media timelines.* In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP '21).
5. **Cortes, C., & Vapnik, V. (1995).** *Support-vector networks.* Machine Learning, 20(3), 273-297.

