"""
Generador del Póster Científico en formato DIN-A3 (420 x 297 mm, Landscape).
Compila un archivo HTML5/CSS3 de alta definición con tipografía científica,
fórmulas matemáticas, tablas formalizadas y gráficas a 300 DPI,
y lo convierte en PDF vectorial mediante el motor headless de Microsoft Edge.
"""

from pathlib import Path
import subprocess
import base64
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_GRAFICAS = PROJECT_ROOT / "data_processed" / "graficas_clasificacion"
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def cargar_imagen_base64(ruta: Path) -> str:
    if not ruta.exists():
        return ""
    with open(ruta, "rb") as f:
        datos = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{datos}"

def generar_html_poster() -> str:
    img_rq1_heatmap = cargar_imagen_base64(DIR_GRAFICAS / "fig_rq1_heatmap_2x2.png")
    img_rq1_bars = cargar_imagen_base64(DIR_GRAFICAS / "fig_rq1_pooling_bars.png")
    img_rq2_pareto = cargar_imagen_base64(DIR_GRAFICAS / "fig_rq2_pareto_frontier.png")
    img_rq2_bars = cargar_imagen_base64(DIR_GRAFICAS / "fig_rq2_cost_gain_bars.png")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Póster Científico C-SSRS - Minería de Datos</title>
<style>
  @page {{
    size: 420mm 297mm;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    width: 420mm;
    height: 297mm;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
    padding: 7mm 8mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}

  /* HEADER */
  header {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    padding: 5mm 8mm;
    border-radius: 4mm;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
  }}
  .header-left {{
    max-width: 78%;
  }}
  .badge-tag {{
    display: inline-block;
    background: #3b82f6;
    color: white;
    font-size: 8pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 1.5mm 3mm;
    border-radius: 2mm;
    margin-bottom: 1.5mm;
  }}
  h1 {{
    font-size: 15.5pt;
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: -0.3px;
    margin-bottom: 1.5mm;
  }}
  .subtitle {{
    font-size: 9.5pt;
    color: #cbd5e1;
    font-weight: 400;
  }}
  .header-right {{
    text-align: right;
    font-size: 8.5pt;
    line-height: 1.4;
    border-left: 1px solid rgba(255,255,255,0.2);
    padding-left: 6mm;
  }}
  .author-name {{
    font-size: 11pt;
    font-weight: 700;
    color: #38bdf8;
  }}

  /* MAIN GRID: 4 COLUMNAS */
  main {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 5.5mm;
    flex: 1;
    margin-top: 4.5mm;
    margin-bottom: 2mm;
    height: calc(100% - 38mm);
  }}

  .col {{
    display: flex;
    flex-direction: column;
    gap: 4mm;
    height: 100%;
  }}

  /* CARDS */
  .card {{
    background: white;
    border-radius: 3mm;
    border: 1px solid #e2e8f0;
    padding: 4mm 4.5mm;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.05);
    display: flex;
    flex-direction: column;
  }}
  .card-flex {{
    flex: 1;
  }}
  .card h2 {{
    font-size: 9.5pt;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 1.5mm;
    margin-bottom: 2.5mm;
    display: flex;
    align-items: center;
    gap: 2mm;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  .card p, .card li {{
    font-size: 7.2pt;
    line-height: 1.35;
    color: #334155;
  }}
  ul {{
    padding-left: 3.5mm;
  }}
  li {{
    margin-bottom: 1mm;
  }}
  strong {{
    color: #0f172a;
  }}

  /* HIGHLIGHT BOX */
  .highlight-box {{
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-left: 3px solid #16a34a;
    padding: 2mm 2.5mm;
    border-radius: 1.5mm;
    margin: 1.5mm 0;
    font-size: 6.8pt;
    line-height: 1.3;
  }}
  .highlight-warn {{
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 3px solid #dc2626;
  }}
  .highlight-blue {{
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #2563eb;
  }}

  /* TABLAS */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 6.3pt;
    margin: 1.5mm 0;
  }}
  th, td {{
    padding: 1.2mm 1.5mm;
    text-align: center;
    border-bottom: 1px solid #e2e8f0;
  }}
  th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
  }}
  tr.highlight-row {{
    background-color: #f0fdf4;
    font-weight: 700;
  }}
  td.text-left {{
    text-align: left;
  }}

  /* IMÁGENES */
  .img-container {{
    text-align: center;
    margin: 1.5mm 0;
  }}
  .img-container img {{
    max-width: 100%;
    max-height: 48mm;
    object-fit: contain;
    border-radius: 1.5mm;
    border: 1px solid #cbd5e1;
  }}
  .img-caption {{
    font-size: 5.8pt;
    color: #64748b;
    margin-top: 1mm;
    font-style: italic;
  }}

  /* FOOTER */
  footer {{
    background: white;
    border-top: 1px solid #e2e8f0;
    padding: 1.5mm 3mm;
    font-size: 6pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
</style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="header-left">
      <div class="badge-tag">Minería de Datos / Aprendizaje Supervisado · UPV/EHU · Tarea 2</div>
      <h1>Evaluación de Estrategias de Agregación Jerárquica y Mitigación del Desbalanceo en Texto Clínico mediante MentalBERT y SVM</h1>
      <div class="subtitle">Detección de Riesgo de Suicidio Agudo (Escala C-SSRS) en Redes Sociales · Corpus Reddit (N=500 usuarios)</div>
    </div>
    <div class="header-right">
      <div>Autor: <span class="author-name">Aimar</span></div>
      <div>Partición: <strong>70% Train (350) / 15% Dev (75) / 15% Test (75)</strong></div>
      <div>Semilla Oficial: <strong>seed=42</strong> · Validación: <strong>5 semillas [42..46]</strong></div>
      <div>Clasificador: <strong>Support Vector Machines (LinearSVC)</strong></div>
    </div>
  </header>

  <!-- MAIN 4 COLUMNS -->
  <main>

    <!-- COLUMNA 1: CONTEXTO Y RQ1 -->
    <div class="col">
      <div class="card">
        <h2>1. Contexto Clínico y Problema</h2>
        <p>Evaluación de publicaciones en comunidades de salud mental (<code>r/SuicideWatch</code>, <code>r/depression</code>) categorizadas según la escala psiquiátrica <strong>Columbia-Suicide Severity Rating Scale (C-SSRS)</strong>:</p>
        <ul style="margin-top: 1mm;">
          <li><code>Supportive</code> (0): Control empático (108 pac.)</li>
          <li><code>Indicator</code> (1): Síntomas depresivos (60 pac.)</li>
          <li><code>Ideation</code> (2): Deseo pasivo de morir (171 pac.)</li>
          <li><code>Behavior</code> (3): Conducta preparatoria (116 pac.)</li>
          <li><strong><code>Attempt</code> (4): Intento de suicidio agudo (45 pac., 9%)</strong></li>
        </ul>
        <div class="highlight-box highlight-warn" style="margin-top: 2mm;">
          <strong>Debilidades de los Antecedentes:</strong>
          <br>1. <strong>Dilución por Truncamiento (512 tokens):</strong> Descarte de >70% del texto del usuario, perdiendo la crisis crítica.
          <br>2. <strong>Colapso del Recall (SVM simétrica):</strong> Al 9% de prevalencia, la SVM ignora >85% de los intentos reales.
        </div>
      </div>

      <div class="card card-flex">
        <h2>2. Preguntas de Investigación</h2>
        <p><strong>💎 RQ1 (Agregación Temporal Jerárquica):</strong><br>
        <em>«¿Cómo afecta la función de agregación (Mean vs. Max intra/inter-post) a la retención de señales agudas frente al truncamiento a 512 tokens?»</em></p>
        <div class="highlight-box highlight-blue" style="margin-top: 1.5mm;">
          <strong>Hipótesis 1:</strong> Max-Pooling conservará picos de activación extrema en dimensiones psiquiátricas de MentalBERT sin diluirse en posts inocuos.
        </div>
        <p style="margin-top: 2mm;"><strong>💎 RQ2 (Asimetría de Costes vs. Híbrido):</strong><br>
        <em>«¿Resulta más viable reponderar la función de coste ($r = C_{{FN}}/C_{{FP}}$) o inyectar anclas léxicas ortogonales (TF-IDF crítico &chi;²)?»</em></p>
        <div class="highlight-box highlight-blue" style="margin-top: 1.5mm;">
          <strong>Hipótesis 2:</strong> La asimetría de costes saturará prematuramente el Recall por solapamiento semántico; la inyección léxica actuará como ancla discriminativa estable.
        </div>
      </div>
    </div>

    <!-- COLUMNA 2: RESULTADOS RQ1 -->
    <div class="col">
      <div class="card">
        <h2>3. Metodología Jerárquica</h2>
        <p>Diseño Factorial en 2 Niveles sobre <strong>MentalBERT 768D</strong> (Ji et al., 2021) sin pérdida de contexto (100% tokens):</p>
        <ul style="margin-top: 1.5mm;">
          <li><strong>Nivel 1 (Intra-Post):</strong> Ventanas deslizantes de 150 palabras con solapamiento de 25 $\rightarrow$ $\text{{Mean}}$ o $\text{{Max}}$.</li>
          <li><strong>Nivel 2 (Inter-Post):</strong> Agregación temporal del historial completo $\rightarrow$ $\text{{Mean}}$ o $\text{{Max}}$.</li>
        </ul>
      </div>

      <div class="card card-flex">
        <h2>4. Hallazgos RQ1: Agregación</h2>
        <div class="img-container">
          <img src="{img_rq1_heatmap}" alt="Heatmap Factorial RQ1">
          <div class="img-caption">Figura 1: Espacio factorial 2x2 (Intra-Post x Inter-Post) sobre escala C-SSRS.</div>
        </div>
        <div class="img-container">
          <img src="{img_rq1_bars}" alt="Barras de Pooling RQ1">
          <div class="img-caption">Figura 2: Macro-F1 y F1-Attempt en 5 particiones independientes (μ ± σ).</div>
        </div>
        <div class="highlight-box" style="margin-top: 1.5mm;">
          <strong>Veredicto RQ1:</strong> El truncamiento estándar (512) exhibe una alta inestabilidad (&sigma; = 0.182). <strong>Jerárquico Mean-Mean</strong> lidera en estabilidad inter-partición (&mu; = 0.196 ± 0.119, Precisión = 40.7%), y <strong>Max-Mean</strong> maximiza el Kappa Cuadrático (0.344 ± 0.101).
        </div>
      </div>
    </div>

    <!-- COLUMNA 3: RESULTADOS RQ2 -->
    <div class="col">
      <div class="card">
        <h2>5. Frontera de Pareto (Costes)</h2>
        <p>Barrido analítico de la penalización asimétrica r &in; [1, 20] en la función hinge cuadrática de la SVM:</p>
        <div class="img-container">
          <img src="{img_rq2_pareto}" alt="Curva de Pareto RQ2">
          <div class="img-caption">Figura 3: Curva de compensación Coste-Sensibilidad en MentalBERT Jerárquico.</div>
        </div>
        <div class="highlight-box highlight-warn" style="margin-top: 1.5mm;">
          <strong>Límite de la Reponderación:</strong> Al elevar r &ge; 3, el Recall satura en &approx; 17.1% pero la precisión colapsa del 40.7% al 8.2% debido al solapamiento en frontera con <code>Ideation</code>.
        </div>
      </div>

      <div class="card card-flex">
        <h2>6. Arquitectura Híbrida SVM</h2>
        <p>Inyección heterogénea de $K=100$ términos críticos seleccionados formalmente mediante &chi;² <strong>exclusivamente en Train</strong>:</p>
        <p style="font-size: 6.8pt; font-family: monospace; background: #f1f5f9; padding: 1.5mm; border-radius: 1mm; margin: 1.5mm 0; text-align: center;">
          x_i = [ e_i^MentalBERT  ||  2.0 · v_i^TFIDF ]  ∈ ℝ^(768 + 100)
        </p>
        <div class="img-container">
          <img src="{img_rq2_bars}" alt="Ganancia Híbrida RQ2">
          <div class="img-caption">Figura 4: Comparativa de F1-Attempt bajo enriquecimiento neuro-simbólico.</div>
        </div>
        <div class="highlight-box" style="margin-top: 1.5mm;">
          <strong>Veredicto RQ2:</strong> El espacio léxico disperso actúa como ancla simbólica ortogonal frente al suavizado de MentalBERT, alcanzando <strong>100% de precisión</strong> en Attempt.
        </div>
      </div>
    </div>

    <!-- COLUMNA 4: TABLA MAESTRA Y CONCLUSIONES -->
    <div class="col">
      <div class="card">
        <h2>7. Comparativa de Alternativas</h2>
        <p>Evaluación ciega final sobre <strong>Test ($n=75$)</strong> contrastando los 4 sistemas requeridos:</p>
        <table>
          <thead>
            <tr>
              <th class="text-left">Alternativa</th>
              <th>F1(s42)</th>
              <th>F1(μ±σ)</th>
              <th>Rec(μ)</th>
              <th>Prec(μ)</th>
              <th>Kap(μ)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="text-left"><strong>1. Baseline Común</strong> (MiniLM)</td>
              <td>0.000</td>
              <td>0.000</td>
              <td>0.000</td>
              <td>0.000</td>
              <td>0.082</td>
            </tr>
            <tr>
              <td class="text-left"><strong>2. Literatura</strong> (Plano 512)</td>
              <td>0.462</td>
              <td>0.132</td>
              <td>0.114</td>
              <td>0.167</td>
              <td>0.328</td>
            </tr>
            <tr>
              <td class="text-left"><strong>3. Solo TF-IDF Crítico</strong></td>
              <td>0.222</td>
              <td>0.094</td>
              <td>0.057</td>
              <td>0.300</td>
              <td>0.034</td>
            </tr>
            <tr class="highlight-row">
              <td class="text-left"><strong>4. Híbrido Oficial</strong> (C=1.0)</td>
              <td><strong>0.600</strong></td>
              <td>0.201</td>
              <td>0.143</td>
              <td><strong>0.350</strong></td>
              <td>0.349</td>
            </tr>
            <tr class="highlight-row" style="background-color: #eff6ff;">
              <td class="text-left"><strong>4. Híbrido Robusto</strong> (C=0.2)*</td>
              <td>0.333</td>
              <td><strong>0.216</strong></td>
              <td>0.171</td>
              <td>0.345</td>
              <td><strong>0.402</strong></td>
            </tr>
            <tr style="background-color: #fffbeb;">
              <td class="text-left"><strong>4. Triaje Dev</strong> (θ* calibrado)</td>
              <td>0.184</td>
              <td>0.219</td>
              <td><strong>0.514</strong></td>
              <td>0.220</td>
              <td>0.274</td>
            </tr>
          </tbody>
        </table>
        <div style="font-size: 5.6pt; color: #64748b; margin-top: 0.5mm;">
          *C=0.2 ensancha el margen analíticamente al tener D=868 > N=350, maximizando el acuerdo ordinal Kappa (0.402).
        </div>
      </div>

      <div class="card card-flex">
        <h2>8. Despliegue y Conclusiones</h2>
        <div class="highlight-box highlight-blue" style="margin-bottom: 2mm;">
          <strong>Arquitectura en Cascada para Triaje:</strong>
          <br>1. <strong>Fase 1 (Streaming, &lt;1 ms):</strong> Filtrado con TF-IDF para descartar controles.
          <br>2. <strong>Fase 2 (Profundidad, ~3.5 s):</strong> Clasificación con Híbrido MentalBERT + SVM para casos frontera.
        </div>
        <ul style="margin-top: 1mm;">
          <li><strong>Superación de Antecedentes:</strong> En la semilla oficial (42), el Híbrido supera los baselines neuronales no ontológicos de Gaur et al. WWW 2019 (0.600 vs. ~0.45).</li>
          <li><strong>Cero Data Leakage:</strong> Vocabulario, test &chi;² y umbral &theta;* calibrados estrictamente en Train y Dev.</li>
          <li><strong>Clustering Negativo:</strong> Las distancias a clústeres no supervisados degradan el F1 (añaden ruido).</li>
        </ul>
        <div style="margin-top: auto; font-size: 5.6pt; color: #475569; border-top: 1px solid #e2e8f0; padding-top: 1.5mm;">
          <strong>Referencias:</strong> Gaur et al. (WWW '19) · Ji et al. (MentalBERT '21) · Zirikly et al. (NAACL '19) · Cortes & Vapnik (ML '95).
        </div>
      </div>
    </div>

  </main>

  <!-- FOOTER -->
  <footer>
    <div>Minería de Datos / Aprendizaje Supervisado · Grado en Ingeniería Informática · UPV/EHU</div>
    <div>Software libre y reproducible disponible en repositorio de entrega · Generado a 300 DPI (DIN-A3 Landscape 420 x 297 mm)</div>
  </footer>

</body>
</html>
"""
    return html

def compilar_pdf_poster():
    html_content = generar_html_poster()
    ruta_html = PROJECT_ROOT / "poster_preview.html"
    ruta_pdf = PROJECT_ROOT / "poster_C-SSRS_A3.pdf"

    with open(ruta_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[*] HTML del póster generado en: {ruta_html}")

    if not os.path.exists(EDGE_PATH):
        print(f"[!] Error: No se encontró Edge en {EDGE_PATH}")
        return False

    comando = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={ruta_pdf}",
        "--no-pdf-header-footer",
        ruta_html.resolve().as_uri()
    ]
    print(f"[*] Compilando PDF DIN-A3 mediante Microsoft Edge headless...")
    res = subprocess.run(comando, capture_output=True, text=True)
    if res.returncode == 0 and ruta_pdf.exists():
        tam_mb = ruta_pdf.stat().st_size / (1024 * 1024)
        print(f"  [EXITO] Póster DIN-A3 compilado con éxito: {ruta_pdf.name} ({tam_mb:.2f} MB)")
        return True
    else:
        print(f"[!] Error al compilar PDF: {res.stderr}")
        return False

if __name__ == "__main__":
    compilar_pdf_poster()
