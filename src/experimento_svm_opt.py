"""
Evaluación de optimizaciones exclusivas para Support Vector Machines (SVM).
Variables a testar bajo AUDITORIA_DOCENTE.md:
  1. Regularización C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
  2. class_weight: None vs 'balanced' vs ratio asimétrico óptimo r*
  3. Calibración de umbral de margen theta* sobre Dev en decision_function
  4. Normalización L2 post-concatenación en el espacio híbrido
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulo_comun_equipo import cargar_particion_oficial_equipo, RUTA_CSV_DEFECTO, limpiar_texto_reddit
from src.extraccion_mentalbert import MentalBertHierarchicalExtractor

def main():
    semillas = [42, 43, 44, 45, 46]
    extractor = MentalBertHierarchicalExtractor()
    rep_dict = extractor.obtener_o_generar_representaciones()

    df_raw = pd.read_csv(RUTA_CSV_DEFECTO)
    textos_raw = np.array(df_raw["Post"].fillna("").apply(limpiar_texto_reddit).tolist())

    particiones = {}
    for s in semillas:
        p = cargar_particion_oficial_equipo(seed=s)
        particiones[s] = {
            "tr": (p["idx"][0], p["y"][0]),
            "dev": (p["idx"][1], p["y"][1]),
            "te": (p["idx"][2], p["y"][2]),
            "clases": list(p["clases"])
        }

    clases = particiones[42]["clases"]
    idx_att = clases.index("Attempt")
    nombre_att = "Attempt"

    print("=" * 105)
    print("OPTIMIZACIÓN EXCLUSIVA PARA SUPPORT VECTOR MACHINES (SVM)")
    print("=" * 105)

    experimentos = [
        # (Nombre, emb_type, C, class_weight, l2_norm, threshold_dev)
        ("1. Baseline actual: Híbrido Truncado + C=1.0 + Simétrico", "plano_truncado", 1.0, None, False, False),
        ("2. Híbrido Truncado + C=0.1 (Regularización suave) + Simétrico", "plano_truncado", 0.1, None, False, False),
        ("3. Híbrido Truncado + C=1.0 + L2-norm híbrida", "plano_truncado", 1.0, None, True, False),
        ("4. Híbrido Truncado + C=1.0 + Umbral Dev (theta* en decision_function)", "plano_truncado", 1.0, None, False, True),
        ("5. Híbrido Jerárquico Mean-Mean + C=1.0 + Simétrico", "mean_mean", 1.0, None, False, False),
        ("6. Híbrido Jerárquico Mean-Mean + C=0.2 + Balanced", "mean_mean", 0.2, "balanced", False, False),
        ("7. Híbrido Jerárquico Mean-Mean + C=1.0 + Balanced", "mean_mean", 1.0, "balanced", False, False),
        ("8. Híbrido Jerárquico Mean-Mean + C=1.0 + Umbral Dev (theta* en margin)", "mean_mean", 1.0, None, False, True),
    ]

    for exp_nom, emb_type, c_val, cw, use_l2, use_th in experimentos:
        runs_f1 = []
        runs_rec = []
        runs_prec = []
        runs_macro = []
        runs_kap = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            emb_base = rep_dict[emb_type]
            X_emb_tr = emb_base[idx_tr]
            X_emb_dev = emb_base[idx_dev]
            X_emb_te = emb_base[idx_te]

            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
            X_tfidf_tr = vec.fit_transform(textos_raw[idx_tr])
            X_tfidf_dev = vec.transform(textos_raw[idx_dev])
            X_tfidf_te = vec.transform(textos_raw[idx_te])

            selector = SelectKBest(chi2, k=100)
            X_sel_tr = selector.fit_transform(X_tfidf_tr, y_tr == nombre_att).toarray()
            X_sel_dev = selector.transform(X_tfidf_dev).toarray()
            X_sel_te = selector.transform(X_tfidf_te).toarray()

            X_tr = np.hstack([X_emb_tr, X_sel_tr * 2.0])
            X_dev = np.hstack([X_emb_dev, X_sel_dev * 2.0])
            X_te = np.hstack([X_emb_te, X_sel_te * 2.0])

            if use_l2:
                X_tr = normalize(X_tr, norm="l2")
                X_dev = normalize(X_dev, norm="l2")
                X_te = normalize(X_te, norm="l2")

            clf = LinearSVC(C=c_val, class_weight=cw, random_state=s, max_iter=3000, dual="auto")
            clf.fit(X_tr, y_tr)

            if use_th:
                # Calibrar umbral theta en la función de margen de Dev
                dec_dev = clf.decision_function(X_dev)
                margin_att_dev = dec_dev[:, idx_att]
                y_dev_bin = (y_dev == nombre_att).astype(int)

                # Barrido sistemático de theta en el rango del margen de Dev
                thetas = np.linspace(margin_att_dev.min(), margin_att_dev.max(), 100)
                best_th = 0.0
                best_f1_d = -1.0
                for th in thetas:
                    pred_d = (margin_att_dev >= th).astype(int)
                    f1_d = f1_score(y_dev_bin, pred_d, zero_division=0)
                    if f1_d > best_f1_d:
                        best_f1_d = f1_d
                        best_th = th

                # Aplicar theta* congelado en Test
                dec_te = clf.decision_function(X_te)
                pred_te = clf.predict(X_te)
                # Si el margen de Attempt supera el umbral óptimo theta*, reclasificar
                mask_forzar = dec_te[:, idx_att] >= best_th
                pred_te[mask_forzar] = nombre_att
                y_pred = pred_te
            else:
                y_pred = clf.predict(X_te)

            f1_att = f1_score(y_te, y_pred, labels=[nombre_att], average="macro", zero_division=0)
            rec_att = recall_score(y_te, y_pred, labels=[nombre_att], average="macro", zero_division=0)
            prec_att = precision_score(y_te, y_pred, labels=[nombre_att], average="macro", zero_division=0)
            macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
            kap = cohen_kappa_score(y_te, y_pred, weights="quadratic")

            runs_f1.append(f1_att)
            runs_rec.append(rec_att)
            runs_prec.append(prec_att)
            runs_macro.append(macro_f1)
            runs_kap.append(kap)

        f1_s42 = runs_f1[0]
        f1_m, f1_s = np.mean(runs_f1), np.std(runs_f1)
        rec_m = np.mean(runs_rec)
        prec_m = np.mean(runs_prec)
        macro_m = np.mean(runs_macro)
        kap_m = np.mean(runs_kap)

        print(f"{exp_nom:<65} | F1(s42): {f1_s42:.3f} | F1(μ): {f1_m:.3f}±{f1_s:.3f} | Rec: {rec_m:.3f} | Prec: {prec_m:.3f} | Macro: {macro_m:.3f} | Kap: {kap_m:.3f}")

if __name__ == "__main__":
    main()

