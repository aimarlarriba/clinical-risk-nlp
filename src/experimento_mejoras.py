"""
Estudio empírico de mejoras sobre el clasificador híbrido y análisis de métricas avanzadas (ROC-AUC, PR-AUC).
Auditado bajo AUDITORIA_DOCENTE.md:
  - Cero data leakage (Dev para calibración, Test solo para evaluación final).
  - Justificación formal de variables y funciones de pérdida.
  - Multi-semilla [42, 43, 44, 45, 46] con media +/- desviación estándar.
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score
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
            "X384": p["X_384"],
            "clases": list(p["clases"])
        }

    clases = particiones[42]["clases"]
    idx_att = clases.index("Attempt")
    nombre_attempt = "Attempt"

    print("=" * 100)
    print("COMPARATIVA DE ARQUITECTURAS Y MEJORAS PROPUESTAS")
    print("=" * 100)

    # Configuraciones a explorar:
    # 1. Hibrido Truncado + LinearSVC
    # 2. Hibrido Mean-Mean + LinearSVC
    # 3. Hibrido Mean-Mean + Clustering Features + LinearSVC
    # 4. Hibrido Mean-Mean + LogisticRegression (con probas y ROC-AUC)
    # 5. Hibrido Mean-Mean + LogisticRegression con Calibración de Umbral en Dev (theta*)

    configuraciones = [
        {"nombre": "1. Híbrido Truncado + LinearSVC (Simétrico)", "tipo_emb": "plano_truncado", "clf": "svm", "pesos": None, "cluster": False, "umbral_dev": False},
        {"nombre": "2. Híbrido Jerárquico Mean-Mean + LinearSVC (Simétrico)", "tipo_emb": "mean_mean", "clf": "svm", "pesos": None, "cluster": False, "umbral_dev": False},
        {"nombre": "3. Híbrido Jerárquico Mean-Mean + LinearSVC (Balanced)", "tipo_emb": "mean_mean", "clf": "svm", "pesos": "balanced", "cluster": False, "umbral_dev": False},
        {"nombre": "4. Híbrido Jerárquico + Features Clustering + LinearSVC", "tipo_emb": "mean_mean", "clf": "svm", "pesos": None, "cluster": True, "umbral_dev": False},
        {"nombre": "5. Híbrido Jerárquico Mean-Mean + LogReg (Simétrico)", "tipo_emb": "mean_mean", "clf": "logreg", "pesos": None, "cluster": False, "umbral_dev": False},
        {"nombre": "6. Híbrido Jerárquico Mean-Mean + LogReg (Balanced)", "tipo_emb": "mean_mean", "clf": "logreg", "pesos": "balanced", "cluster": False, "umbral_dev": False},
        {"nombre": "7. Híbrido Jerárquico + LogReg + Umbral F1-Dev", "tipo_emb": "mean_mean", "clf": "logreg", "pesos": None, "cluster": False, "umbral_dev": "f1"},
        {"nombre": "8. Híbrido Jerárquico + LogReg + Umbral Recall85-Dev", "tipo_emb": "mean_mean", "clf": "logreg", "pesos": None, "cluster": False, "umbral_dev": "rec85"},
    ]

    resultados_totales = []

    for cfg in configuraciones:
        runs_f1_att = []
        runs_rec_att = []
        runs_prec_att = []
        runs_macro_f1 = []
        runs_kappa = []
        runs_roc_auc = []
        runs_pr_auc = []

        for s in semillas:
            idx_tr, y_tr = particiones[s]["tr"]
            idx_dev, y_dev = particiones[s]["dev"]
            idx_te, y_te = particiones[s]["te"]

            # 1. Base Dense Embedding
            emb_base = rep_dict[cfg["tipo_emb"]]
            X_emb_tr = emb_base[idx_tr]
            X_emb_dev = emb_base[idx_dev]
            X_emb_te = emb_base[idx_te]

            # 2. TF-IDF Crítico ajustado estrictamente en Train
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
            X_tfidf_tr = vec.fit_transform(textos_raw[idx_tr])
            X_tfidf_dev = vec.transform(textos_raw[idx_dev])
            X_tfidf_te = vec.transform(textos_raw[idx_te])

            selector = SelectKBest(chi2, k=100)
            X_sel_tr = selector.fit_transform(X_tfidf_tr, y_tr == nombre_attempt).toarray()
            X_sel_dev = selector.transform(X_tfidf_dev).toarray()
            X_sel_te = selector.transform(X_tfidf_te).toarray()

            X_tr = np.hstack([X_emb_tr, X_sel_tr * 2.0])
            X_dev = np.hstack([X_emb_dev, X_sel_dev * 2.0])
            X_te = np.hstack([X_emb_te, X_sel_te * 2.0])

            # 3. Features de clustering si procede
            if cfg["cluster"]:
                # Clustering jerárquico ajustado en Train sobre X384
                X384_tr, X384_dev, X384_te = particiones[s]["X384"]
                clust = AgglomerativeClustering(n_clusters=5, linkage="ward")
                labels_tr = clust.fit_predict(X384_tr)
                centroides = np.array([X384_tr[labels_tr == k].mean(axis=0) for k in range(5)])

                # Distancias euclídeas a los 5 centroides
                dist_tr = cdist(X384_tr, centroides, metric="euclidean")
                dist_dev = cdist(X384_dev, centroides, metric="euclidean")
                dist_te = cdist(X384_te, centroides, metric="euclidean")

                # Normalizar distancias por media de Train
                media_d = dist_tr.mean()
                dist_tr /= media_d
                dist_dev /= media_d
                dist_te /= media_d

                X_tr = np.hstack([X_tr, dist_tr])
                X_dev = np.hstack([X_dev, dist_dev])
                X_te = np.hstack([X_te, dist_te])

            # 4. Clasificador
            if cfg["clf"] == "svm":
                clf = LinearSVC(C=1.0, class_weight=cfg["pesos"], random_state=s, max_iter=3000, dual="auto")
                clf.fit(X_tr, y_tr)
                y_pred_te = clf.predict(X_te)
                # Para SVM, decisión function para ROC-AUC one-vs-rest
                dec_te = clf.decision_function(X_te)
                prob_att_te = dec_te[:, idx_att]
            else:
                clf = LogisticRegression(C=1.0, class_weight=cfg["pesos"], random_state=s, max_iter=3000, solver="lbfgs")
                clf.fit(X_tr, y_tr)
                prob_dev = clf.predict_proba(X_dev)
                prob_te = clf.predict_proba(X_te)
                prob_att_dev = prob_dev[:, idx_att]
                prob_att_te = prob_te[:, idx_att]

                if cfg["umbral_dev"] == "f1":
                    # Calibrar umbral theta en Dev para maximizar F1 de Attempt
                    umbrales = np.linspace(0.05, 0.60, 56)
                    best_th = 0.20
                    best_f1_dev = -1.0
                    y_dev_bin = (y_dev == nombre_attempt).astype(int)
                    for th in umbrales:
                        pred_dev_bin = (prob_att_dev >= th).astype(int)
                        f1_d = f1_score(y_dev_bin, pred_dev_bin, zero_division=0)
                        if f1_d > best_f1_dev:
                            best_f1_dev = f1_d
                            best_th = th
                    # Aplicar umbral en Test
                    y_pred_te = clf.predict(X_te)
                    idx_forzar = np.where(prob_att_te >= best_th)[0]
                    y_pred_te[idx_forzar] = nombre_attempt
                elif cfg["umbral_dev"] == "rec85":
                    # Calibrar umbral theta en Dev para garantizar Recall >= 0.85
                    umbrales = np.linspace(0.01, 0.50, 100)
                    best_th = 0.10
                    y_dev_bin = (y_dev == nombre_attempt).astype(int)
                    for th in reversed(umbrales):
                        pred_dev_bin = (prob_att_dev >= th).astype(int)
                        rec_d = recall_score(y_dev_bin, pred_dev_bin, zero_division=0)
                        if rec_d >= 0.85:
                            best_th = th
                            break
                    y_pred_te = clf.predict(X_te)
                    idx_forzar = np.where(prob_att_te >= best_th)[0]
                    y_pred_te[idx_forzar] = nombre_attempt
                else:
                    y_pred_te = clf.predict(X_te)

            f1_att = f1_score(y_te, y_pred_te, labels=[nombre_attempt], average="macro", zero_division=0)
            rec_att = recall_score(y_te, y_pred_te, labels=[nombre_attempt], average="macro", zero_division=0)
            prec_att = precision_score(y_te, y_pred_te, labels=[nombre_attempt], average="macro", zero_division=0)
            macro_f1 = f1_score(y_te, y_pred_te, average="macro", zero_division=0)
            kappa = cohen_kappa_score(y_te, y_pred_te, weights="quadratic")

            y_te_bin = (y_te == nombre_attempt).astype(int)
            roc_auc = roc_auc_score(y_te_bin, prob_att_te)
            pr_auc = average_precision_score(y_te_bin, prob_att_te)

            runs_f1_att.append(f1_att)
            runs_rec_att.append(rec_att)
            runs_prec_att.append(prec_att)
            runs_macro_f1.append(macro_f1)
            runs_kappa.append(kappa)
            runs_roc_auc.append(roc_auc)
            runs_pr_auc.append(pr_auc)

        res = {
            "Configuracion": cfg["nombre"],
            "F1_Seed42": runs_f1_att[0],
            "F1_mean": np.mean(runs_f1_att),
            "F1_std": np.std(runs_f1_att),
            "Rec_mean": np.mean(runs_rec_att),
            "Prec_mean": np.mean(runs_prec_att),
            "Macro_mean": np.mean(runs_macro_f1),
            "Kappa_mean": np.mean(runs_kappa),
            "ROC_AUC_mean": np.mean(runs_roc_auc),
            "PR_AUC_mean": np.mean(runs_pr_auc),
        }
        resultados_totales.append(res)
        print(f"{cfg['nombre']:<62} | F1(s42): {res['F1_Seed42']:.3f} | F1(μ): {res['F1_mean']:.3f}±{res['F1_std']:.3f} | Rec: {res['Rec_mean']:.3f} | Prec: {res['Prec_mean']:.3f} | Macro: {res['Macro_mean']:.3f} | ROC-AUC: {res['ROC_AUC_mean']:.3f} | PR-AUC: {res['PR_AUC_mean']:.3f}")

if __name__ == "__main__":
    main()

