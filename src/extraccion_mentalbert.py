"""
Paso 1: Extractor Jerárquico de MentalBERT (768D) en 2 Niveles.
Estructura de agregación para responder a RQ1:
  Nivel 1 (Intra-Post): Combina fragmentos de posts largos (>150 palabras) mediante Mean o Max.
  Nivel 2 (Inter-Post): Combina la línea temporal de posts de cada usuario mediante Mean o Max.

Genera y cachea en disco las 4 representaciones factoriales (500 x 768):
  1. emb_mean_mean: Mean(Post) -> Mean(Usuario)  [Puro promedio clásico]
  2. emb_max_mean:  Max(Post)  -> Mean(Usuario)  [Pico intra-post, promedio temporal]
  3. emb_mean_max:  Mean(Post) -> Max(Usuario)   [Promedio intra-post, rescate de post crítico]
  4. emb_max_max:   Max(Post)  -> Max(Usuario)   [Puro valor extremo: mayor señal en todo el historial]
"""
from pathlib import Path
import os
import sys
import ast
import re
import time
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src.modulo_comun_equipo import RUTA_CSV_DEFECTO

DIR_DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data_processed"
CACHE_FILES = {
    "plano_truncado": DIR_DATA_PROCESSED / "embeddings_mentalbert_plano_truncado.npy",
    "mean_mean": DIR_DATA_PROCESSED / "embeddings_mentalbert_mean_mean.npy",
    "max_mean":  DIR_DATA_PROCESSED / "embeddings_mentalbert_max_mean.npy",
    "mean_max":  DIR_DATA_PROCESSED / "embeddings_mentalbert_mean_max.npy",
    "max_max":   DIR_DATA_PROCESSED / "embeddings_mentalbert_max_max.npy",
}


def extraer_lista_posts(raw_val: str) -> List[str]:
    """Extrae la lista de posts individuales de la cadena serializada del CSV."""
    s = str(raw_val).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lista = ast.literal_eval(s)
            if isinstance(lista, list) and len(lista) > 0:
                posts_limpios = [str(p).strip() for p in lista if len(str(p).strip()) > 10]
                if posts_limpios:
                    return posts_limpios
        except Exception:
            pass
    partes = re.split(r"'\s*,\s*'|\"\s*,\s*\"", s.strip("[]"))
    limpios = [p.strip("'\" \n\t\r") for p in partes if len(p.strip("'\" \n\t\r")) > 10]
    return limpios if limpios else [s]


def trocear_en_ventanas(texto: str, max_palabras: int = 150, solapamiento: int = 25) -> List[str]:
    """
    Divide posts largos en ventanas con solapamiento para no truncar ningún token.
    Si el post mide <= 150 palabras se devuelve íntegro como 1 fragmento.
    """
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return [texto]
    
    ventanas = []
    paso = max_palabras - solapamiento
    for i in range(0, len(palabras), paso):
        trozo = " ".join(palabras[i:i + max_palabras])
        ventanas.append(trozo)
        if i + max_palabras >= len(palabras):
            break
    return ventanas


class MentalBertHierarchicalExtractor:
    def __init__(self, model_name: str = "mental/mental-bert-base-uncased"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _cargar_modelo(self):
        if self.tokenizer is None or self.model is None:
            print(f"[*] Cargando {self.model_name} en dispositivo '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    def encodear_lote_fragmentos(
        self,
        textos: List[str],
        batch_size: int = 64,
        max_len: int = 192,
        desc: str = "Fragmentos",
        guardar_cada_n_lotes: int = 5,
        ruta_checkpoint: Path = None,
    ) -> np.ndarray:
        """
        Codifica fragmentos y extrae el vector medio con máscara de atención (768D).
        Incluye guardado automático en disco cada `guardar_cada_n_lotes` lotes (por defecto 5)
        y capacidad de reanudar automáticamente si el proceso se interrumpió.
        """
        self._cargar_modelo()
        total_textos = len(textos)
        total_lotes = (total_textos + batch_size - 1) // batch_size
        t_inicio = time.time()

        embs_list = []
        offset_textos = 0

        # 1. Comprobar si existe un checkpoint previo para reanudar sin repetir trabajo
        if ruta_checkpoint and Path(ruta_checkpoint).exists():
            try:
                embs_previos = np.load(ruta_checkpoint)
                n_prev = len(embs_previos)
                if n_prev >= total_textos:
                    print(f"[*] Checkpoint completo detectado en {Path(ruta_checkpoint).name}: {n_prev} textos listos.", flush=True)
                    return embs_previos[:total_textos]
                elif n_prev > 0:
                    offset_textos = n_prev
                    embs_list.append(embs_previos)
                    lote_inicio = (offset_textos // batch_size) + 1
                    print(
                        f"[*] Reanudando [{desc}] desde fragmento {offset_textos}/{total_textos} (Lote {lote_inicio}/{total_lotes}) "
                        f"cargado desde {Path(ruta_checkpoint).name}",
                        flush=True
                    )
            except Exception as e:
                print(f"[!] Advertencia al leer checkpoint {ruta_checkpoint}: {e}. Iniciando desde el principio.", flush=True)

        textos_pendientes = total_textos - offset_textos
        print(f"\n[*] Iniciando codificación [{desc}]: {textos_pendientes} fragmentos pendientes de {total_textos} en lotes (batch_size={batch_size})...", flush=True)

        lote_inicial_idx = (offset_textos // batch_size) + 1

        with torch.no_grad():
            for idx_rel, i in enumerate(range(offset_textos, total_textos, batch_size), start=1):
                idx_lote = (i // batch_size) + 1
                lote = textos[i:i + batch_size]
                tokens = self.tokenizer(
                    lote, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
                ).to(self.device)
                
                out = self.model(**tokens)
                mask = tokens["attention_mask"].unsqueeze(-1).expand(out[0].size()).float()
                sum_emb = torch.sum(out[0] * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_emb / sum_mask
                embs_list.append(mean_pooled.cpu().numpy())

                # Verbose de telemetría en tiempo real
                t_transcurrido = time.time() - t_inicio
                seg_por_lote = t_transcurrido / max(idx_rel, 1)
                lotes_restantes = total_lotes - idx_lote
                min_restantes = (seg_por_lote * lotes_restantes) / 60.0
                textos_procesados = min(i + batch_size, total_textos)
                porcentaje = (textos_procesados / total_textos) * 100.0
                velocidad = (textos_procesados - offset_textos) / max(t_transcurrido, 1e-6)

                if idx_lote % 2 == 0 or idx_lote == total_lotes:
                    print(
                        f"  [{desc}] Lote {idx_lote:>3}/{total_lotes} ({porcentaje:>5.1f}%) | "
                        f"{textos_procesados:>5}/{total_textos} textos | "
                        f"Velocidad: {velocidad:.1f} txt/s | "
                        f"Transcurrido: {t_transcurrido/60.0:.1f}m | Restante estimado: ~{min_restantes:.1f}m",
                        flush=True
                    )

                # Guardar checkpoint cada guardar_cada_n_lotes (5 por defecto)
                if ruta_checkpoint and (idx_rel % guardar_cada_n_lotes == 0 or idx_lote == total_lotes):
                    embs_consolidados = np.vstack(embs_list)
                    np.save(ruta_checkpoint, embs_consolidados)
                    print(
                        f"    -> [CHECKPOINT CADA {guardar_cada_n_lotes} LOTES] Guardados {len(embs_consolidados)}/{total_textos} fragmentos en {Path(ruta_checkpoint).name}",
                        flush=True
                    )
                    embs_list = [embs_consolidados]  # mantener lista compacta

        embs_finales = np.vstack(embs_list) if embs_list else np.empty((0, 768), dtype=np.float32)
        if ruta_checkpoint:
            np.save(ruta_checkpoint, embs_finales)
        t_total = time.time() - t_inicio
        print(f"[OK] Codificación [{desc}] completada en {t_total/60.0:.2f} minutos.\n", flush=True)
        return embs_finales

    def obtener_o_generar_representaciones(self, ruta_csv: str = RUTA_CSV_DEFECTO) -> Dict[str, np.ndarray]:
        """
        Devuelve un diccionario con las 4 representaciones jerárquicas:
        {'mean_mean', 'max_mean', 'mean_max', 'max_max'}, cada una de tamaño (500, 768).
        """
        DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        # Comprobar si las 4 ya están cacheadas
        if all(ruta.exists() for ruta in CACHE_FILES.values()):
            print("[*] Cargando las 4 representaciones de MentalBERT desde la caché local...", flush=True)
            for nombre, ruta in CACHE_FILES.items():
                print(f"  [OK] Cargado {nombre:<15}: {ruta.name}", flush=True)
            return {nombre: np.load(ruta) for nombre, ruta in CACHE_FILES.items()}

        df = pd.read_csv(ruta_csv)
        print(f"[*] Iniciando extracción jerárquica para {len(df)} usuarios...", flush=True)

        # 1. Parsear usuarios -> posts -> fragmentos guardando el árbol de índices
        todos_los_fragmentos = []
        estructura_usuarios = []  # Para cada usuario: lista de tuplas (idx_inicio_post, idx_fin_post)
        
        cursor_global = 0
        for posts_raw in df["Post"]:
            posts = extraer_lista_posts(posts_raw)
            posts_de_este_usuario = []
            
            for post in posts:
                frags = trocear_en_ventanas(post, max_palabras=150, solapamiento=25)
                n_frags = len(frags)
                todos_los_fragmentos.extend(frags)
                posts_de_este_usuario.append((cursor_global, cursor_global + n_frags))
                cursor_global += n_frags
                
            estructura_usuarios.append(posts_de_este_usuario)

        print(f"[*] Total fragmentos a procesar con MentalBERT (cobertura 100% tokens): {len(todos_los_fragmentos)}", flush=True)
        
        # 2. Codificación por lotes con checkpointing automático cada 5 lotes
        ruta_ckpt_frags = DIR_DATA_PROCESSED / "checkpoint_fragmentos_mentalbert.npy"
        embs_fragmentos = self.encodear_lote_fragmentos(
            todos_los_fragmentos,
            batch_size=64,
            max_len=192,
            desc="Fragmentos Intra-Post",
            guardar_cada_n_lotes=5,
            ruta_checkpoint=ruta_ckpt_frags
        )
        print("[*] Codificación de fragmentos completada. Realizando agregación jerárquica en 2 niveles...", flush=True)

        # 3. Agregación en 2 niveles (Fragmentos -> Post -> Usuario)
        res_mean_mean, res_max_mean, res_mean_max, res_max_max = [], [], [], []

        for idx_u, posts_usuario in enumerate(estructura_usuarios):
            posts_vec_mean = []
            posts_vec_max = []

            for ini, fin in posts_usuario:
                frags_post = embs_fragmentos[ini:fin]
                # Nivel 1: Intra-Post
                posts_vec_mean.append(np.mean(frags_post, axis=0))
                posts_vec_max.append(np.max(frags_post, axis=0))

            posts_arr_mean = np.array(posts_vec_mean, dtype=np.float32)
            posts_arr_max = np.array(posts_vec_max, dtype=np.float32)

            # Nivel 2: Inter-Post (a lo largo del historial del usuario)
            res_mean_mean.append(np.mean(posts_arr_mean, axis=0))
            res_max_mean.append(np.mean(posts_arr_max, axis=0))
            res_mean_max.append(np.max(posts_arr_mean, axis=0))
            res_max_max.append(np.max(posts_arr_max, axis=0))

            if (idx_u + 1) % 50 == 0 or idx_u + 1 == len(estructura_usuarios):
                print(f"  [Agregación] Usuario {idx_u + 1}/{len(estructura_usuarios)} procesado ({(idx_u+1)/len(estructura_usuarios)*100:.1f}%)", flush=True)

        # Guardar inmediatamente las 4 representaciones jerárquicas
        arr_mean_mean = np.array(res_mean_mean, dtype=np.float32)
        arr_max_mean  = np.array(res_max_mean, dtype=np.float32)
        arr_mean_max  = np.array(res_mean_max, dtype=np.float32)
        arr_max_max   = np.array(res_max_max, dtype=np.float32)

        np.save(CACHE_FILES["mean_mean"], arr_mean_mean)
        np.save(CACHE_FILES["max_mean"], arr_max_mean)
        np.save(CACHE_FILES["mean_max"], arr_mean_max)
        np.save(CACHE_FILES["max_max"], arr_max_max)
        print("[OK] Guardadas las 4 representaciones jerárquicas en disco.", flush=True)

        # 4. Texto completo tradicional concatenado (truncado a 512 tokens)
        if CACHE_FILES["plano_truncado"].exists():
            print(f"[*] Cargando plano truncado desde caché: {CACHE_FILES['plano_truncado'].name}...", flush=True)
            arr_plano_truncado = np.load(CACHE_FILES["plano_truncado"])
        else:
            print("[*] Codificando texto completo concatenado tradicional (truncado a 512 tokens)...", flush=True)
            from src.modulo_comun_equipo import limpiar_texto_reddit
            textos_completos = df["Post"].fillna("").apply(limpiar_texto_reddit).tolist()
            ruta_ckpt_plano = DIR_DATA_PROCESSED / "checkpoint_plano_truncado_parcial.npy"
            res_plano_truncado = self.encodear_lote_fragmentos(
                textos_completos,
                batch_size=32,
                max_len=512,
                desc="Plano Truncado (512 tokens)",
                guardar_cada_n_lotes=5,
                ruta_checkpoint=ruta_ckpt_plano
            )
            arr_plano_truncado = np.array(res_plano_truncado, dtype=np.float32)
            np.save(CACHE_FILES["plano_truncado"], arr_plano_truncado)
            print(f"[OK] Guardado {CACHE_FILES['plano_truncado'].name}", flush=True)

        resultados = {
            "plano_truncado": arr_plano_truncado,
            "mean_mean": arr_mean_mean,
            "max_mean":  arr_max_mean,
            "mean_max":  arr_mean_max,
            "max_max":   arr_max_max,
        }

        print(f"[*] Todas las representaciones factoriales están disponibles en {DIR_DATA_PROCESSED}", flush=True)

        return resultados


if __name__ == "__main__":
    extractor = MentalBertHierarchicalExtractor()
    datos = extractor.obtener_o_generar_representaciones()
    for k, v in datos.items():
        print(f"  - {k:<10}: shape {v.shape}")
