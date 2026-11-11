"""
Script de empaquetado para la entrega oficial en eGela.
Genera el archivo software.zip con el código fuente documentado, scripts ejecutables,
documentación científica y requerimientos, excluyendo datos clínicos bajo NDA,
cachés pesadas .npy y archivos internos de auditoría.
"""

from pathlib import Path
import zipfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent

ARCHIVOS_RAIZ = [
    "main.py",
    "pipeline.py",
    "ejecutar_pipeline.bat",
    "ejecutar_pipeline.sh",
    "ejecutar_pipeline.ps1",
    "requirements.txt",
    "README.md",
    "INFORME_CIENTIFICO_C-SSRS.md"
]

DIRECTORIOS_INCLUIDOS = [
    "src"
]

EXCLUSIONES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "AUDITORIA_DOCENTE.md",
    "poster_preview.html",
    ".DS_Store"
}

def empaquetar_software():
    ruta_zip = PROJECT_ROOT / "software.zip"
    print(f"[*] Empaquetando software entregable en: {ruta_zip.name}...")

    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Archivos en raíz
        for nombre in ARCHIVOS_RAIZ:
            ruta_archivo = PROJECT_ROOT / nombre
            if ruta_archivo.exists():
                zipf.write(ruta_archivo, arcname=nombre)
                print(f"  + [Raíz] {nombre}")

        # 2. Directorio src
        dir_src = PROJECT_ROOT / "src"
        if dir_src.exists():
            for archivo in dir_src.rglob("*"):
                if archivo.is_file():
                    # Verificar exclusiones
                    partes = archivo.relative_to(PROJECT_ROOT).parts
                    if any(exc in partes for exc in EXCLUSIONES):
                        continue
                    if archivo.suffix in [".pyc", ".pyo"]:
                        continue
                    rel_path = archivo.relative_to(PROJECT_ROOT)
                    zipf.write(archivo, arcname=str(rel_path))
                    print(f"  + [src] {rel_path}")

    tam_kb = ruta_zip.stat().st_size / 1024
    print(f"[*] Paquete creado con éxito: {ruta_zip.name} ({tam_kb:.1f} KB)")
    print(f"[*] Listo para entrega en eGela junto al póster 'poster_C-SSRS_A3.pdf'.")

if __name__ == "__main__":
    empaquetar_software()
