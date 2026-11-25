# ==============================================================================
# Script de ejecución de pipeline para Windows PowerShell
# Tarea: Clasificación Supervisada de Texto (Corpus Reddit C-SSRS, N=500)
# ==============================================================================
param (
    [switch]$baseline,
    [switch]$rq1,
    [switch]$rq2,
    [switch]$mejoras,
    [switch]$svm_opt
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " text-classification-MD: Ejecutor de Pipeline PowerShell" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonCmd = "python"
if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
    $pythonCmd = "python3"
    if (-not (Get-Command $pythonCmd -ErrorAction SilentlyContinue)) {
        Write-Error "No se encontro un interprete de Python en el PATH."
        exit 1
    }
}

$argsList = @()
if ($baseline) { $argsList += "--baseline" }
if ($rq1) { $argsList += "--rq1" }
if ($rq2) { $argsList += "--rq2" }
if ($mejoras) { $argsList += "--mejoras" }
if ($svm_opt) { $argsList += "--svm-opt" }

& $pythonCmd main.py @argsList

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[EXITO] Ejecucion finalizada correctamente." -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] El script finalizo con codigo de error: $LASTEXITCODE" -ForegroundColor Red
}
