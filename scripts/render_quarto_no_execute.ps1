param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$quarto = "C:\Program Files\Quarto\bin\quarto.exe"
if (-not (Test-Path $quarto)) {
    throw "Quarto wurde nicht gefunden: $quarto"
}

$logDir = Join-Path $ProjectRoot "reports\quarto\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$logPath = Join-Path $logDir ("render_quarto_no_execute_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
$statusPath = Join-Path $logDir "render_quarto_no_execute_status.csv"

function Write-RenderLog {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Wait-OtherQuartoRender {
    param([string]$NotebookName)

    while ($true) {
        $currentProcessId = $PID
        $running = Get-CimInstance Win32_Process |
            Where-Object {
                $_.ProcessId -ne $currentProcessId -and
                $_.CommandLine -match "quarto(\.exe)?`" render" -and
                $_.CommandLine -match [regex]::Escape($NotebookName)
            }

        if (-not $running) {
            break
        }

        Write-RenderLog "Warte auf bereits laufendes Quarto-Rendering für $NotebookName."
        Start-Sleep -Seconds 10
    }
}

$notebooks = @(
    "01_eda_and_preprocessing.ipynb",
    "02_split_data.ipynb",
    "03_baseline_models.ipynb",
    "04_hyperparameter_tuning.ipynb"
)

$statusRows = @()

Write-RenderLog "Starte Quarto-Rendering ohne Code-Ausführung."
Write-RenderLog "ProjectRoot: $ProjectRoot"

foreach ($notebookName in $notebooks) {
    $notebookPath = Join-Path $ProjectRoot "notebooks\$notebookName"
    $htmlName = [System.IO.Path]::ChangeExtension($notebookName, ".html")
    $htmlPath = Join-Path $ProjectRoot "reports\quarto\notebooks\$htmlName"

    if (-not (Test-Path $notebookPath)) {
        Write-RenderLog "Überspringe fehlendes Notebook: $notebookName"
        $statusRows += [PSCustomObject]@{
            notebook = $notebookName
            status = "missing"
            exit_code = ""
            duration_seconds = 0
            output = $htmlPath
            timestamp = Get-Date -Format "s"
        }
        continue
    }

    Wait-OtherQuartoRender -NotebookName $notebookName

    if ((Test-Path $htmlPath) -and (-not $Force)) {
        Write-RenderLog "Überspringe vorhandene HTML-Datei: $htmlPath"
        $statusRows += [PSCustomObject]@{
            notebook = $notebookName
            status = "skipped_existing"
            exit_code = 0
            duration_seconds = 0
            output = $htmlPath
            timestamp = Get-Date -Format "s"
        }
        continue
    }

    Write-RenderLog "Rendere $notebookName mit --no-execute."
    $start = Get-Date

    Push-Location $ProjectRoot
    try {
        & $quarto render $notebookPath --no-execute 2>&1 |
            ForEach-Object {
                Write-RenderLog ($_ | Out-String).TrimEnd()
            }
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    $duration = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)

    if ($exitCode -eq 0) {
        Write-RenderLog "Fertig: $notebookName in $duration Sekunden."
        $status = "rendered"
    }
    else {
        Write-RenderLog "Fehler: $notebookName ExitCode=$exitCode."
        $status = "failed"
    }

    $statusRows += [PSCustomObject]@{
        notebook = $notebookName
        status = $status
        exit_code = $exitCode
        duration_seconds = $duration
        output = $htmlPath
        timestamp = Get-Date -Format "s"
    }
}

$statusRows | Export-Csv -LiteralPath $statusPath -NoTypeInformation -Encoding UTF8
Write-RenderLog "Status gespeichert: $statusPath"
Write-RenderLog "Rendering beendet."
