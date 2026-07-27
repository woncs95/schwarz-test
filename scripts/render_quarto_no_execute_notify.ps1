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

$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "render_quarto_all_no_execute_$runStamp.log"
$statusPath = Join-Path $logDir "render_quarto_all_no_execute_status.csv"

function Write-RenderLog {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Read-DotEnvValue {
    param(
        [string]$Path,
        [string]$Key
    )

    if (-not (Test-Path $Path)) {
        return $null
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*#") {
            continue
        }
        if ($line -match "^\s*$([regex]::Escape($Key))\s*=\s*(.+?)\s*$") {
            return $matches[1].Trim('"').Trim("'")
        }
    }

    return $null
}

function Send-TelegramMessageToRegisteredChats {
    param(
        [string]$Message
    )

    $token = $env:RICECOOKER_TELEGRAM_BOT_TOKEN
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = Read-DotEnvValue `
            -Path (Join-Path $ProjectRoot ".env") `
            -Key "RICECOOKER_TELEGRAM_BOT_TOKEN"
    }
    if ([string]::IsNullOrWhiteSpace($token)) {
        Write-RenderLog "Telegram token nicht gefunden. Benachrichtigung wird übersprungen."
        return
    }

    try {
        $updatesUri = "https://api.telegram.org/bot$token/getUpdates"
        $updates = Invoke-RestMethod -Uri $updatesUri -Method Get -TimeoutSec 30

        $chatIds = @()
        foreach ($result in $updates.result) {
            if ($null -ne $result.message.chat.id) {
                $chatIds += $result.message.chat.id
            }
            if ($null -ne $result.edited_message.chat.id) {
                $chatIds += $result.edited_message.chat.id
            }
            if ($null -ne $result.channel_post.chat.id) {
                $chatIds += $result.channel_post.chat.id
            }
        }

        $chatIds = $chatIds | Sort-Object -Unique

        if (-not $chatIds -or $chatIds.Count -eq 0) {
            Write-RenderLog "Keine registrierten Telegram chat_id-Werte gefunden."
            return
        }

        foreach ($chatId in $chatIds) {
            $sendUri = "https://api.telegram.org/bot$token/sendMessage"
            $body = @{
                chat_id = $chatId
                text = $Message
            }
            Invoke-RestMethod -Uri $sendUri -Method Post -Body $body -TimeoutSec 30 | Out-Null
        }

        Write-RenderLog "Telegram-Benachrichtigung an $($chatIds.Count) Chat(s) gesendet."
    }
    catch {
        Write-RenderLog "Telegram-Benachrichtigung fehlgeschlagen: $($_.Exception.Message)"
    }
}

$notebooks = @(
    "01_eda_and_preprocessing.ipynb",
    "02_split_data.ipynb",
    "03_baseline_models.ipynb",
    "04_hyperparameter_tuning.ipynb",
    "05_final_evaluation.ipynb"
)

$statusRows = @()
$failedRows = @()

Write-RenderLog "Starte Quarto-Rendering ohne Code-Ausführung."
Write-RenderLog "ProjectRoot: $ProjectRoot"
Write-RenderLog "Force: $Force"

foreach ($notebookName in $notebooks) {
    $notebookPath = Join-Path $ProjectRoot "notebooks\$notebookName"
    $htmlName = [System.IO.Path]::ChangeExtension($notebookName, ".html")
    $htmlPath = Join-Path $ProjectRoot "reports\quarto\notebooks\$htmlName"

    if (-not (Test-Path $notebookPath)) {
        Write-RenderLog "Überspringe fehlendes Notebook: $notebookName"
        $row = [PSCustomObject]@{
            notebook = $notebookName
            status = "missing"
            exit_code = ""
            duration_seconds = 0
            output = $htmlPath
            timestamp = Get-Date -Format "s"
        }
        $statusRows += $row
        $failedRows += $row
        continue
    }

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

    $notebookStem = [System.IO.Path]::GetFileNameWithoutExtension($notebookName)
    $stdoutPath = Join-Path $logDir "quarto_${notebookStem}_${runStamp}.out.log"
    $stderrPath = Join-Path $logDir "quarto_${notebookStem}_${runStamp}.err.log"

    try {
        $renderProcess = Start-Process `
            -FilePath $quarto `
            -ArgumentList @("render", $notebookPath, "--no-execute") `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -NoNewWindow `
            -Wait `
            -PassThru

        $exitCode = $renderProcess.ExitCode

        if (Test-Path $stdoutPath) {
            Get-Content -LiteralPath $stdoutPath -ErrorAction SilentlyContinue |
                ForEach-Object { Write-RenderLog $_ }
        }
        if (Test-Path $stderrPath) {
            Get-Content -LiteralPath $stderrPath -ErrorAction SilentlyContinue |
                ForEach-Object { Write-RenderLog $_ }
        }
    }
    catch {
        $exitCode = 1
        Write-RenderLog "Exception bei ${notebookName}: $($_.Exception.Message)"
    }

    $duration = [Math]::Round(((Get-Date) - $start).TotalSeconds, 2)

    if ($exitCode -eq 0) {
        $status = "rendered"
        Write-RenderLog "Fertig: $notebookName in $duration Sekunden."
    }
    else {
        $status = "failed"
        Write-RenderLog "Fehler: $notebookName ExitCode=$exitCode."
    }

    $row = [PSCustomObject]@{
        notebook = $notebookName
        status = $status
        exit_code = $exitCode
        duration_seconds = $duration
        output = $htmlPath
        timestamp = Get-Date -Format "s"
    }

    $statusRows += $row
    if ($exitCode -ne 0) {
        $failedRows += $row
    }
}

$statusRows | Export-Csv -LiteralPath $statusPath -NoTypeInformation -Encoding UTF8
Write-RenderLog "Status gespeichert: $statusPath"

$renderedCount = ($statusRows | Where-Object { $_.status -eq "rendered" }).Count
$failedCount = ($statusRows | Where-Object { $_.status -eq "failed" -or $_.status -eq "missing" }).Count
$skippedCount = ($statusRows | Where-Object { $_.status -eq "skipped_existing" }).Count

$message = @"
Quarto Rendering beendet.

Projekt: schwarz-test
Modus: --no-execute
Gerendert: $renderedCount
Übersprungen: $skippedCount
Fehler/fehlend: $failedCount

Output:
$ProjectRoot\reports\quarto\notebooks

Log:
$logPath
"@

if ($failedCount -gt 0) {
    $failedList = ($failedRows | ForEach-Object { "$($_.notebook): $($_.status)" }) -join "`n"
    $message += "`nProblematische Dateien:`n$failedList"
}

Send-TelegramMessageToRegisteredChats -Message $message
Write-RenderLog "Rendering beendet."
