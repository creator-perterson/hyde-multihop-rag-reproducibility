param(
    [string]$Python = "python",
    [string]$ArtifactRoot = "",
    [int]$TimeoutSeconds = 180,
    [int]$Retries = 3,
    [double]$SleepSeconds = 0.05,
    [int]$MaxParallel = 4,
    [string[]]$ModeFilter = @(
        "keyword_expansion",
        "direct_rewrite",
        "question_decomposition",
        "document_like_passage"
    ),
    [string[]]$ReaderFilter = @("qwenmax", "qwenturbo")
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $ArtifactRoot = Join-Path $RepoRoot "local-artifacts\equal_budget_query_diagnostic"
}
$CodeRoot = $RepoRoot
$Base = $ArtifactRoot
$RetrievalDir = Join-Path $Base "retrieval"
$PromptDir = Join-Path $Base "reader_prompts"
$AnswerDir = Join-Path $Base "reader_answers"
$LogDir = Join-Path $Base "logs"
$StatusFile = Join-Path $Base "cross_reader_status.txt"
$SummaryCsv = Join-Path $Base "equal_budget_cross_reader_summary.csv"
$ContrastsCsv = Join-Path $Base "equal_budget_cross_reader_contrasts.csv"

function Normalize-ProcessPath {
    $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable("PATH", "Process")
    }
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $pathValue, "Process")
}

Normalize-ProcessPath

New-Item -ItemType Directory -Force -Path $PromptDir | Out-Null
New-Item -ItemType Directory -Force -Path $AnswerDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Content -LiteralPath $StatusFile -Value "started $(Get-Date -Format s)"

$Datasets = @(
    @{
        Key = "hotpotqa"
        Name = "HotpotQA"
        Prefix = "ircot_hotpotqa_test500_equal_budget_bge_base"
    },
    @{
        Key = "2wiki"
        Name = "2WikiMultihopQA"
        Prefix = "ircot_2wiki_test500_equal_budget_bge_base"
    }
)

$Modes = @(
    "keyword_expansion",
    "direct_rewrite",
    "question_decomposition",
    "document_like_passage"
)

$Readers = @(
    @{
        Label = "qwenmax"
        Model = "qwen3.7-max"
    },
    @{
        Label = "qwenturbo"
        Model = "qwen-turbo"
    }
)

$Modes = @($Modes | Where-Object { $ModeFilter -contains $_ })
$Readers = @($Readers | Where-Object { $ReaderFilter -contains $_.Label })
if ($Modes.Count -eq 0) {
    throw "ModeFilter selected no supported query modes."
}
if ($Readers.Count -eq 0) {
    throw "ReaderFilter selected no supported readers."
}

function Count-JsonlRows {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Where-Object { $_.Trim().Length -gt 0 }).Count
}

function Build-ReaderPrompts {
    foreach ($dataset in $Datasets) {
        foreach ($mode in $Modes) {
            $retrieval = Join-Path $RetrievalDir "$($dataset.Prefix)_${mode}_top10_retrieval.jsonl"
            $prompts = Join-Path $PromptDir "$($dataset.Prefix)_${mode}_top10_extractive_prompts.jsonl"
            $rowsBefore = Count-JsonlRows $prompts
            if ($rowsBefore -ge 500) {
                Add-Content -LiteralPath $StatusFile -Value "skip_prompts $($dataset.Key) $mode rows_before=$rowsBefore $(Get-Date -Format s)"
                continue
            }
            & $Python "src\generator\build_rag_prompts.py" `
                --retrieval $retrieval `
                --out $prompts `
                --top_k 10 `
                --style extractive
            $rowsAfter = Count-JsonlRows $prompts
            Add-Content -LiteralPath $StatusFile -Value "prompts_done $($dataset.Key) $mode rows=$rowsAfter $(Get-Date -Format s)"
            if ($rowsAfter -ne 500) {
                throw "Expected 500 reader prompts for $($dataset.Key) $mode, found $rowsAfter in $prompts"
            }
        }
    }
}

function Start-ReaderProcess {
    param(
        [hashtable]$Dataset,
        [string]$Mode,
        [hashtable]$Reader,
        [string]$Prompts,
        [string]$Answers
    )
    $stdout = Join-Path $LogDir "$($Dataset.Key)_${Mode}_$($Reader.Label)_reader.stdout.log"
    $stderr = Join-Path $LogDir "$($Dataset.Key)_${Mode}_$($Reader.Label)_reader.stderr.log"
    $args = @(
        "src\generator\generate_answers_openai_compatible.py",
        "--prompts", $Prompts,
        "--out", $Answers,
        "--model", $Reader.Model,
        "--max_tokens", "64",
        "--temperature", "0.0",
        "--resume",
        "--sleep", "$SleepSeconds",
        "--timeout", "$TimeoutSeconds",
        "--retries", "$Retries"
    )
    Add-Content -LiteralPath $StatusFile -Value "launch_reader $($Dataset.Key) $Mode $($Reader.Label) rows_before=$(Count-JsonlRows $Answers) $(Get-Date -Format s)"
    return Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $CodeRoot -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
}

function Wait-ForOneReader {
    param([System.Collections.ArrayList]$Running)
    while ($true) {
        for ($i = 0; $i -lt $Running.Count; $i++) {
            $item = $Running[$i]
            $proc = $item.Proc
            $proc.Refresh()
            if ($proc.HasExited) {
                $rowsAfter = Count-JsonlRows $item.Answers
                Add-Content -LiteralPath $StatusFile -Value "complete_reader $($item.DatasetKey) $($item.Mode) $($item.ReaderLabel) rows=$rowsAfter exit=$($proc.ExitCode) $(Get-Date -Format s)"
                if ($rowsAfter -lt 500) {
                    throw "Reader process for $($item.DatasetKey) $($item.Mode) $($item.ReaderLabel) produced $rowsAfter rows. Check logs in $LogDir."
                }
                $Running.RemoveAt($i)
                return
            }
        }
        Start-Sleep -Seconds 10
    }
}

Build-ReaderPrompts

$running = [System.Collections.ArrayList]::new()
foreach ($reader in $Readers) {
    foreach ($dataset in $Datasets) {
        foreach ($mode in $Modes) {
            $prompts = Join-Path $PromptDir "$($dataset.Prefix)_${mode}_top10_extractive_prompts.jsonl"
            $answers = Join-Path $AnswerDir "$($dataset.Prefix)_${mode}_top10_answers_$($reader.Label)_500.jsonl"
            $rowsBefore = Count-JsonlRows $answers
            if ($rowsBefore -ge 500) {
                Add-Content -LiteralPath $StatusFile -Value "skip_reader $($dataset.Key) $mode $($reader.Label) rows_before=$rowsBefore $(Get-Date -Format s)"
                continue
            }
            while ($running.Count -ge $MaxParallel) {
                Wait-ForOneReader -Running $running
            }
            $proc = Start-ReaderProcess -Dataset $dataset -Mode $mode -Reader $reader -Prompts $prompts -Answers $answers
            [void]$running.Add([PSCustomObject]@{
                Proc = $proc
                DatasetKey = $dataset.Key
                Mode = $mode
                ReaderLabel = $reader.Label
                Answers = $answers
            })
        }
    }
}

while ($running.Count -gt 0) {
    Wait-ForOneReader -Running $running
}

$summaryScript = Join-Path $CodeRoot "src\evaluation\summarize_equal_budget_reader.py"
$summaryArgs = @(
    $summaryScript,
    "--answers_dir", $AnswerDir,
    "--summary_csv", $SummaryCsv,
    "--contrasts_csv", $ContrastsCsv
)
foreach ($reader in $Readers) {
    $summaryArgs += "--reader"
    $summaryArgs += "$($reader.Label)=$($reader.Model)"
}
foreach ($mode in $Modes) {
    $summaryArgs += "--mode"
    $summaryArgs += $mode
}
& $Python @summaryArgs

Add-Content -LiteralPath $StatusFile -Value "summary_done $(Get-Date -Format s)"
Add-Content -LiteralPath $StatusFile -Value "done $(Get-Date -Format s)"
