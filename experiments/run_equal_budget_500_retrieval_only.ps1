param(
    [string]$Python = "python",
    [string]$ArtifactRoot = "",
    [int]$TimeoutSeconds = 180,
    [int]$Retries = 3,
    [double]$SleepSeconds = 0.05,
    [int]$MaxParallel = 4
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $ArtifactRoot = Join-Path $RepoRoot "local-artifacts\equal_budget_query_diagnostic"
}
$CodeRoot = $RepoRoot
$Base = $ArtifactRoot
$PromptDir = Join-Path $Base "prompts"
$GenerationDir = Join-Path $Base "generations"
$RetrievalDir = Join-Path $Base "retrieval"
$LogDir = Join-Path $Base "logs"
$CacheDir = Join-Path $Base "cache"
$StatusFile = Join-Path $Base "equal_budget_500_status.txt"

New-Item -ItemType Directory -Force -Path $GenerationDir | Out-Null
New-Item -ItemType Directory -Force -Path $RetrievalDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Set-Content -LiteralPath $StatusFile -Value "started $(Get-Date -Format s)" -Encoding UTF8

$Datasets = @(
    @{
        Key = "hotpotqa"
        PromptPrefix = "ircot_hotpotqa_test500"
        OutPrefix = "ircot_hotpotqa_test500_equal_budget_bge_base"
    },
    @{
        Key = "2wiki"
        PromptPrefix = "ircot_2wikimultihopqa_test500"
        OutPrefix = "ircot_2wiki_test500_equal_budget_bge_base"
    }
)

$Modes = @(
    "keyword_expansion",
    "direct_rewrite",
    "question_decomposition",
    "document_like_passage"
)

function Count-JsonlRows {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $count = 0
    Get-Content -LiteralPath $Path -Encoding UTF8 | ForEach-Object {
        if ($_.Trim().Length -gt 0) {
            $count += 1
        }
    }
    return $count
}

function Start-GenerationProcess {
    param(
        [hashtable]$Dataset,
        [string]$Mode
    )
    $prompt = Join-Path $PromptDir "$($Dataset.PromptPrefix)_${Mode}_prompts.jsonl"
    $out = Join-Path $GenerationDir "$($Dataset.OutPrefix)_${Mode}_generations.jsonl"
    $stdout = Join-Path $LogDir "$($Dataset.Key)_${Mode}_generation.stdout.log"
    $stderr = Join-Path $LogDir "$($Dataset.Key)_${Mode}_generation.stderr.log"
    $args = @(
        "src\generator\generate_answers_openai_compatible.py",
        "--prompts", $prompt,
        "--out", $out,
        "--model", "qwen3.7-max",
        "--max_tokens", "64",
        "--temperature", "0.0",
        "--resume",
        "--sleep", "$SleepSeconds",
        "--timeout", "$TimeoutSeconds",
        "--retries", "$Retries",
        "--retry_sleep", "5"
    )
    Add-Content -LiteralPath $StatusFile -Value "launch $($Dataset.Key) $Mode rows_before=$(Count-JsonlRows $out) $(Get-Date -Format s)"
    return Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $CodeRoot -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
}

foreach ($dataset in $Datasets) {
    Add-Content -LiteralPath $StatusFile -Value "dataset_start $($dataset.Key) $(Get-Date -Format s)"
    $running = @()
    foreach ($mode in $Modes) {
        $out = Join-Path $GenerationDir "$($dataset.OutPrefix)_${mode}_generations.jsonl"
        $rowsBefore = Count-JsonlRows $out
        if ($rowsBefore -ge 500) {
            Add-Content -LiteralPath $StatusFile -Value "skip $($dataset.Key) $mode rows_before=$rowsBefore $(Get-Date -Format s)"
            continue
        }
        while ($running.Count -ge $MaxParallel) {
            $finished = $running | Where-Object { $_.HasExited }
            foreach ($proc in $finished) {
                if ($proc.ExitCode -ne 0) {
                    throw "Generation process failed with exit code $($proc.ExitCode). Check logs in $LogDir."
                }
                $running = @($running | Where-Object { $_.Id -ne $proc.Id })
            }
            if ($running.Count -ge $MaxParallel) {
                Start-Sleep -Seconds 10
            }
        }
        $running += Start-GenerationProcess -Dataset $dataset -Mode $mode
    }

    foreach ($proc in $running) {
        Wait-Process -Id $proc.Id
        $proc.Refresh()
        if ($proc.ExitCode -ne 0) {
            throw "Generation process failed with exit code $($proc.ExitCode). Check logs in $LogDir."
        }
    }

    foreach ($mode in $Modes) {
        $out = Join-Path $GenerationDir "$($dataset.OutPrefix)_${mode}_generations.jsonl"
        $rows = Count-JsonlRows $out
        Add-Content -LiteralPath $StatusFile -Value "complete $($dataset.Key) $mode rows=$rows $(Get-Date -Format s)"
        if ($rows -ne 500) {
            throw "Expected 500 rows for $($dataset.Key) $mode, found $rows in $out"
        }
    }
    Add-Content -LiteralPath $StatusFile -Value "dataset_done $($dataset.Key) $(Get-Date -Format s)"
}

& $Python "src\retriever\run_equal_budget_query_retrieval.py" `
    --datasets hotpotqa 2wiki `
    --generated_dir $GenerationDir `
    --results_dir $RetrievalDir `
    --summary_csv (Join-Path $Base "equal_budget_query_retrieval_summary.csv") `
    --length_csv (Join-Path $Base "equal_budget_query_length_summary.csv") `
    --cache_dir $CacheDir `
    --local_files_only `
    --doc_batch_size 64 `
    --query_batch_size 64

Add-Content -LiteralPath $StatusFile -Value "retrieval_done $(Get-Date -Format s)"
Add-Content -LiteralPath $StatusFile -Value "done $(Get-Date -Format s)"
