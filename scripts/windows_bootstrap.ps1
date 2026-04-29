Param(
    [switch]$InstallWSL
)

$ErrorActionPreference = "Stop"

Write-Host "== YOLO-SLAM Windows Bootstrap =="
Write-Host "Project root: $PSScriptRoot\.."

if ($InstallWSL) {
    Write-Host "Installing WSL + Ubuntu 22.04..."
    wsl --install -d Ubuntu-22.04
    Write-Host "If prompted, reboot Windows and re-run this script without -InstallWSL."
    exit 0
}

Write-Host "Checking local Python..."
python --version

Write-Host "Running local smoke checks..."
python "$PSScriptRoot\..\datasets\generate_dataset.py" --num-images 100
python "$PSScriptRoot\..\src\evaluation\run_trials.py" --num-trials 30
python "$PSScriptRoot\..\src\evaluation\plot_results.py" --results-dir "$PSScriptRoot\..\results"

Write-Host "Done. Local scaffold is healthy."
Write-Host "Next: run scripts/wsl_bootstrap.sh inside Ubuntu (WSL)."
