# PowerShell test runner for Windows
# This script does NOT change any source code or affect your macOS workflow.
# It is safe to keep in your repo and ignore on other platforms.

# Activate the virtual environment (venv)
$env:VIRTUAL_ENV = "$PSScriptRoot\venv"
$activateScript = "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
} else {
    Write-Host "Virtual environment not found. Please create it with: python -m venv venv"
    exit 1
}

# Install dependencies (idempotent)
pip install -r requirements.txt
pip install -r requirements_test.txt

# Run pytest
pytest
