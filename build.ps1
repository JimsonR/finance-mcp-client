param(
    [string]$OutputDir = "dist",
    [switch]$Clean
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

try {
    if ($Clean.IsPresent -and (Test-Path $OutputDir)) {
        Write-Host "Cleaning existing build output..."
        Remove-Item $OutputDir -Recurse -Force
    }

    python -m pip install --upgrade pip build

    python -m build --wheel --outdir $OutputDir
}
finally {
    Pop-Location
}