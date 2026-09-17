param(
    [ValidateSet("cuda121", "cpu")]
    [string]$Backend = "cuda121",
    [string]$EnvironmentName = "face-preprocess",
    [string]$EnvironmentPath,
    [string]$CacheRoot,
    [string]$CondaExecutable
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($CondaExecutable) {
    if (-not [IO.Path]::IsPathRooted($CondaExecutable)) {
        throw "CondaExecutable must be absolute."
    }
    $CondaExe = [IO.Path]::GetFullPath($CondaExecutable)
    if (-not (Test-Path -LiteralPath $CondaExe -PathType Leaf)) {
        throw "CondaExecutable does not exist: $CondaExe"
    }
} else {
    $CondaCommand = Get-Command mamba -ErrorAction SilentlyContinue
    if (-not $CondaCommand) {
        $CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
    }
    if (-not $CondaCommand) {
        $CondaCandidates = @(
            (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
            (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
            (Join-Path $env:ProgramData "miniconda3\Scripts\conda.exe"),
            (Join-Path $env:ProgramData "anaconda3\Scripts\conda.exe")
        )
        $CondaPath = $CondaCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($CondaPath) {
            $CondaCommand = Get-Item -LiteralPath $CondaPath
        }
    }
    if (-not $CondaCommand) {
        throw "Conda or Mamba is required."
    }
    $CondaExe = if ($CondaCommand.Source) { $CondaCommand.Source } else { $CondaCommand.FullName }
}

if ($EnvironmentPath) {
    if (-not [IO.Path]::IsPathRooted($EnvironmentPath)) {
        throw "EnvironmentPath must be absolute."
    }
    $EnvironmentPath = [IO.Path]::GetFullPath($EnvironmentPath)
    $EnvironmentArguments = @("-p", $EnvironmentPath)
    $EnvironmentDisplay = $EnvironmentPath
    if (-not $CacheRoot) {
        $CacheRoot = Join-Path (Split-Path $EnvironmentPath -Parent) ".face-preprocess-cache"
    }
} else {
    $EnvironmentArguments = @("-n", $EnvironmentName)
    $EnvironmentDisplay = $EnvironmentName
}

$SpaceCheckPath = if ($EnvironmentPath) { $EnvironmentPath } else { $env:USERPROFILE }
$SpaceCheckRoot = [IO.Path]::GetPathRoot($SpaceCheckPath)
$SpaceCheckDrive = [IO.DriveInfo]::new($SpaceCheckRoot)
if ($SpaceCheckDrive.AvailableFreeSpace -lt 3GB) {
    throw "At least 3 GB free space is required on $SpaceCheckRoot. Use -EnvironmentPath on another drive."
}

if ($CacheRoot) {
    if (-not [IO.Path]::IsPathRooted($CacheRoot)) {
        throw "CacheRoot must be absolute."
    }
    $CacheRoot = [IO.Path]::GetFullPath($CacheRoot)
    $env:CONDA_PKGS_DIRS = Join-Path $CacheRoot "conda-pkgs"
    $env:PIP_CACHE_DIR = Join-Path $CacheRoot "pip"
    $env:TEMP = Join-Path $CacheRoot "temp"
    $env:TMP = $env:TEMP
    New-Item -ItemType Directory -Force -Path $env:CONDA_PKGS_DIRS, $env:PIP_CACHE_DIR, $env:TEMP | Out-Null
}

function Invoke-CondaCommand {
    param([string[]]$CommandArguments)
    & $CondaExe @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed with exit code $LASTEXITCODE."
    }
}

Invoke-CondaCommand (@("create", "-y") + $EnvironmentArguments + @("python=3.10", "pip=24.3"))
Invoke-CondaCommand (@("run") + $EnvironmentArguments + @(
    "python", "-m", "pip", "install",
    "-r", (Join-Path $ProjectDir "environments\requirements-windows-$Backend.lock.txt")
))
Invoke-CondaCommand (@("run") + $EnvironmentArguments + @(
    "python", "-m", "pip", "install", "--no-deps", $ProjectDir
))
Invoke-CondaCommand (@("run") + $EnvironmentArguments + @(
    "face-preprocess", "--json", "doctor"
))

Write-Host "Environment $EnvironmentDisplay is ready."
