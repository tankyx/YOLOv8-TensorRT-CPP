#Requires -Version 5.1
<#
.SYNOPSIS
  From-scratch Windows bootstrap for YOLOv8-TensorRT-CPP.

.DESCRIPTION
  Installs and verifies every dependency the Windows build needs, then configures
  and builds the project. Idempotent: each step detects an existing installation
  and skips it.

  Steps
    1 Preflight      OS / GPU / compute capability / disk checks
    2 Build tools    git, CMake, Visual Studio 2022 (Desktop C++ workload)
    3 CUDA Toolkit   12.4+ (detect | winget | direct download | instructions)
    4 TensorRT 10.x  merged into the CUDA root (needs an NVIDIA login -> -TensorRtZip)
    5 OpenCV + CUDA  built from source (the project uses cv::cuda / cuda* modules)
    6 Environment    writes scripts\env.bat (CUDA_PATH, OpenCV_DIR, PATH)
    7 Project        CMake configure + Release build + output verification

.PARAMETER ProjectBuildDir
  Build directory, relative to the repo root. Defaults to build2, which is the
  path the checked-in run_*.bat launchers expect. Use 'build' for the README layout.

.PARAMETER TensorRtZip
  Path or URL of the TensorRT Windows zip. NVIDIA requires a free account, so this
  cannot be downloaded unattended.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows_setup.ps1 -DryRun

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows_setup.ps1 -TensorRtZip "$HOME\Downloads\TensorRT-10.9.0.34.Windows.win10.cuda-12.9.zip"
#>
[CmdletBinding()]
param(
    # Where CUDA and OpenCV are installed.
    [string] $InstallRoot      = 'C:\',
    # Scratch space for downloads, OpenCV sources and build trees.
    [string] $WorkDir          = '',
    # CUDA Toolkit version to install when none is detected (README: 12.4+, 12.9 tested).
    [string] $CudaVersion      = '12.9',
    # Optional direct installer URL override.
    [string] $CudaInstallerUrl = '',
    # Path or URL to the TensorRT Windows zip.
    [string] $TensorRtZip      = '',
    [string] $TensorRtVersion  = '10.9.0.34',
    # Best effort: pull TensorRT from PyPI instead of the NVIDIA zip.
    [switch] $UsePipTensorRt,
    [string] $OpenCvVersion    = '4.14.0',
    [string] $OpenCvInstallDir = '',
    # 'auto' detects the GPU compute capability via nvidia-smi (8.9 = RTX 40, 12.0 = RTX 50).
    [string] $CudaArch         = 'auto',
    [ValidateSet('Community', 'BuildTools', 'Professional', 'Enterprise')]
    [string] $VsEdition        = 'Community',
    [string] $ProjectBuildDir  = 'build2',
    # Optional: Python + torch/ultralytics for scripts\pytorch2onnx.py.
    [switch] $WithPython,
    [switch] $SkipCuda,
    [switch] $SkipTensorRt,
    [switch] $SkipOpenCv,
    [switch] $SkipProjectBuild,
    # Reinstall even when a working installation is detected.
    [switch] $Force,
    # Print the actions without executing them.
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'

$script:StepNo    = 0
$script:StepTotal = 7

function Write-Step { param([string] $Text)
    $script:StepNo++
    Write-Host ''
    Write-Host ("[{0}/{1}] {2}" -f $script:StepNo, $script:StepTotal, $Text) -ForegroundColor Cyan
}
function Write-Ok   { param([string] $Text) Write-Host "       ok   $Text" -ForegroundColor Green }
function Write-Skip { param([string] $Text) Write-Host "       skip $Text" -ForegroundColor DarkGray }
function Write-Warn { param([string] $Text) Write-Host "       warn $Text" -ForegroundColor Yellow }
function Write-Info { param([string] $Text) Write-Host "            $Text" -ForegroundColor Gray }

function Die { param([string] $Text)
    Write-Host ''
    Write-Host "       FAIL $Text" -ForegroundColor Red
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}

function Test-Tool { param([string] $Name)
    return [bool] (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal] $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Cmd {
    param(
        [Parameter(Mandatory)] [string]   $Exe,
        [Parameter(Mandatory)] [string[]] $ArgList,
        [string] $What = ''
    )
    if ($What) { Write-Info $What }
    if ($DryRun) {
        Write-Info ("DRY-RUN: {0} {1}" -f $Exe, ($ArgList -join ' '))
        return
    }
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) {
        Die ("command failed (exit {0}): {1} {2}" -f $LASTEXITCODE, $Exe, ($ArgList -join ' '))
    }
}

function Get-Download {
    param(
        [Parameter(Mandatory)] [string] $Url,
        [Parameter(Mandatory)] [string] $Dest
    )
    if (Test-Path $Dest) { Write-Skip "already downloaded: $Dest"; return }
    Write-Info "downloading $Url"
    if ($DryRun) { Write-Info "DRY-RUN: download -> $Dest"; return }
    $parent = Split-Path $Dest -Parent
    if ($parent -and -not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if (Test-Tool 'curl.exe') {
        # curl ships with Windows 10+ and handles large files far better than Invoke-WebRequest.
        & curl.exe -L --fail --retry 3 -o $Dest $Url
        if ($LASTEXITCODE -ne 0) { Die "download failed: $Url" }
    } else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    }
}

function Test-WingetPackage { param([string] $Id)
    if (-not (Test-Tool 'winget')) { return $false }
    $out = (& winget list --id $Id -e --accept-source-agreements 2>$null | Out-String)
    return ($out -match [regex]::Escape($Id))
}

function Install-WingetPackage {
    param(
        [Parameter(Mandatory)] [string]   $Id,
        [Parameter(Mandatory)] [string]   $Label,
        [string[]] $ExtraArgs = @(),
        [string]   $VerifyTool = ''
    )
    if (-not $Force) {
        if (Test-WingetPackage $Id) { Write-Skip "$Label already installed ($Id)"; return }
        if ($VerifyTool -and (Test-Tool $VerifyTool)) { Write-Skip "$Label already on PATH ($VerifyTool)"; return }
    }
    if (-not (Test-Tool 'winget')) {
        Die "winget is not available and $Label is missing; install it manually and re-run."
    }
    $argList = @('install', '--id', $Id, '-e', '--silent',
                 '--accept-package-agreements', '--accept-source-agreements') + $ExtraArgs
    Invoke-Cmd -Exe 'winget' -ArgList $argList -What "installing $Label"
    if ($VerifyTool -and -not $DryRun -and -not (Test-Tool $VerifyTool)) {
        Write-Warn "$Label installed, but '$VerifyTool' is not on this shell's PATH yet - open a new terminal."
    }
}

function Find-VsInstall {
    # Path of the newest Visual Studio with the Desktop C++ workload (falls back
    # to any Visual Studio). Returns $null when none is installed.
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    if (-not $path) { $path = & $vswhere -latest -products '*' -property installationPath 2>$null }
    if (-not $path) { return $null }
    return ($path | Select-Object -First 1).Trim()
}

function Get-VsGenerator {
    # Map the Visual Studio product version to the CMake generator name.
    param([string] $InstallPath)
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path $vswhere) {
        $ver = & $vswhere -latest -products '*' -property installationVersion 2>$null | Select-Object -First 1
        if ($ver -match '^(\d+)') {
            switch ($Matches[1]) {
                '18' { return 'Visual Studio 18 2026' }
                '17' { return 'Visual Studio 17 2022' }
                '16' { return 'Visual Studio 16 2019' }
            }
        }
    }
    return 'Visual Studio 17 2022'
}

function Find-VsCmake {
    # Visual Studio bundles its own CMake under Common7\IDE. Prefer it when
    # cmake is not already on PATH so no winget/admin is required.
    param([string] $VsInstall)
    $candidates = @()
    if ($VsInstall) { $candidates += (Join-Path $VsInstall 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe') }
    $candidates += (Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe')
    $candidates += (Join-Path ${env:ProgramFiles(x86)} 'CMake\bin\cmake.exe')
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

function Convert-ToCMakePath {
    # CMake wants forward slashes on Windows. Backslashes in a PATHS value get
    # re-parsed as escapes (e.g. '\P' in '\Program Files') by the FindCUDA
    # shim shipped with CMake 4.x and abort the configure.
    param([string] $Path)
    if (-not $Path) { return $Path }
    return ($Path -replace '\\', '/')
}
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$script:RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $WorkDir)          { $WorkDir = Join-Path $InstallRoot 'yolo-deps' }
if (-not $OpenCvInstallDir) { $OpenCvInstallDir = Join-Path $InstallRoot ("opencv-{0}" -f $OpenCvVersion) }
if (-not [IO.Path]::IsPathRooted($ProjectBuildDir)) {
    $ProjectBuildDir = Join-Path $script:RepoRoot $ProjectBuildDir
}

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' YOLOv8-TensorRT-CPP - Windows dependency bootstrap'          -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
Write-Info ("repo        : {0}" -f $script:RepoRoot)
Write-Info ("work dir    : {0}" -f $WorkDir)
Write-Info ("OpenCV dir  : {0}" -f $OpenCvInstallDir)
Write-Info ("build dir   : {0}" -f $ProjectBuildDir)
Write-Info ("dry run     : {0}" -f [bool] $DryRun)

# ---------------------------------------------------------------------------
# 1. Preflight
# ---------------------------------------------------------------------------
Write-Step 'Preflight (OS, GPU, disk, rights)'

if (-not [Environment]::Is64BitOperatingSystem) { Die 'A 64-bit Windows installation is required.' }
Write-Ok ("Windows {0} (64-bit)" -f [Environment]::OSVersion.Version)
if ([Environment]::OSVersion.Version.Build -lt 19041) {
    Write-Warn ('Windows 10 build 19041+ is recommended (found {0}).' -f [Environment]::OSVersion.Version.Build)
}

$isAdmin = Test-Admin
if ($isAdmin) { Write-Ok 'running elevated' }
else { Write-Warn 'not elevated - CUDA / Visual Studio / OpenCV installs into C:\ need an admin shell' }

$gpuName = ''
$gpuArch = ''
if (Test-Tool 'nvidia-smi') {
    try {
        $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
        $driver  = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
        $capRaw  = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        Write-Ok ("GPU: {0} (driver {1})" -f $gpuName.Trim(), $driver.Trim())
        if ($capRaw -and ($capRaw.Trim() -match '^([0-9]+)\.([0-9]+)$')) { $gpuArch = "$($Matches[1]).$($Matches[2])" }
    } catch {
        Write-Warn 'nvidia-smi present but not responding; is the NVIDIA driver installed?'
    }
} else {
    Write-Warn 'nvidia-smi not found - install the NVIDIA display driver first (the CUDA toolkit alone is not enough)'
}

if (-not $gpuArch -and $gpuName) {
    # Older nvidia-smi builds have no compute_cap field: infer from the GPU family.
    if     ($gpuName -match 'RTX\s*50|Blackwell')                { $gpuArch = '12.0' }
    elseif ($gpuName -match 'RTX\s*40|L40|L4\b')                 { $gpuArch = '8.9' }
    elseif ($gpuName -match 'RTX\s*30|A40|A10|A30|A5000|A6000')  { $gpuArch = '8.6' }
    elseif ($gpuName -match 'A100')                              { $gpuArch = '8.0' }
    elseif ($gpuName -match 'H100|H200')                         { $gpuArch = '9.0' }
    else                                                         { $gpuArch = '8.9' }
    Write-Warn "compute capability not reported by nvidia-smi; assuming $gpuArch from the GPU name"
}
if (-not $gpuArch) { $gpuArch = '8.9'; Write-Warn "no GPU detected; defaulting to CUDA arch $gpuArch" }

if ($CudaArch -ne 'auto') { $gpuArch = $CudaArch }
if ($gpuArch -notmatch '^[0-9]+\.[0-9]+$') { Die "invalid -CudaArch '$gpuArch' (expected e.g. 8.9 or 12.0)" }
$archNoDot = $gpuArch.Replace('.', '')
Write-Ok ("CUDA arch: {0} (sm_{1})" -f $gpuArch, $archNoDot)

if ($InstallRoot -match '^([A-Za-z]):') {
    try {
        $freeGb = [math]::Round((Get-PSDrive -Name $Matches[1]).Free / 1GB, 1)
        if ($freeGb -lt 40) { Write-Warn "only $freeGb GB free on $InstallRoot - CUDA + TensorRT + an OpenCV CUDA build want ~40-60 GB" }
        else { Write-Ok "$freeGb GB free on $InstallRoot" }
    } catch { Write-Warn 'could not read free disk space' }
}

if (-not $DryRun) {
    if (-not (Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null }
    try { Start-Transcript -Path (Join-Path $WorkDir 'windows_setup.log') -Append | Out-Null } catch { }
}

# ---------------------------------------------------------------------------
# 2. Build tools
# ---------------------------------------------------------------------------
Write-Step 'Build tools (git, CMake, Visual Studio 2022)'

if (Test-Tool 'git') { Write-Ok ((& git --version) -join '') }
else { Install-WingetPackage -Id 'Git.Git' -Label 'Git' -VerifyTool 'git' }

# Visual Studio first: it determines the CMake generator and may provide a
# bundled CMake (no winget / admin required).
$vsInstall   = Find-VsInstall
$vsGenerator = 'Visual Studio 17 2022'
if ($vsInstall) {
    $vsGenerator = Get-VsGenerator $vsInstall
    Write-Ok ("Visual Studio: {0} [{1}]" -f $vsInstall, $vsGenerator)
} else {
    Write-Warn 'Visual Studio with the Desktop C++ (MSVC) workload was not found - add it via the Visual Studio Installer'
    $vsOverride = '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended'
    Install-WingetPackage -Id (@{
        Community    = 'Microsoft.VisualStudio.2022.Community'
        BuildTools   = 'Microsoft.VisualStudio.2022.BuildTools'
        Professional = 'Microsoft.VisualStudio.2022.Professional'
        Enterprise   = 'Microsoft.VisualStudio.2022.Enterprise'
    }[$VsEdition]) -Label "Visual Studio 2022 $VsEdition (Desktop C++)" -ExtraArgs @('--override', $vsOverride)
}

if (Test-Tool 'cmake') {
    $v = (& cmake --version | Select-Object -First 1)
    Write-Ok $v
    if ($v -match '(\d+)\.(\d+)' -and [int]$Matches[1] -lt 3 -and [int]$Matches[2] -lt 22) {
        Write-Warn 'CMake 3.22+ is recommended'
    }
} else {
    $vsCmake = Find-VsCmake $vsInstall
    if ($vsCmake) {
        $env:PATH = (Split-Path $vsCmake -Parent) + ';' + $env:PATH
        Write-Ok ("using the CMake bundled with Visual Studio: {0}" -f (& $vsCmake --version | Select-Object -First 1))
    } elseif (Test-Tool 'winget') {
        Install-WingetPackage -Id 'Kitware.CMake' -Label 'CMake' -VerifyTool 'cmake'
    } else {
        # No winget and no bundled CMake: drop a portable CMake into the work dir.
        $cmakeVersion = '3.31.6'
        $cmakeZip = Join-Path $WorkDir "cmake-$cmakeVersion-windows-x86_64.zip"
        $cmakeDir = Join-Path $WorkDir "cmake-$cmakeVersion-windows-x86_64"
        Get-Download -Url "https://github.com/Kitware/CMake/releases/download/v$cmakeVersion/cmake-$cmakeVersion-windows-x86_64.zip" -Dest $cmakeZip
        if (-not $DryRun) {
            if (-not (Test-Path (Join-Path $cmakeDir 'bin\cmake.exe'))) {
                Expand-Archive -Path $cmakeZip -DestinationPath $WorkDir -Force
            }
            $env:PATH = (Join-Path $cmakeDir 'bin') + ';' + $env:PATH
        }
        Write-Ok "CMake $cmakeVersion (portable, no admin needed)"
    }
}

# ---------------------------------------------------------------------------
# 3. CUDA Toolkit
# ---------------------------------------------------------------------------
Write-Step "CUDA Toolkit $CudaVersion"

function Find-CudaRoot {
    $base = Join-Path ${env:ProgramFiles} 'NVIDIA GPU Computing Toolkit\CUDA'
    if (-not (Test-Path $base)) { return $null }
    $candidates = @(Get-ChildItem -Path $base -Directory -Filter 'v*' -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName 'bin\nvcc.exe') })
    if ($candidates.Count -eq 0) { return $null }
    $sorted = $candidates | Sort-Object -Property @{ Expression = { [version] ($_.Name.TrimStart('v')) } } -Descending
    return $sorted[0].FullName
}

$cudaRoot = Find-CudaRoot
if ($cudaRoot -and -not $Force) {
    $nvccLine = ''
    try { $nvccLine = (& (Join-Path $cudaRoot 'bin\nvcc.exe') --version | Select-String -Pattern 'release\s+[0-9.]+' | Select-Object -First 1) }
    catch { }
    Write-Skip "CUDA already installed: $cudaRoot"
    if ($nvccLine) { Write-Info $nvccLine.ToString().Trim() }
} elseif ($SkipCuda) {
    Die 'no CUDA Toolkit found and -SkipCuda was given'
} else {
    if (-not $isAdmin -and -not $DryRun) { Die 'installing the CUDA Toolkit needs an elevated (admin) PowerShell' }
    $installed = $false
    if (Test-Tool 'winget') {
        Write-Info 'trying winget (Nvidia.CUDA)'
        $cudaArgs = @('install', '--id', 'Nvidia.CUDA', '-e', '--silent',
                      '--accept-package-agreements', '--accept-source-agreements')
        if ($DryRun) { Write-Info ("DRY-RUN: winget {0}" -f ($cudaArgs -join ' ')) }
        else {
            & winget @cudaArgs
            $cudaRoot = Find-CudaRoot
            if ($LASTEXITCODE -eq 0 -and $cudaRoot) { $installed = $true; Write-Ok "CUDA: $cudaRoot" }
            else { Write-Warn 'winget could not install CUDA; falling back to the direct installer' }
        }
    }
    if (-not $installed -and -not $DryRun) {
        $url = $CudaInstallerUrl
        if (-not $url) {
            $url = "https://developer.download.nvidia.com/compute/cuda/$CudaVersion.0/local_installers/cuda_$CudaVersion.0_576.57_windows.exe"
            Write-Info "no -CudaInstallerUrl given; trying $url"
        }
        $installer = Join-Path $WorkDir (Split-Path $url -Leaf)
        Get-Download -Url $url -Dest $installer
        Invoke-Cmd -Exe $installer -ArgList @('-s', 'cuda_runtime', 'cuda_dev', 'cuda_nvcc', 'cuda_libraries') `
                   -What 'running the CUDA installer silently (this takes several minutes)'
        $cudaRoot = Find-CudaRoot
        if (-not $cudaRoot) {
            Die ('the CUDA installer ran but no CUDA installation was detected. Install it by hand from ' +
                 'https://developer.nvidia.com/cuda-downloads and re-run this script.')
        }
        Write-Ok "CUDA: $cudaRoot"
    }
    if ($DryRun -and -not $cudaRoot) { $cudaRoot = Join-Path ${env:ProgramFiles} "NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion" }
}

# Forward-slash form for every path handed to CMake (see Convert-ToCMakePath).
$cudaRootCm = Convert-ToCMakePath $cudaRoot

# ---------------------------------------------------------------------------
# 4. TensorRT (merged into the CUDA root, matching CMakeLists.txt)
# ---------------------------------------------------------------------------
Write-Step 'TensorRT (merged into the CUDA root)'

function Test-TensorRtInCudaRoot { param([string] $Root)
    if (-not $Root) { return $false }
    if (-not (Test-Path (Join-Path $Root 'include\NvInfer.h'))) { return $false }
    foreach ($suffix in @('_10', '_11', '')) {
        foreach ($sub in @('lib', 'bin')) {
            $dir = Join-Path $Root $sub
            if ((Test-Path (Join-Path $dir "nvinfer$suffix.lib")) -and
                (Test-Path (Join-Path $dir "nvonnxparser$suffix.lib")) -and
                (Test-Path (Join-Path $dir "nvinfer_plugin$suffix.lib"))) {
                return $true
            }
        }
    }
    return $false
}

function Install-TensorRtFromZip {
    param(
        [Parameter(Mandatory)] [string] $ZipPath,
        [Parameter(Mandatory)] [string] $CudaRoot
    )
    $extractDir = Join-Path $WorkDir 'tensorrt'
    if ($DryRun) { Write-Info "DRY-RUN: extract $ZipPath and merge into $CudaRoot"; return }
    if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    Write-Info "extracting $ZipPath"
    Expand-Archive -Path $ZipPath -DestinationPath $extractDir -Force

    $header = @(Get-ChildItem -Path $extractDir -Recurse -Filter 'NvInfer.h' -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($header.Count -eq 0) { Die "NvInfer.h not found inside $ZipPath - is this the TensorRT Windows zip?" }
    $trtRoot = Split-Path -Parent (Split-Path -Parent $header[0].FullName)

    foreach ($sub in @('include', 'lib', 'bin')) {
        $from = Join-Path $trtRoot $sub
        if (-not (Test-Path $from)) { continue }
        $to = Join-Path $CudaRoot $sub
        if (-not (Test-Path $to)) { New-Item -ItemType Directory -Path $to -Force | Out-Null }
        Copy-Item -Path (Join-Path $from '*') -Destination $to -Recurse -Force
        Write-Info ("copied {0} -> {1}" -f $from, $to)
    }
    if (-not (Test-TensorRtInCudaRoot $CudaRoot)) { Die "TensorRT files do not look usable in $CudaRoot" }
    Write-Ok "TensorRT merged into $CudaRoot"
}

if ($SkipTensorRt) {
    Write-Skip 'skipped (-SkipTensorRt)'
} elseif ((Test-TensorRtInCudaRoot $cudaRoot) -and -not $Force) {
    Write-Skip "TensorRT already present in $cudaRoot"
} elseif ($TensorRtZip) {
    $zipPath = $TensorRtZip
    if ($zipPath -match '^https?://') {
        $zipPath = Join-Path $WorkDir (Split-Path $TensorRtZip -Leaf)
        Get-Download -Url $TensorRtZip -Dest $zipPath
    } elseif (-not (Test-Path $zipPath)) {
        Die "TensorRT zip not found: $zipPath"
    }
    if (-not $isAdmin -and -not $DryRun -and ($cudaRoot -like '*Program Files*')) {
        Die ("merging TensorRT into $cudaRoot needs an elevated (admin) PowerShell. " +
             "Re-open the terminal as Administrator and re-run, or pass -InstallRoot to a user-writable location.")
    }
    Install-TensorRtFromZip -ZipPath $zipPath -CudaRoot $cudaRoot
} elseif ($UsePipTensorRt) {
    if (-not (Test-Tool 'python')) { Die '-UsePipTensorRt needs Python on PATH' }
    Invoke-Cmd -Exe 'python' -ArgList @('-m', 'pip', 'install', '--upgrade', "tensorrt==$TensorRtVersion") -What 'installing tensorrt from PyPI'
    if (-not $DryRun) {
        $site = (& python -c "import site,os;print(os.path.dirname(site.getsitepackages()[0]))").Trim()
        $libs = Join-Path $site 'tensorrt_libs'
        $inc  = Join-Path $site 'tensorrt\include'
        if (-not (Test-Path $libs)) { Die 'tensorrt_libs not found after the pip install' }
        Copy-Item -Path (Join-Path $libs '*.dll') -Destination (Join-Path $cudaRoot 'bin') -Force -ErrorAction SilentlyContinue
        if (Test-Path $inc) { Copy-Item -Path (Join-Path $inc '*.h') -Destination (Join-Path $cudaRoot 'include') -Force }
        if (-not (Test-TensorRtInCudaRoot $cudaRoot)) {
            Die ('the PyPI TensorRT package did not provide NvInfer.h + nvinfer / nvonnxparser / nvinfer_plugin libs. ' +
                 'Download the TensorRT Windows zip from https://developer.nvidia.com/tensorrt and re-run with -TensorRtZip <path>.')
        }
        Write-Ok 'TensorRT installed from PyPI'
    }
} else {
    Write-Host ''
    Write-Warn 'TensorRT is not installed and no source was given.'
    Write-Info 'NVIDIA requires a free account, so this download cannot be automated:'
    Write-Info '  1. open https://developer.nvidia.com/tensorrt and download the TensorRT Windows zip that matches your CUDA version'
    Write-Info '  2. re-run:  scripts\windows_setup.bat -TensorRtZip <path-to-zip>'
    Die 'stopping before the (30-60 minute) OpenCV build because the project cannot link without TensorRT'
}
# ---------------------------------------------------------------------------
# 5. OpenCV with CUDA
# ---------------------------------------------------------------------------
Write-Step "OpenCV $OpenCvVersion with CUDA (cv::cuda required by the source)"

function Find-OpenCvConfigDir { param([string] $Root)
    foreach ($candidate in @($Root, (Join-Path $Root 'build'), (Join-Path $Root 'lib\cmake\opencv4'))) {
        if ($candidate -and (Test-Path (Join-Path $candidate 'OpenCVConfig.cmake'))) { return $candidate }
    }
    return $null
}

$opencvDir = Find-OpenCvConfigDir $OpenCvInstallDir
if ($opencvDir -and -not $Force) {
    Write-Skip "OpenCV already installed: $opencvDir"
} elseif ($SkipOpenCv) {
    Die 'no usable OpenCV found and -SkipOpenCv was given'
} else {
    Write-Info 'the official prebuilt OpenCV for Windows has no CUDA modules, so this builds from source'
    Write-Info 'expect 30-60 minutes and ~15 GB of scratch space'
    $openCvSrc     = Join-Path $WorkDir "opencv-$OpenCvVersion"
    $openCvContrib = Join-Path $WorkDir "opencv_contrib-$OpenCvVersion"
    $openCvBuild   = Join-Path $WorkDir "opencv-build-$OpenCvVersion"

    if (-not (Test-Path (Join-Path $openCvSrc 'CMakeLists.txt'))) {
        Invoke-Cmd -Exe 'git' -ArgList @('clone', '--depth', '1', '--branch', $OpenCvVersion,
                                         'https://github.com/opencv/opencv.git', $openCvSrc) -What 'cloning opencv'
    } else { Write-Skip "opencv sources present: $openCvSrc" }

    if (-not (Test-Path (Join-Path $openCvContrib 'modules'))) {
        Invoke-Cmd -Exe 'git' -ArgList @('clone', '--depth', '1', '--branch', $OpenCvVersion,
                                         'https://github.com/opencv/opencv_contrib.git', $openCvContrib) -What 'cloning opencv_contrib'
    } else { Write-Skip "opencv_contrib sources present: $openCvContrib" }

    $openCvSrcCm     = Convert-ToCMakePath $openCvSrc
    $openCvBuildCm   = Convert-ToCMakePath $openCvBuild
    $openCvInstallCm = Convert-ToCMakePath $OpenCvInstallDir
    $openCvContribCm = Convert-ToCMakePath (Join-Path $openCvContrib 'modules')
    $configureArgs = @(
        '-S', $openCvSrcCm, '-B', $openCvBuildCm,
        '-G', $vsGenerator, '-A', 'x64',
        "-DCMAKE_INSTALL_PREFIX=$openCvInstallCm",
        "-DOPENCV_EXTRA_MODULES_PATH=$openCvContribCm",
        "-DCUDA_TOOLKIT_ROOT_DIR=$cudaRootCm",
        '-DWITH_CUDA=ON',
        # Inference goes through TensorRT, not the OpenCV DNN CUDA backend, so
        # cuDNN is not required (and is usually absent on dev machines).
        '-DWITH_CUDNN=OFF',
        '-DOPENCV_DNN_CUDA=OFF',
        "-DCUDA_ARCH_BIN=$gpuArch",
        '-DCUDA_FAST_MATH=ON',
        '-DWITH_CUBLAS=ON',
        '-DBUILD_opencv_world=ON',
        '-DBUILD_TESTS=OFF',
        '-DBUILD_PERF_TESTS=OFF',
        '-DBUILD_EXAMPLES=OFF',
        '-DBUILD_opencv_python3=OFF'
    )
    Invoke-Cmd -Exe 'cmake' -ArgList $configureArgs -What 'configuring OpenCV'
    Invoke-Cmd -Exe 'cmake' -ArgList @('--build', $openCvBuildCm, '--config', 'Release', '--target', 'INSTALL', '--', '/m') `
               -What 'building and installing OpenCV (long; -DryRun prints the exact command)'

    $opencvDir = Find-OpenCvConfigDir $OpenCvInstallDir
    if ($DryRun -and -not $opencvDir) { $opencvDir = $OpenCvInstallDir }
    if (-not $opencvDir) {
        Die ("OpenCV built but no OpenCVConfig.cmake was installed under $OpenCvInstallDir. If the configure " +
             "step rejected CUDA_ARCH_BIN=$gpuArch, this OpenCV tag predates sm_${archNoDot}: retry with a newer " +
             "-OpenCvVersion (4.14+ knows sm_120 / CUDA 13) or with -CudaArch 8.9.")
    }
    Write-Ok "OpenCV: $opencvDir"
}

# ---------------------------------------------------------------------------
# 6. Environment file
# ---------------------------------------------------------------------------
Write-Step 'Environment file (scripts\env.bat)'

$opencvBinDir = ''
$vcRoot = Join-Path $OpenCvInstallDir 'x64'
if (Test-Path $vcRoot) {
    foreach ($vc in @(Get-ChildItem -Path $vcRoot -Directory -Filter 'vc*' -ErrorAction SilentlyContinue | Sort-Object Name -Descending)) {
        $bin = Join-Path $vc.FullName 'bin'
        if (Test-Path $bin) { $opencvBinDir = $bin; break }
    }
}
if (-not $opencvBinDir) { Write-Warn "no x64\vc*\bin directory found under $OpenCvInstallDir; OpenCV DLLs may not be on PATH" }

$envBat     = Join-Path $PSScriptRoot 'env.bat'
$envFormat  = @'
@echo off
rem Generated by scripts\windows_setup.ps1.
rem Run this in a cmd.exe window before launching the detector, or call it from your own .bat:
rem   scripts\env.bat
set "CUDA_PATH={0}"
set "OpenCV_DIR={1}"
rem CUDA 13 moved the runtime DLLs into bin\x64 (CUDA 12 and earlier keep them in bin).
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;{2}%PATH%"
'@
$envPrefix = ''
if ($opencvBinDir) { $envPrefix = "$opencvBinDir;" }
$envText = $envFormat -f $cudaRoot, $opencvDir, $envPrefix
if ($DryRun) { Write-Info "DRY-RUN: write $envBat" }
else {
    Set-Content -Path $envBat -Value $envText -Encoding ASCII
    Write-Ok "wrote $envBat"
}
Write-Info ("CUDA_PATH  = {0}" -f $cudaRoot)
Write-Info ("OpenCV_DIR = {0}" -f $opencvDir)

# ---------------------------------------------------------------------------
# 7. Project build
# ---------------------------------------------------------------------------
Write-Step 'Project build (CMake configure + Release)'

$exe = Join-Path $ProjectBuildDir 'bin\Release\detect_object_image.exe'
if ($SkipProjectBuild) {
    Write-Skip 'skipped (-SkipProjectBuild)'
} elseif (Test-Tool 'cmake') {
    $repoRootCm  = Convert-ToCMakePath $script:RepoRoot
    $buildDirCm  = Convert-ToCMakePath $ProjectBuildDir
    $opencvDirCm = Convert-ToCMakePath $opencvDir
    Invoke-Cmd -Exe 'cmake' -ArgList @(
        '-S', $repoRootCm, '-B', $buildDirCm,
        '-G', $vsGenerator, '-A', 'x64',
        '-DCMAKE_BUILD_TYPE=Release',
        "-DOpenCV_DIR=$opencvDirCm",
        "-DTENSORRT_ROOT=$cudaRootCm",
        "-DCUDAToolkit_ROOT=$cudaRootCm",
        "-DCMAKE_CUDA_ARCHITECTURES=$archNoDot"
    ) -What 'configuring the project'
    Invoke-Cmd -Exe 'cmake' -ArgList @('--build', $buildDirCm, '--config', 'Release', '--', '/m') `
               -What 'building the project (Release)'
    if (-not $DryRun) {
        if (Test-Path $exe) { Write-Ok "built: $exe" }
        else { Die "the build finished but $exe does not exist" }
        if (Test-Path (Join-Path $ProjectBuildDir 'bin\Release\config_cs2.ini')) { Write-Ok 'config INIs copied next to the exe' }
        else { Write-Warn 'dep\*.ini were not copied next to the exe - copy the config you want by hand' }
    }
} else {
    Write-Warn 'cmake is not available; skipping the project build'
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
if ($DryRun) { Write-Host ' Dry run finished - nothing was installed.' -ForegroundColor Cyan }
else        { Write-Host ' Done.' -ForegroundColor Cyan }
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''
Write-Host 'To run the detector (every new cmd.exe window):' -ForegroundColor White
Write-Host ('  cd /d "{0}\bin\Release"' -f $ProjectBuildDir)
Write-Host ('  "{0}"' -f $envBat)
Write-Host '  detect_object_image.exe config_cs2.ini'
Write-Host ''
Write-Info 'run_cs2.bat / run_valo.bat also work, but load scripts\env.bat in that window first.'
Write-Info 'The first launch compiles the TensorRT engine from the ONNX model: 30-60 seconds.'
Write-Info 'Press Insert to exit.'
if (-not $DryRun) { try { Stop-Transcript | Out-Null } catch { } }

