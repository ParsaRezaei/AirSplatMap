# Download Benchmark Datasets for AirSplatMap
# Supports: TUM RGB-D, Replica, 7-Scenes, ICL-NUIM
# All datasets have RGB color images + depth + ground truth poses

param(
    [Parameter(Position=0)]
    [string]$DatasetsDir = ".\datasets",
    
    [Parameter(Position=1)]
    [ValidateSet("tum", "replica", "7scenes", "icl", "scannet", "all", "list", "help")]
    [string]$Command = "help",
    
    [Parameter(Position=2)]
    [string]$Scenes = ""
)

# ============================================
# Helper Functions
# ============================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "========================================" -ForegroundColor Blue
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[-] $Message" -ForegroundColor Cyan
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Download-File {
    param(
        [string]$Url,
        [string]$Output
    )
    
    Write-Info "Downloading: $Output"
    
    try {
        # Check if curl.exe exists (real curl, not PowerShell alias)
        $curlExe = Get-Command "curl.exe" -ErrorAction SilentlyContinue
        
        if ($curlExe) {
            # Use real curl.exe (available on Windows 10 1803+)
            & curl.exe -L --progress-bar -o $Output $Url
            if ($LASTEXITCODE -ne 0) { throw "curl.exe failed with exit code $LASTEXITCODE" }
        }
        else {
            # Use Invoke-WebRequest (built-in PowerShell)
            $ProgressPreference = 'Continue'
            Invoke-WebRequest -Uri $Url -OutFile $Output -UseBasicParsing
        }
        return $true
    }
    catch {
        Write-ErrorMsg "Failed to download: $Url"
        Write-ErrorMsg $_.Exception.Message
        return $false
    }
}

function Extract-TarGz {
    param(
        [string]$Archive,
        [string]$Destination = "."
    )
    
    Write-Info "Extracting: $Archive"
    
    # Try using tar (available on Windows 10+)
    $tarCmd = Get-Command "tar" -ErrorAction SilentlyContinue
    if ($tarCmd) {
        try {
            & tar -xzf $Archive -C $Destination 2>$null
            if ($LASTEXITCODE -eq 0) {
                return $true
            }
        }
        catch { }
    }
    
    # Fall back to 7-Zip if available
    $7zPaths = @(
        "C:\Program Files\7-Zip\7z.exe",
        "C:\Program Files (x86)\7-Zip\7z.exe",
        "$env:LOCALAPPDATA\7-Zip\7z.exe"
    )
    
    foreach ($7zPath in $7zPaths) {
        if (Test-Path $7zPath) {
            # Extract .tar.gz in two steps with 7-Zip
            $tarFile = $Archive -replace "\.gz$", ""
            & $7zPath x -y $Archive -o"$Destination" | Out-Null
            if (Test-Path $tarFile) {
                & $7zPath x -y $tarFile -o"$Destination" | Out-Null
                Remove-Item $tarFile -Force
            }
            return $true
        }
    }
    
    Write-ErrorMsg "No extraction tool found. Please install tar (Windows 10+) or 7-Zip."
    return $false
}

function Extract-Zip {
    param(
        [string]$Archive,
        [string]$Destination = "."
    )
    
    Write-Info "Extracting: $Archive"
    
    try {
        Expand-Archive -Path $Archive -DestinationPath $Destination -Force
        return $true
    }
    catch {
        Write-ErrorMsg "Failed to extract: $Archive"
        Write-ErrorMsg $_.Exception.Message
        return $false
    }
}

# ============================================
# TUM RGB-D Dataset
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset
# ============================================
function Download-TUM {
    param(
        [string]$DatasetsDir,
        [string]$SceneSelection
    )
    
    Write-Header "TUM RGB-D Dataset"
    
    $TumDir = Join-Path $DatasetsDir "tum"
    if (-not (Test-Path $TumDir)) {
        New-Item -ItemType Directory -Path $TumDir -Force | Out-Null
    }
    
    # Available TUM sequences
    $TumScenes = @{
        # Freiburg1 - Handheld SLAM
        "fr1_xyz"     = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
        "fr1_desk"    = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
        "fr1_desk2"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz"
        "fr1_room"    = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
        "fr1_360"     = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz"
        "fr1_floor"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz"
        "fr1_plant"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz"
        "fr1_teddy"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz"
        
        # Freiburg2 - Robot SLAM
        "fr2_xyz"           = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
        "fr2_desk"          = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz"
        "fr2_desk_person"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk_with_person.tgz"
        "fr2_large_no_loop" = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_no_loop.tgz"
        "fr2_pioneer_360"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz"
        "fr2_pioneer_slam"  = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz"
        
        # Freiburg3 - Structure vs texture
        "fr3_office"        = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
        "fr3_nstr_tex_near" = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz"
        "fr3_str_notex_far" = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_notexture_far.tgz"
        "fr3_str_tex_far"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz"
        "fr3_sitting_xyz"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz"
        "fr3_sitting_half"  = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz"
        "fr3_walking_xyz"   = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"
        "fr3_cabinet"       = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet.tgz"
    }
    
    # Default selection
    $DefaultTum = @("fr1_xyz", "fr1_desk", "fr1_desk2", "fr1_room", "fr2_xyz", "fr2_desk", "fr3_office")
    
    # Parse selection
    $Selected = switch ($SceneSelection) {
        "all"     { $TumScenes.Keys }
        "minimal" { @("fr1_xyz", "fr1_desk2", "fr3_office") }
        ""        { $DefaultTum }
        default   { $SceneSelection -split "," }
    }
    
    Write-Host "Downloading TUM RGB-D scenes to: $TumDir"
    Write-Host "Selected scenes: $($Selected -join ', ')"
    Write-Host ""
    
    Push-Location $TumDir
    
    foreach ($scene in $Selected) {
        $url = $TumScenes[$scene]
        if (-not $url) {
            Write-Warning "Unknown scene: $scene (skipping)"
            continue
        }
        
        $filename = Split-Path $url -Leaf
        $dirname = $filename -replace "\.tgz$", ""
        
        if (Test-Path $dirname) {
            Write-Success "Already exists: $dirname"
            continue
        }
        
        if (Download-File -Url $url -Output $filename) {
            if (Extract-TarGz -Archive $filename -Destination ".") {
                Remove-Item $filename -Force -ErrorAction SilentlyContinue
                Write-Success "Done: $dirname"
            }
        }
        Write-Host ""
    }
    
    Pop-Location
}

# ============================================
# Replica Dataset (NICE-SLAM version)
# ============================================
function Download-Replica {
    param([string]$DatasetsDir)
    
    Write-Header "Replica Dataset"
    
    $ReplicaDir = Join-Path $DatasetsDir "replica"
    
    if ((Test-Path $ReplicaDir) -and (Get-ChildItem $ReplicaDir -ErrorAction SilentlyContinue)) {
        Write-Success "Replica dataset already exists at: $ReplicaDir"
        return
    }
    
    if (-not (Test-Path $DatasetsDir)) {
        New-Item -ItemType Directory -Path $DatasetsDir -Force | Out-Null
    }
    
    Push-Location $DatasetsDir
    
    Write-Info "Downloading Replica dataset (NICE-SLAM version)..."
    Write-Info "This may take a while (~5GB)..."
    
    $url = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
    
    if (Download-File -Url $url -Output "Replica.zip") {
        if (Extract-Zip -Archive "Replica.zip" -Destination ".") {
            if (Test-Path "Replica") {
                Rename-Item "Replica" "replica" -Force
            }
            Remove-Item "Replica.zip" -Force -ErrorAction SilentlyContinue
            Write-Success "Replica dataset downloaded to: $ReplicaDir"
            
            Write-Host ""
            Write-Host "Available scenes:"
            Get-ChildItem $ReplicaDir -Directory | ForEach-Object { Write-Host "  $($_.Name)" }
        }
    }
    
    Pop-Location
}

# ============================================
# Microsoft 7-Scenes Dataset
# ============================================
function Download-7Scenes {
    param(
        [string]$DatasetsDir,
        [string]$SceneSelection
    )
    
    Write-Header "Microsoft 7-Scenes Dataset"
    
    $ScenesDir = Join-Path $DatasetsDir "7scenes"
    if (-not (Test-Path $ScenesDir)) {
        New-Item -ItemType Directory -Path $ScenesDir -Force | Out-Null
    }
    
    $SevenScenes = @{
        "chess"      = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
        "fire"       = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
        "heads"      = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
        "office"     = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
        "pumpkin"    = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
        "redkitchen" = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
        "stairs"     = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
    }
    
    # Default selection (good for GS)
    $Default7Scenes = @("chess", "fire", "office", "redkitchen")
    
    # Parse selection
    $Selected = switch ($SceneSelection) {
        "all"     { $SevenScenes.Keys }
        "minimal" { @("chess") }
        ""        { $Default7Scenes }
        default   { $SceneSelection -split "," }
    }
    
    Write-Host "Downloading 7-Scenes to: $ScenesDir"
    Write-Host "Selected scenes: $($Selected -join ', ')"
    Write-Host ""
    Write-Host "Note: Each scene is 1-4 GB. Total ~17GB for all scenes."
    Write-Host ""
    
    Push-Location $ScenesDir
    
    foreach ($scene in $Selected) {
        $url = $SevenScenes[$scene]
        if (-not $url) {
            Write-Warning "Unknown scene: $scene (skipping)"
            continue
        }
        
        $sceneDir = Join-Path $ScenesDir $scene
        
        # Check if already exists with extracted sequences
        if (Test-Path $sceneDir) {
            $seqDirs = Get-ChildItem $sceneDir -Directory -Filter "seq-*" -ErrorAction SilentlyContinue
            if ($seqDirs.Count -gt 0) {
                $frameFiles = Get-ChildItem (Join-Path $sceneDir "seq-01") -Filter "frame-*.color.png" -ErrorAction SilentlyContinue
                if ($frameFiles.Count -gt 0) {
                    Write-Success "Already exists: $scene"
                    continue
                }
            }
        }
        
        $zipFile = "$scene.zip"
        
        if (-not (Test-Path $zipFile)) {
            if (-not (Download-File -Url $url -Output $zipFile)) {
                continue
            }
        }
        
        if (Extract-Zip -Archive $zipFile -Destination ".") {
            Remove-Item $zipFile -Force -ErrorAction SilentlyContinue
            
            # Extract nested seq-XX.zip files
            if (Test-Path $scene) {
                Push-Location $scene
                $seqZips = Get-ChildItem -Filter "seq-*.zip" -ErrorAction SilentlyContinue
                foreach ($seqZip in $seqZips) {
                    $seqName = $seqZip.BaseName
                    Write-Info "  Extracting sequence: $seqName"
                    Expand-Archive -Path $seqZip.FullName -DestinationPath "." -Force
                    Remove-Item $seqZip.FullName -Force
                }
                Pop-Location
            }
            
            Write-Success "Done: $scene"
        }
        Write-Host ""
    }
    
    Pop-Location
    
    Write-Host ""
    Write-Host "7-Scenes format:"
    Write-Host "  Each frame has: color.png, depth.png, pose.txt"
    Write-Host "  Depth: 16-bit PNG in millimeters"
    Write-Host "  Pose: 4x4 camera-to-world matrix"
}

# ============================================
# ICL-NUIM Dataset
# ============================================
function Download-IclNuim {
    param(
        [string]$DatasetsDir,
        [string]$SceneSelection
    )
    
    Write-Header "ICL-NUIM Dataset"
    
    $IclDir = Join-Path $DatasetsDir "icl_nuim"
    if (-not (Test-Path $IclDir)) {
        New-Item -ItemType Directory -Path $IclDir -Force | Out-Null
    }
    
    $IclScenes = @{
        "lr_kt0" = "https://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz"
        "lr_kt1" = "https://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz"
        "lr_kt2" = "https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz"
        "lr_kt3" = "https://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz"
        "of_kt0" = "https://www.doc.ic.ac.uk/~ahanda/office_room_traj0_frei_png.tar.gz"
        "of_kt1" = "https://www.doc.ic.ac.uk/~ahanda/office_room_traj1_frei_png.tar.gz"
        "of_kt2" = "https://www.doc.ic.ac.uk/~ahanda/office_room_traj2_frei_png.tar.gz"
        "of_kt3" = "https://www.doc.ic.ac.uk/~ahanda/office_room_traj3_frei_png.tar.gz"
    }
    
    $DefaultIcl = @("lr_kt0", "lr_kt1", "of_kt0", "of_kt1")
    
    # Parse selection
    $Selected = switch ($SceneSelection) {
        "all"     { $IclScenes.Keys }
        "minimal" { @("lr_kt0") }
        ""        { $DefaultIcl }
        default   { $SceneSelection -split "," }
    }
    
    Write-Host "Downloading ICL-NUIM scenes to: $IclDir"
    Write-Host "Selected scenes: $($Selected -join ', ')"
    Write-Host ""
    
    Push-Location $IclDir
    
    foreach ($scene in $Selected) {
        $url = $IclScenes[$scene]
        if (-not $url) {
            Write-Warning "Unknown scene: $scene (skipping)"
            continue
        }
        
        $filename = Split-Path $url -Leaf
        
        # Determine expected directory name
        if ($scene -like "lr_*") {
            $trajNum = $scene.Substring(4, 1)
            $expectedDir = "living_room_traj${trajNum}_frei_png"
        }
        else {
            $trajNum = $scene.Substring(4, 1)
            $expectedDir = "office_room_traj${trajNum}_frei_png"
        }
        
        if (Test-Path $expectedDir) {
            Write-Success "Already exists: $scene"
            continue
        }
        
        if (Download-File -Url $url -Output $filename) {
            if (Extract-TarGz -Archive $filename -Destination ".") {
                Remove-Item $filename -Force -ErrorAction SilentlyContinue
                Write-Success "Done: $scene"
            }
        }
        Write-Host ""
    }
    
    Pop-Location
}

# ============================================
# ScanNet Info
# ============================================
function Show-ScanNetInfo {
    Write-Header "ScanNet Dataset"
    
    Write-Host "ScanNet requires accepting a license agreement."
    Write-Host ""
    Write-Host "To download ScanNet:"
    Write-Host "1. Go to: http://www.scan-net.org/"
    Write-Host "2. Fill out the Terms of Use agreement"
    Write-Host "3. You will receive a download script via email"
    Write-Host ""
    Write-Host "After receiving access, place scenes in: $DatasetsDir\scannet\"
    Write-Host ""
    Write-Host "Recommended scenes for benchmarking:"
    Write-Host "  - scene0000_00, scene0059_00, scene0106_00"
    Write-Host "  - scene0169_00, scene0181_00, scene0207_00"
}

# ============================================
# List Scenes
# ============================================
function Show-SceneList {
    Write-Header "Available Scenes"
    
    Write-Host "TUM RGB-D Dataset:" -ForegroundColor Yellow
    Write-Host "  Freiburg1: fr1_xyz, fr1_desk, fr1_desk2, fr1_room, fr1_360, fr1_floor, fr1_plant, fr1_teddy"
    Write-Host "  Freiburg2: fr2_xyz, fr2_desk, fr2_desk_person, fr2_large_no_loop, fr2_pioneer_360, fr2_pioneer_slam"
    Write-Host "  Freiburg3: fr3_office, fr3_nstr_tex_near, fr3_str_notex_far, fr3_str_tex_far, fr3_sitting_xyz, fr3_walking_xyz, fr3_cabinet"
    Write-Host ""
    
    Write-Host "Replica Dataset (synthetic):" -ForegroundColor Yellow
    Write-Host "  office0, office1, office2, office3, office4"
    Write-Host "  room0, room1, room2"
    Write-Host ""
    
    Write-Host "7-Scenes Dataset (RGB-D with KinectFusion GT):" -ForegroundColor Yellow
    Write-Host "  chess, fire, heads, office, pumpkin, redkitchen, stairs"
    Write-Host "  Best for GS: chess, office, redkitchen (good texture)"
    Write-Host ""
    
    Write-Host "ICL-NUIM Dataset (synthetic):" -ForegroundColor Yellow
    Write-Host "  Living Room: lr_kt0, lr_kt1, lr_kt2, lr_kt3"
    Write-Host "  Office: of_kt0, of_kt1, of_kt2, of_kt3"
    Write-Host ""
}

# ============================================
# Show Usage
# ============================================
function Show-Usage {
    Write-Host ""
    Write-Host "AirSplatMap Dataset Downloader" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\download_datasets.ps1 [DatasetsDir] <Command> [Scenes]"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  tum [scenes]      Download TUM RGB-D dataset (real indoor scenes)"
    Write-Host "  7scenes [scenes]  Download Microsoft 7-Scenes (RGB-D with GT poses)"
    Write-Host "  replica           Download Replica dataset (synthetic, NICE-SLAM version)"
    Write-Host "  icl [scenes]      Download ICL-NUIM dataset (synthetic with perfect GT)"
    Write-Host "  scannet           Show ScanNet download instructions"
    Write-Host "  all               Download all datasets (default selection)"
    Write-Host "  list              List available scenes"
    Write-Host ""
    Write-Host "Scene options:" -ForegroundColor Yellow
    Write-Host "  all               Download all available scenes"
    Write-Host "  minimal           Download minimal set for testing"
    Write-Host "  scene1,scene2     Comma-separated list of scenes"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\download_datasets.ps1 .\datasets tum                    # Default TUM scenes"
    Write-Host "  .\download_datasets.ps1 .\datasets tum all                # All TUM scenes"
    Write-Host "  .\download_datasets.ps1 .\datasets tum fr1_xyz,fr2_desk   # Specific scenes"
    Write-Host "  .\download_datasets.ps1 .\datasets replica                # Replica dataset"
    Write-Host "  .\download_datasets.ps1 .\datasets 7scenes                # Default 7-Scenes"
    Write-Host "  .\download_datasets.ps1 .\datasets 7scenes all            # All 7-Scenes (~17GB)"
    Write-Host "  .\download_datasets.ps1 .\datasets all                    # All datasets"
    Write-Host ""
}

# ============================================
# Main
# ============================================

# Resolve absolute path for datasets directory
$DatasetsDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DatasetsDir)

# Create datasets directory if needed
if (-not (Test-Path $DatasetsDir)) {
    New-Item -ItemType Directory -Path $DatasetsDir -Force | Out-Null
}

switch ($Command.ToLower()) {
    "tum" {
        Download-TUM -DatasetsDir $DatasetsDir -SceneSelection $Scenes
    }
    "replica" {
        Download-Replica -DatasetsDir $DatasetsDir
    }
    "7scenes" {
        Download-7Scenes -DatasetsDir $DatasetsDir -SceneSelection $Scenes
    }
    "icl" {
        Download-IclNuim -DatasetsDir $DatasetsDir -SceneSelection $Scenes
    }
    "scannet" {
        Show-ScanNetInfo
    }
    "all" {
        Download-TUM -DatasetsDir $DatasetsDir -SceneSelection ""
        Download-Replica -DatasetsDir $DatasetsDir
        Download-7Scenes -DatasetsDir $DatasetsDir -SceneSelection ""
        Download-IclNuim -DatasetsDir $DatasetsDir -SceneSelection ""
    }
    "list" {
        Show-SceneList
    }
    "help" {
        Show-Usage
    }
    default {
        Write-ErrorMsg "Unknown command: $Command"
        Show-Usage
        exit 1
    }
}

Write-Header "Done!"
Write-Host "Datasets directory: $DatasetsDir"
Write-Host ""
Write-Host "Current datasets:"
$dirs = Get-ChildItem $DatasetsDir -Directory -ErrorAction SilentlyContinue
if ($dirs) {
    $dirs | ForEach-Object { Write-Host "  $($_.Name)" }
}
else {
    Write-Host "  (none)"
}
