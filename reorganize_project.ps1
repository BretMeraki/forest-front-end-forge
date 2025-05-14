# PowerShell script to reorganize Forest project structure
Write-Host "Starting project reorganization..."

$rootDir = "c:\Users\schlansk\Downloads\ForestFInal-main"
$nestedDir = "c:\Users\schlansk\Downloads\ForestFInal-main\ForestFInal-main"

# Step 1: Move core files from nested directory to root
# Copy root level files (excluding directories)
Write-Host "Moving core files from nested directory to root..."
Get-ChildItem -Path $nestedDir -File | ForEach-Object {
    $targetPath = Join-Path -Path $rootDir -ChildPath $_.Name
    if (-not (Test-Path $targetPath)) {
        Write-Host "Copying $($_.Name) to root..."
        Copy-Item -Path $_.FullName -Destination $rootDir
    } else {
        Write-Host "File $($_.Name) already exists in root, skipping..."
    }
}

# Step 2: Handle forest_app directory
# The nested forest_app directory seems more complete, so let's replace the root one
$rootForestApp = Join-Path -Path $rootDir -ChildPath "forest_app"
$nestedForestApp = Join-Path -Path $nestedDir -ChildPath "forest_app"

if (Test-Path $rootForestApp) {
    Write-Host "Backing up original forest_app directory..."
    Rename-Item -Path $rootForestApp -NewName "forest_app_original_backup"
}

Write-Host "Copying complete forest_app from nested directory..."
Copy-Item -Path $nestedForestApp -Destination $rootDir -Recurse

# Step 3: Copy other directories not present in root
Get-ChildItem -Path $nestedDir -Directory | Where-Object { $_.Name -ne "forest_app" -and $_.Name -ne "tasks" } | ForEach-Object {
    $targetPath = Join-Path -Path $rootDir -ChildPath $_.Name
    if (-not (Test-Path $targetPath)) {
        Write-Host "Copying directory $($_.Name) to root..."
        Copy-Item -Path $_.FullName -Destination $rootDir -Recurse
    } else {
        Write-Host "Directory $($_.Name) already exists in root, checking for unique files..."
        # Here we could implement deeper merging if needed
    }
}

Write-Host "Project reorganization completed!"
Write-Host "Please verify the structure manually before removing any backup directories."
