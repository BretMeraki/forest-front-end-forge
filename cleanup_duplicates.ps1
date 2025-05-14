# PowerShell script for final cleanup of duplicated directories
Write-Host "Starting final cleanup of duplicate directories..."

$rootDir = "c:\Users\schlansk\Downloads\ForestFInal-main"
$nestedDir = "c:\Users\schlansk\Downloads\ForestFInal-main\ForestFInal-main"
$backupDir = "c:\Users\schlansk\Downloads\ForestFInal-main\forest_app_original_backup"

# Create a backup directory for safety
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupRoot = "c:\Users\schlansk\Downloads\ForestFInal-backup_$timestamp"
New-Item -Path $backupRoot -ItemType Directory -Force

# Backup the nested directory before removing
Write-Host "Creating safety backup at $backupRoot..."
Copy-Item -Path $nestedDir -Destination $backupRoot -Recurse -Force
Copy-Item -Path $backupDir -Destination $backupRoot -Recurse -Force

# Remove the nested ForestFInal-main directory since all files have been moved up
if (Test-Path $nestedDir) {
    Write-Host "Removing nested ForestFInal-main directory..."
    # Use a safer approach with Remove-Item rather than rm -rf
    Remove-Item -Path $nestedDir -Recurse -Force
}

# Remove the backup forest_app directory
if (Test-Path $backupDir) {
    Write-Host "Removing forest_app_original_backup directory..."
    Remove-Item -Path $backupDir -Recurse -Force
}

# Cleanup any __pycache__ directories for a fresh start
Write-Host "Cleaning up __pycache__ directories..."
Get-ChildItem -Path $rootDir -Filter "__pycache__" -Directory -Recurse | ForEach-Object {
    Write-Host "Removing $($_.FullName)"
    Remove-Item -Path $_.FullName -Recurse -Force
}

Write-Host "Final cleanup completed!"
Write-Host "A safety backup has been created at: $backupRoot"
