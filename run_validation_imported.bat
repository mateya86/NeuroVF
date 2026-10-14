@echo off
setlocal
set PYTHONUTF8=1

if not exist imported_databases\_database_index.json (
    echo ERROR: imported_databases\_database_index.json is missing.
    exit /b 1
)

python app\NeuroVF_research.py --export-dir imported_databases --out exports_imported
if errorlevel 1 exit /b 1

python oracle\run_oracle.py --databases imported_databases --exports exports_imported --dicom data\dicom --out oracle_report_imported
if errorlevel 1 exit /b 1

echo.
echo Oracle report: oracle_report_imported\oracle_report.txt
endlocal
