#!/bin/sh
set -eu

if [ ! -f imported_databases/_database_index.json ]; then
    echo "ERROR: imported_databases/_database_index.json is missing."
    exit 1
fi

python3 app/NeuroVF_research.py \
    --export-dir imported_databases \
    --out exports_imported

python3 oracle/run_oracle.py \
    --databases imported_databases \
    --exports exports_imported \
    --dicom data/dicom \
    --out oracle_report_imported

echo "Oracle report: oracle_report_imported/oracle_report.txt"
