# NeuroVF

NeuroVF is a visual-field research application for cohort-level and region-specific analysis of standard automated perimetry data.

The application helps researchers import visual-field examinations, review the extracted values, organize examinations in a study database, and examine visual-field findings across groups of participants.

## Application Versions

The repository includes two versions of NeuroVF:

- `NeuroVF.py`: the standard application, which provides visual maps and analytical results.
- `NeuroVF_research.py`: the validation version, which also exports the numerical results used to verify NeuroVF's calculations against an independent oracle.

The research version is included for validation and reproducibility. It is not required for routine use of the standard application.

## Validation Data

The repository includes:

- 1,440 synthetic DICOM visual-field files;
- 18 reviewed databases created by importing the DICOM files through NeuroVF;
- validation and consistency tests; and
- an independent numerical oracle that recalculates the expected analytical results.

The reviewed databases cover different visual-field patterns, data types, and study configurations.

All included DICOM files, databases, clinical variables, and visual-field measurements are synthetic. The repository contains no patient information.

## Repository Structure

```text
app/                       NeuroVF.py and NeuroVF_research.py
data/dicom/                Synthetic DICOM files and digitizer settings
imported_databases/        18 reviewed DICOM/OCR-imported databases
oracle/                    Independent calculations and validation tools
tests/                     Quick application and oracle checks
docs/                      Validation documentation
run_validation_imported.bat
run_validation_imported.sh
requirements.txt
```

## Installation

Install the required Python packages from the repository folder:

```bat
python -m pip install -r requirements.txt
```

## Running NeuroVF

Start the standard application with:

```bat
python app\NeuroVF.py
```

## Numerical Validation

The validation workflow uses the 18 reviewed DICOM/OCR-imported databases as the common input for NeuroVF and the independent oracle.

On Windows, run:

```bat
run_validation_imported.bat
```

On Linux, run:

```sh
sh run_validation_imported.sh
```

The workflow exports NeuroVF's numerical results, independently recalculates the expected results, and compares the two outputs.

Detailed information about the validation workflow is available in `docs/VALIDATION.md`.

## Quick Checks

Run:

```bat
python tests\test_selfchecks.py
```

These checks compile both application versions, run the built-in consistency checks, and test selected oracle calculations.

## Important Notes

- All distributed data are synthetic.
- OCR-extracted values should be reviewed before use.
- Missing visual-field measurements are retained as missing and are not replaced with estimated values.
- Validation supports the computational correctness of the tested functions but does not establish clinical superiority or diagnostic accuracy.
- NeuroVF is designed as a research platform for cohort-level visual-field analysis and may complement conventional visual-field assessment.