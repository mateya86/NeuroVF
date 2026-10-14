# Validation

NeuroVF is evaluated against an independently implemented numerical oracle located in `oracle/`. The oracle recalculates the numerical results exported by the research build without importing or calling NeuroVF's analytical functions.

The analytical definitions are described in `oracle/RULES.md`, and the prespecified numerical tolerances are recorded in `oracle/tolerances.json`.

## Validation data

All validation data distributed in this repository are synthetic and contain no patient-derived information.

| Folder | Content |
|---|---|
| `data/dicom/files/` | 1,440 synthetic DICOM files covering 18 study configurations, with 20 subjects, two eyes, and two timepoints per configuration. |
| `data/dicom/grid_params/` | Digitizer-alignment files for the supported visual-field pattern and data-type combinations. |
| `imported_databases/` | Eighteen reviewed databases created by importing the synthetic DICOM files through NeuroVF's digitizer and manually reviewing the extracted grids. |
| `imported_databases/_database_index.json` | Index used by the research export and numerical oracle to identify the reviewed validation databases. |

The source numerical datasets used during development to assess DICOM/OCR import fidelity are not distributed in this public repository.

## Full numerical validation

Run the validation from the repository root.

### Windows

```bat
run_validation_imported.bat
```

### Linux

```sh
sh run_validation_imported.sh
```

The workflow:

1. reads the 18 reviewed DICOM/OCR-imported databases from `imported_databases/`;
2. generates NeuroVF numerical exports in `exports_imported/`;
3. independently recalculates the expected results using the oracle; and
4. writes the comparison reports to `oracle_report_imported/`.

The expected summary is:

```text
NeuroVF oracle report -- 131804 PASS, 0 FAIL, 0 SKIP
FAILURES (0)
```

Numerical values pass when:

```text
|NeuroVF - oracle| <= absolute tolerance + relative tolerance x |oracle|
```

Identifiers, labels, masks, counts, metadata, and missing-value patterns must match exactly.

## Quick self-checks

```bat
python tests\test_selfchecks.py
```

These checks:

- compile both applications;
- run NeuroVF's built-in consistency checks; and
- run the oracle self-test.

The quick self-checks do not replace the full numerical validation workflow.

## Checking the oracle

```bat
python oracle\selftest_oracle.py
```

The oracle self-test checks selected calculations against independently calculated examples, mathematical identities, simulations, and SciPy reference results.

## Mutation self-check

Mutation testing is performed on a completed NeuroVF numerical export:

```bat
python oracle\mutation_selfcheck.py --databases imported_databases --export exports_imported\neurovf_ptc_30-2_total_deviation
```

The mutation self-check deliberately alters numerical values, masks, counts, labels, metadata, and statistical results in a copy of the export. The completed validation used 36 predefined alterations, all of which were detected.

## Interpretation

Agreement with the oracle supports correct and reproducible implementation of the prespecified analytical rules. It does not establish that the registration, complete-case, or statistical policies are clinically optimal.

The reviewed databases were created through DICOM/OCR import followed by manual quality control. The public validation therefore evaluates NeuroVF's calculations from the reviewed measurements and should not be interpreted as a test of fully automated OCR accuracy.

All validation datasets are synthetic test data. They are intended to test software implementation and must not be presented as patient-derived or clinical data.

## Important 24-2 note

In the 24-2 pattern, two nasal locations at 27 degrees on the horizontal meridian have no corresponding mirror-image locations within the same grid. NeuroVF retains their measured values without remapping them. Results at these locations should not be interpreted as anatomically registered across eyes or lesion sides.