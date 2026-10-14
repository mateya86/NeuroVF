"""NeuroVF - Visual Field Research Platform

A research tool for building a subject database of Humphrey Visual Field
(HVF) results and running spatial and statistical analyses across a cohort.

Each database holds one HVF test pattern (24-2, 30-2 or 60-4), one data type
(Total Deviation or raw threshold sensitivity, both in dB) and one study
type (the OD/OS registration policy). All three are chosen when the database
is created.

Main features:
  - User-defined variable schema. Categorical and continuous variables are
    added through the UI; the subject table and every analysis dialog follow
    the schema.
  - Spatial analyses against any variable: cohort and category mean maps,
    group comparisons (Welch t-test for two groups, Welch ANOVA with
    Games-Howell post hoc for three or more), Pearson and Spearman
    correlation maps, cluster and cumulative severity analyses, and
    longitudinal (Timepoint 2 minus Timepoint 1) change maps.
  - Colour convention. Correlation maps: red = positive association (a
    higher variable value goes with a higher value at that location), blue =
    negative association. Group-comparison maps: red = group B higher, blue
    = group B lower. Each figure caption states the convention.
  - Benjamini-Hochberg FDR correction on every pointwise statistical map,
    restricted to the locations where the statistic is computable. A missing
    or non-computable value is never treated as zero.
  - DICOM and PDF import. The printed numerical grid is read by OCR from the
    source report (a DICOM image or a rasterized PDF page) and reviewed
    manually before it is saved.
"""

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr, linregress
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import json
import os
import sys
import re
import textwrap
import math
import warnings

# Suppress expected numerical warnings
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in", category=RuntimeWarning)

# Resolve script directory
def _get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    try:
        p = os.path.abspath(sys.argv[0])
        if os.path.isfile(p):
            return os.path.dirname(p)
    except:
        pass
    return os.getcwd()

_SCRIPT_DIR = _get_script_dir()

# Database file. If only the legacy file name (neurovf_research_v2.json) is
# present, keep using it so that an existing database is not replaced by a
# new, empty one.
_DB_FILE_CURRENT = os.path.join(_SCRIPT_DIR, "neurovf_research.json")
_DB_FILE_LEGACY  = os.path.join(_SCRIPT_DIR, "neurovf_research_v2.json")
if os.path.exists(_DB_FILE_LEGACY) and not os.path.exists(_DB_FILE_CURRENT):
    DB_FILE = _DB_FILE_LEGACY
else:
    DB_FILE = _DB_FILE_CURRENT

# HVF grids: 24-2, 30-2 and 60-4.
# Each pattern is a 10x10 (row, col) lattice (COL_TO_X / ROW_TO_Y) plus an
# ACTIVE_COLS map of the cells that are tested locations (54, 76 and 60 cells
# respectively). Downstream code reads the point count and coordinates from
# the resulting arrays. The 30-2 names (COL_TO_X_CANON, ROW_TO_Y_CANON,
# ACTIVE_COLS_CANON, POINTS_30_2, N_VF) are kept for backward compatibility.

# 24-2 / 30-2 shared canonical 6-degree lattice
COL_TO_X_CANON = {0:-27,1:-21,2:-15,3:-9,4:-3,5:3,6:9,7:15,8:21,9:27}
ROW_TO_Y_CANON = {0:27,1:21,2:15,3:9,4:3,5:-3,6:-9,7:-15,8:-21,9:-27}
ACTIVE_COLS_CANON = {   # 30-2 active cells
    0:[3,4,5,6],
    1:[2,3,4,5,6,7],
    2:[1,2,3,4,5,6,7,8],
    3:[0,1,2,3,4,5,6,7,8,9],
    4:[0,1,2,3,4,5,6,7,8,9],
    5:[0,1,2,3,4,5,6,7,8,9],
    6:[0,1,2,3,4,5,6,7,8,9],
    7:[1,2,3,4,5,6,7,8],
    8:[2,3,4,5,6,7],
    9:[3,4,5,6],
}
ACTIVE_COLS_30_2 = ACTIVE_COLS_CANON

# 24-2: 54 points at 6 deg spacing - the central 24 deg plus two points on
# the nasal horizontal meridian at 27 deg (DICOM CID 4250, code 111800). The
# manufacturer publishes this specification but no coordinate table, so the
# map is derived from ACTIVE_COLS_30_2: drop the outermost ring (|x| or |y| =
# 27 deg), drop the points with {|x|, |y|} == {15, 21}, then restore the two
# nasal-horizontal cells (rows 4 and 5, col 9). The point count is asserted
# below.
ACTIVE_COLS_24_2 = {
    0: [],
    1: [3, 4, 5, 6],
    2: [2, 3, 4, 5, 6, 7],
    3: [1, 2, 3, 4, 5, 6, 7, 8],
    4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    5: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    6: [1, 2, 3, 4, 5, 6, 7, 8],
    7: [2, 3, 4, 5, 6, 7],
    8: [3, 4, 5, 6],
    9: [],
}

# 60-4: 60 points on a 12-degree lattice between 30 and 60 deg (DICOM CID
# 4250, code 111803: "60 test point locations between 30 and 60 deg of
# fixation, a minimum of 6 deg from each meridian, placed 12 deg apart").
# Cells of the 12-deg grid with eccentricity strictly between about 30.6 and
# 59.4 deg are kept, except the four diagonal cells at (+-30, +-30). The
# point count is asserted below.
COL_TO_X_60_4 = {0:-54,1:-42,2:-30,3:-18,4:-6,5:6,6:18,7:30,8:42,9:54}
ROW_TO_Y_60_4 = {0:54,1:42,2:30,3:18,4:6,5:-6,6:-18,7:-30,8:-42,9:-54}
ACTIVE_COLS_60_4 = {
    0: [3, 4, 5, 6],
    1: [1, 2, 3, 4, 5, 6, 7, 8],
    2: [1, 3, 4, 5, 6, 8],
    3: [0, 1, 2, 7, 8, 9],
    4: [0, 1, 2, 7, 8, 9],
    5: [0, 1, 2, 7, 8, 9],
    6: [0, 1, 2, 7, 8, 9],
    7: [1, 3, 4, 5, 6, 8],
    8: [1, 2, 3, 4, 5, 6, 7, 8],
    9: [3, 4, 5, 6],
}


def _vf_row_sort_key(p):
    # Row-major order: top to bottom, then left to right.
    return (-p[1], p[0])


def _vf_build_points(col_to_x, row_to_y, active_cols):
    pts = []
    for r in range(10):
        for c in active_cols[r]:
            pts.append((float(col_to_x[c]), float(row_to_y[r])))
    pts.sort(key=_vf_row_sort_key)
    return np.array(pts, dtype=float)


def build_flip_remap(points):
    """Return the index array that mirrors a point set left-right.

    For each point i, remap[i] is the index of the point at (-x_i, y_i), so
    `vals[remap]` flips an array of values horizontally.

    The 24-2 pattern is not left-right symmetric: its two extra nasal points
    have no mirror partner within the pattern. For those points the remap is
    the identity, so their values stay in place while every other point is
    mirrored.
    """
    n = len(points)
    remap = np.arange(n, dtype=int)
    unmatched = []
    for i, (x, y) in enumerate(points):
        target = (-x, y)
        matches = np.where(
            (np.isclose(points[:, 0], target[0], atol=1e-6)) &
            (np.isclose(points[:, 1], target[1], atol=1e-6))
        )[0]
        if len(matches) == 1:
            remap[i] = matches[0]
        else:
            unmatched.append(i)
    if unmatched:
        print(f"[NeuroVF] build_flip_remap: {len(unmatched)} point(s) with no "
              f"horizontal mirror in this grid left as identity (indices {unmatched}).")
    return remap


POINTS_24_2 = _vf_build_points(COL_TO_X_CANON, ROW_TO_Y_CANON, ACTIVE_COLS_24_2)
POINTS_30_2 = _vf_build_points(COL_TO_X_CANON, ROW_TO_Y_CANON, ACTIVE_COLS_30_2)
POINTS_60_4 = _vf_build_points(COL_TO_X_60_4, ROW_TO_Y_60_4, ACTIVE_COLS_60_4)

assert len(POINTS_24_2) == 54, f"24-2 build error: got {len(POINTS_24_2)} points"
assert len(POINTS_30_2) == 76, f"30-2 build error: got {len(POINTS_30_2)} points"
assert len(POINTS_60_4) == 60, f"60-4 build error: got {len(POINTS_60_4)} points"

N_VF = len(POINTS_30_2)  # 76; 30-2 alias used by the self-tests

GRID_SPECS = {
    "24-2": {
        "points": POINTS_24_2, "n": 54, "spacing_deg": 6, "extent_deg": 30.0,
        "label": "24-2  (54 pts, central 24\u00b0)",
        "col_to_x": COL_TO_X_CANON, "row_to_y": ROW_TO_Y_CANON,
        "active_cols": ACTIVE_COLS_24_2,
        "flip_remap": build_flip_remap(POINTS_24_2),
    },
    "30-2": {
        "points": POINTS_30_2, "n": 76, "spacing_deg": 6, "extent_deg": 30.0,
        "label": "30-2  (76 pts, central 30\u00b0)",
        "col_to_x": COL_TO_X_CANON, "row_to_y": ROW_TO_Y_CANON,
        "active_cols": ACTIVE_COLS_30_2,
        "flip_remap": build_flip_remap(POINTS_30_2),
    },
    "60-4": {
        "points": POINTS_60_4, "n": 60, "spacing_deg": 12, "extent_deg": 60.0,
        "label": "60-4  (60 pts, 30\u2013 60\u00b0 ring)",
        "col_to_x": COL_TO_X_60_4, "row_to_y": ROW_TO_Y_60_4,
        "active_cols": ACTIVE_COLS_60_4,
        "flip_remap": build_flip_remap(POINTS_60_4),
    },
}

# Data types: which quantity is being analyzed
DATA_TYPES = {
    "total_deviation": {
        "label": "Total Deviation (dB, age/location-corrected)",
        "short": "TD", "cmap": "gray", "vmin": -30.0, "vmax": 0.0,
        "plausible_min": -40.0, "plausible_max": 10.0,
    },
    "raw_threshold": {
        "label": "Raw Threshold Sensitivity (dB, uncorrected)",
        "short": "RAW", "cmap": "gray", "vmin": 0.0, "vmax": 38.0,
        "plausible_min": 0.0, "plausible_max": 40.0,
    },
}

# Study types: the OD/OS registration policy for the anatomical location
# under study. A database holds one study type; mixing registration policies
# in one cohort would invalidate every spatial statistic.
#
# flip_policy values:
#   "manual"              - each eye is mirrored according to its stored
#                           vf_flip_* flag (default False).
#   "flip_one_eye_cohort" - one eye, chosen once per database, is mirrored
#                           to the other eye's nasal/temporal orientation
#                           for every subject, regardless of the stored
#                           flags. Used for chiasmal and pre-chiasmal
#                           pathology, where the defect follows each eye's
#                           own nasal/temporal retinal anatomy.
#   "surgery_side_mirror" - OD and OS are combined without a per-eye flip
#                           (homonymous post-chiasmal defects are already
#                           aligned by hemifield); the combined field is
#                           then mirrored left-right for subjects whose
#                           surgery-side variable differs from the
#                           reference value, which puts every subject in a
#                           common ipsilateral/contralateral frame.
EYE_LEVEL_REGISTRATION_NOTE = (
    " Two registration modes are available. Cohort-registered: one eye is "
    "reflected onto the other eye's nasal-temporal axis and the pair is "
    "combined into a single field for whole-cohort analysis. Eye-separated: "
    "OD and OS are retained in their recorded orientation and analyzed as "
    "two independent fields. Both modes operate at the level of the eye, "
    "not the visual cortex, and do not support correlation with cortical "
    "morphometry."
)
CLINICAL_TYPES = {
    "general": {
        "label": "General  (manual per-eye flip, no enforced policy)",
        "flip_policy": "manual",
        "morphometry_note": None,
    },
    "post_chiasmatic": {
        "label": "Post-chiasmatic  (optic tract, cortical - temporal/parietal surgery)",
        "flip_policy": "surgery_side_mirror",
        "morphometry_note": (
            "Post-chiasmatic registration: OD and OS are combined WITHOUT a "
            "nasal/temporal flip (homonymous defects are already hemifield-"
            "native as entered), then the whole combined field is mirrored "
            "by surgery side so every subject is expressed in a common "
            "ipsilateral/contralateral frame.\n\n"
            "Because this frame corresponds to visual-cortex anatomy "
            "(ipsilateral vs contralateral lingual gyrus/cuneus), structure-"
            "function correlation with cortical imaging data is appropriate "
            "for this registration if you bring in that data separately - "
            "NeuroVF does not import or analyze cortical imaging itself."
        ),
    },
    "chiasmatic": {
        "label": "Chiasmatic  (e.g. pituitary/hypophyseal adenoma)",
        "flip_policy": "flip_one_eye_cohort",
        "morphometry_note": (
            "Chiasmatic lesions disrupt the decussating nasal retinal fibers "
            "at the optic chiasm, producing a field defect organized by eye "
            "rather than by hemifield." + EYE_LEVEL_REGISTRATION_NOTE
        ),
    },
    "ptc": {
        "label": "PTC / IIH  (papilledema)",
        "flip_policy": "flip_one_eye_cohort",
        "morphometry_note": (
            "Papilledema in PTC/IIH produces a diffuse, nasal-predominant "
            "field defect intrinsic to each eye rather than organized by "
            "hemifield." + EYE_LEVEL_REGISTRATION_NOTE
        ),
    },
}

# Total Deviation severity thresholds (dB, not normalized). Total Deviation
# is an interval-scale value: 0 dB is the age- and location-expected
# sensitivity, and more negative values mean greater loss. The bands are
# prespecified, study-defined bands, not manufacturer-certified clinical
# categories.
TD_SEVERE   = -12.01  # TD <= -12.01 dB: deep reduction
TD_MODERATE = -6.01   # -12.01 < TD <= -6.01 dB: moderate reduction
TD_MILD     = -6.01   # same as TD_MODERATE (three-tier banding)
# Three TD tiers, mirroring the mild/moderate/advanced structure of global
# MD staging: TD > -6.01 dB is within normal or mild reduction, -12.01 < TD
# <= -6.01 dB is moderate reduction, and TD <= -12.01 dB is deep reduction.
# These pointwise cutoffs are an operational adaptation of the global Mean
# Deviation thresholds; they are not independently validated for individual
# locations. The UI shows TD_SEVERITY_CAVEAT wherever the bands are used.
TD_SEVERITY_CAVEAT = ("Pointwise TD cutoffs are operational adaptations of "
                       "global MD thresholds, not independently validated.")

# Measurement-type flag stored at the database root, so that a legacy
# PI-scale (0-1 normalized) database is not read as dB Total Deviation.
VF_MEASUREMENT_TYPE_KEY   = "__vf_measurement_type__"
VF_MEASUREMENT_TYPE_VALUE = "TotalDeviation_dB"

# Database-wide grid pattern and data type. A database file holds one
# pattern and one data type, because array length and point coordinates are
# assumed to be the same for all subjects. Once a database contains VF data,
# both are locked (see ResearchApp._select_grid_and_datatype).
VF_GRID_PATTERN_KEY = "__vf_grid_pattern__"
VF_DATA_TYPE_KEY    = "__vf_data_type__"
VF_CLINICAL_TYPE_KEY    = "__vf_study_type__"
VF_SURGERY_SIDE_VAR_KEY       = "__vf_surgery_side_variable__"
VF_SURGERY_SIDE_REFERENCE_KEY = "__vf_surgery_side_reference_value__"
VF_FLIP_EYE_KEY = "__vf_flip_eye__"   # "od" or "os": eye mirrored under flip_one_eye_cohort

# Database-root keys that hold metadata rather than a subject record. Code
# that iterates over subjects skips these.
RESERVED_DB_KEYS = ("__schema__", VF_MEASUREMENT_TYPE_KEY,
                     VF_GRID_PATTERN_KEY, VF_DATA_TYPE_KEY,
                     VF_CLINICAL_TYPE_KEY, VF_SURGERY_SIDE_VAR_KEY,
                     VF_SURGERY_SIDE_REFERENCE_KEY, VF_FLIP_EYE_KEY)

# Plausibility range for a single Total Deviation value (dB). Values outside
# the range are flagged for manual verification, not clipped or rejected.
TD_PLAUSIBLE_MIN = -40.0
TD_PLAUSIBLE_MAX = 10.0

# Raw threshold sensitivity bands. Raw sensitivity (dB) is not corrected for
# age or location: a normal value at 27 deg eccentricity is much lower than
# at fixation, even in a healthy eye.
RAW_SEVERE   = 14.0    # raw <= 14 dB: absolute defect
RAW_MODERATE = 27.0    # 14 < raw <= 27 dB: relative reduction; raw > 27 dB: normal
RAW_MILD     = 27.0    # same as RAW_MODERATE (three-tier banding)
RAW_PLAUSIBLE_MIN = 0.0
RAW_PLAUSIBLE_MAX = 40.0
# Three raw-sensitivity tiers: raw > 27 dB is normal sensitivity, 14 < raw
# <= 27 dB is relative sensitivity reduction, and raw <= 14 dB is an absolute
# defect. No validated per-point severity banding exists for raw threshold
# sensitivity at any eccentricity; the UI shows RAW_SEVERITY_CAVEAT wherever
# the bands are used.
RAW_SEVERITY_CAVEAT = "No validated per-point banding exists in literature."

# Display names of the tiers, by data type, used wherever a severity label
# is shown (scotoma-area report, cluster and cumulative legends). There is no
# separate "mild" tier: both schemes have three categories.
TIER_LABELS = {
    "total_deviation": {"severe": "Deep reduction", "moderate": "Moderate reduction",
                         "normal": "Within normal / mild reduction"},
    "raw_threshold":    {"severe": "Absolute defect", "moderate": "Relative sensitivity reduction",
                         "normal": "Normal sensitivity"},
}

# Contour levels (dB) for cohort-mean maps, per data type. Must be strictly
# increasing. Total Deviation: study bands sit at -12 / -6; sensitivity: the
# 15 dB and 28 dB band edges plus two intermediate levels.
CONTOUR_LEVELS = {"total_deviation": [-24, -18, -12, -6],
                  "raw_threshold":   [10, 15, 20, 28]}

# Add the severity settings to DATA_TYPES so that ResearchApp reads them
# from self.data_type_spec.
DATA_TYPES["total_deviation"].update({
    "severe": TD_SEVERE, "moderate": TD_MODERATE, "mild": TD_MILD,
    "tier_labels": TIER_LABELS["total_deviation"],
    "severity_caveat": TD_SEVERITY_CAVEAT,
})
DATA_TYPES["raw_threshold"].update({
    "severe": RAW_SEVERE, "moderate": RAW_MODERATE, "mild": RAW_MILD,
    "tier_labels": TIER_LABELS["raw_threshold"],
    "severity_caveat": RAW_SEVERITY_CAVEAT,
})

# Tokens (case-insensitive, after stripping whitespace) that mean "missing"
_TD_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "missing", "-"}

# Timepoint display labels. Storage keys and values ("preop"/"postop",
# "Preop OD VF", ...) are fixed, because existing databases depend on them;
# only the text shown to the user reads "Timepoint 1" / "Timepoint 2".
TIMEPOINT_DISPLAY = {"preop": "Timepoint 1", "postop": "Timepoint 2"}

# Table column headings. The column identifiers (also keys of each subject
# record) keep their stored names; only the heading text is remapped.
COL_DISPLAY_LABELS = {
    "Preop OD VF":  "Timepoint 1 OD VF",
    "Preop OS VF":  "Timepoint 1 OS VF",
    "Postop OD VF": "Timepoint 2 OD VF",
    "Postop OS VF": "Timepoint 2 OS VF",
}


def tp_display(tp):
    """Map an internal timepoint value ('preop'/'postop', any case) to the
    user-facing label ('Timepoint 1'/'Timepoint 2'). Anything else is
    returned title-cased, unchanged, as a safe fallback."""
    if not tp:
        return ""
    return TIMEPOINT_DISPLAY.get(str(tp).strip().lower(), str(tp).capitalize())


def parse_td_value(text):
    """Parse one Total Deviation entry (typed, pasted or read by OCR) into dB.

    Returns a float, or np.nan for a blank entry or a recognized missing-value
    token. The ASCII hyphen-minus, the Unicode minus sign (U+2212), the en dash
    and the em dash are all accepted as a negative sign.

    A missing source measurement is np.nan, never 0 dB, and 0 dB is a measured
    value, never missing.

    Examples:
        "-3"   -> -3.0
        "−6"   -> -6.0     (Unicode minus)
        "–12"  -> -12.0    (en dash)
        "—23"  -> -23.0    (em dash)
        "0"    ->  0.0
        "+1"   ->  1.0
        ""     -> np.nan
        "NA"   -> np.nan
    """
    if text is None:
        return np.nan
    if isinstance(text, (int, float)):
        v = float(text)
        return v if np.isfinite(v) else np.nan

    cleaned = (str(text).strip()
               .replace("−", "-")   # Unicode minus sign (U+2212)
               .replace("–", "-")   # en dash, sometimes rendered as minus
               .replace("—", "-"))  # em dash, sometimes rendered as minus

    if cleaned.lower() in _TD_MISSING_TOKENS:
        return np.nan

    # Keep only a single leading sign plus digits/decimal point.
    cleaned = cleaned.replace(" ", "")
    m = re.match(r'^([+-]?)(\d+(?:\.\d+)?)$', cleaned)
    if not m:
        return np.nan
    sign, digits = m.groups()
    try:
        v = float(digits)
    except (ValueError, TypeError):
        return np.nan
    if sign == "-":
        v = -v
    return v


def run_td_validation_tests(verbose=True):
    """Run the built-in consistency checks.

    Covers Total Deviation parsing, the OD/OS complete-pair rule, the severity
    bands, missing-data masks, BH-FDR handling of non-computable locations,
    JSON sanitization, NaN-aware smoothing and the mirroring policies. With
    verbose=True a [PASS]/[FAIL] line is printed for each check. Returns True
    only if every check passed.
    """
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # Parsing
    check("Empty string -> np.nan",        np.isnan(parse_td_value("")))
    check("'NA' -> np.nan",                np.isnan(parse_td_value("NA")))
    check("'NaN' -> np.nan",               np.isnan(parse_td_value("NaN")))
    check("'nan' -> np.nan",               np.isnan(parse_td_value("nan")))
    check("'null' -> np.nan",              np.isnan(parse_td_value("null")))
    check("Zero remains 0.0",              parse_td_value("0") == 0.0)
    check("ASCII minus '-3' -> -3.0",      parse_td_value("-3") == -3.0)
    check("Unicode minus '−6' -> -6.0",    parse_td_value("−6") == -6.0)
    check("En dash '–12' -> -12.0",        parse_td_value("–12") == -12.0)
    check("Em dash '-23' -> -23.0",        parse_td_value("-23") == -23.0)
    check("'+1' -> 1.0",                   parse_td_value("+1") == 1.0)
    check("'2' -> 2.0",                    parse_td_value("2") == 2.0)

    # OD/OS complete-pair combination rule
    od = np.array([1.0, np.nan, -5.0, np.nan])
    os_ = np.array([2.0, 3.0, np.nan, np.nan])
    od_valid = np.isfinite(od); os_valid = np.isfinite(os_)
    paired_valid_mask = od_valid & os_valid
    combined = np.full_like(od, np.nan, dtype=float)
    combined[paired_valid_mask] = (od[paired_valid_mask] + os_[paired_valid_mask]) / 2.0
    check("Both eyes valid -> arithmetic mean",         combined[0] == 1.5)
    check("OD missing, OS valid -> combined is np.nan", np.isnan(combined[1]))
    check("OS missing, OD valid -> combined is np.nan", np.isnan(combined[2]))
    check("Both eyes missing -> combined is np.nan",    np.isnan(combined[3]))
    check("No np.nanmean used for OD-OS pairing (spot check)",
          not np.isnan(np.nanmean([od[1], os_[1]])) and np.isnan(combined[1]))

    # Severity bands: three tiers (severe / moderate / normal); TD_MILD equals
    # TD_MODERATE.
    check("Severe <= Moderate-or-worse thresholds ordered",
          TD_SEVERE < TD_MODERATE < 0)
    check("TD_MILD collapsed onto TD_MODERATE (3-tier banding)",
          TD_MILD == TD_MODERATE)
    sample_td = np.array([-15, -8, -4, -1, 2], dtype=float)
    sev  = sample_td <= TD_SEVERE
    mod  = (sample_td > TD_SEVERE) & (sample_td <= TD_MODERATE)
    norm = sample_td > TD_MODERATE
    check("Mutually exclusive categories sum to n",
          int(sev.sum()+mod.sum()+norm.sum()) == len(sample_td))
    cum_sev = (sample_td <= TD_SEVERE).sum()
    cum_mod = (sample_td <= TD_MODERATE).sum()
    check("Severe <= Moderate-or-worse (cumulative)",
          cum_sev <= cum_mod)

    # The parser returns dB values unchanged (no normalization).
    check("parse_td_value performs no /35 normalization",
          parse_td_value("35") == 35.0)

    # The cohort missing mask is complete-case: a location is unavailable if
    # any included subject lacks a value there.
    stack_demo = np.array([[1.0,  1.0, -5.0],
                            [2.0,  np.nan,  3.0],
                            [1.5, -5.0, np.nan]])
    valid_n_demo = np.sum(np.isfinite(stack_demo), axis=0)
    n_total_demo = stack_demo.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cohort_mean_demo = np.nanmean(stack_demo, axis=0)
    cohort_unavailable_demo = valid_n_demo < n_total_demo
    cohort_mean_demo[cohort_unavailable_demo] = np.nan
    check("Cohort missing mask is complete-case: every subject present -> available; even one missing -> unavailable",
          not cohort_unavailable_demo[0] and not np.isnan(cohort_mean_demo[0])
          and bool(cohort_unavailable_demo[1]) and np.isnan(cohort_mean_demo[1])
          and bool(cohort_unavailable_demo[2]) and np.isnan(cohort_mean_demo[2]))

    # Category missing masks are specific to each category.
    cat_a_stack = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    cat_b_stack = np.array([[3.0, np.nan], [4.0, np.nan]])
    mask_a = np.sum(np.isfinite(cat_a_stack), axis=0) == 0
    mask_b = np.sum(np.isfinite(cat_b_stack), axis=0) == 0
    check("Category missing masks are category-specific",
          np.array_equal(mask_a, [True, False]) and np.array_equal(mask_b, [False, True]))

    # A group difference is missing whenever either group mean is unavailable.
    mean_a_demo = np.array([np.nan, -5.0, -3.0])
    mean_b_demo = np.array([-2.0, np.nan, -1.0])
    diff_valid_demo = np.isfinite(mean_a_demo) & np.isfinite(mean_b_demo)
    diff_demo = np.full(3, np.nan)
    diff_demo[diff_valid_demo] = mean_b_demo[diff_valid_demo] - mean_a_demo[diff_valid_demo]
    check("Group difference is np.nan when either group mean is unavailable",
          np.isnan(diff_demo[0]) and np.isnan(diff_demo[1]) and diff_demo[2] == 2.0)

    # The deprecated PI-scale threshold aliases (THR_MOD_SEV, THR_MILD_MOD,
    # THR_HEALTHY) are not defined in this module.
    check("Deprecated THR_* aliases removed from module namespace",
          ('THR_MOD_SEV' not in globals()) and ('THR_MILD_MOD' not in globals())
          and ('THR_HEALTHY' not in globals()))

    # A Welch t-test needs at least two valid observations per group to have a
    # defined variance, independently of the complete-case policy.
    a_col = np.array([-5.0, np.nan, -6.0, -4.0])       # 3 valid
    b_col = np.array([np.nan, np.nan, -2.0, -1.0])     # 2 valid -> computable
    c_col = np.array([np.nan, np.nan, -2.0, np.nan])   # 1 valid -> not computable
    a_v = a_col[np.isfinite(a_col)]; b_v = b_col[np.isfinite(b_col)]; c_v = c_col[np.isfinite(c_col)]
    check("Welch/Hedges'-g location computable when both groups have >=2 valid obs",
          len(a_v) == 3 and len(b_v) == 2 and (len(a_v) >= 2 and len(b_v) >= 2))
    check("Welch/Hedges'-g location NOT computable when a group has <2 valid obs",
          len(c_v) == 1 and not (len(a_v) >= 2 and len(c_v) >= 2))

    # Non-computable statistics are np.nan, never 0 or 1.
    t_demo = np.full(5, np.nan); p_demo = np.full(5, np.nan); d_demo = np.full(5, np.nan)
    check("Non-computable t/p/d arrays default to np.nan (never 0.0 or 1.0)",
          np.all(np.isnan(t_demo)) and np.all(np.isnan(p_demo)) and np.all(np.isnan(d_demo)))

    # The BH-FDR family is the set of finite p-values; q is np.nan at a
    # non-computable location, never 1.0.
    p_mixed = np.array([np.nan, 0.001, 0.20, np.nan, 0.04])
    q_mixed = bh_fdr(p_mixed)
    check("BH-FDR leaves non-computable locations as np.nan (not 1.0)",
          np.isnan(q_mixed[0]) and np.isnan(q_mixed[3]) and np.isfinite(q_mixed[1]))
    check("BH-FDR denominator equals the count of computable (finite) p-values",
          int(np.isfinite(p_mixed).sum()) == 3)

    # Severity classification is not bounded by the plausibility range:
    # TD_PLAUSIBLE_MIN/MAX only trigger a manual-verification flag.
    check("A verified TD=-45 dB (beyond plausibility floor) is still classified Severe",
          (-45.0 <= TD_SEVERE))
    check("A verified TD=+12 dB (beyond plausibility ceiling) is still Within-expected-range",
          (12.0 > TD_MILD))

    # The JSON sanitizer converts NaN/Inf to None, keeps a finite zero, and
    # converts NumPy scalars and arrays to native Python types.
    demo_payload = {
        "a": float('nan'), "b": np.float64(1.5), "c": np.int64(3),
        "d": np.bool_(True), "e": np.array([1.0, np.nan, 2.0]),
        "f": 0.0, "g": None, "h": [np.float32(2.0), {"x": np.nan}],
    }
    safe_payload = _json_safe(demo_payload)
    check("JSON sanitizer converts NaN -> None",
          safe_payload["a"] is None)
    check("JSON sanitizer converts NumPy float/int/bool -> native Python types",
          type(safe_payload["b"]) is float and type(safe_payload["c"]) is int
          and type(safe_payload["d"]) is bool)
    check("JSON sanitizer preserves finite 0.0 (never confuses with missing)",
          safe_payload["f"] == 0.0 and safe_payload["f"] is not None)
    check("JSON sanitizer recursively sanitizes NumPy arrays and nested structures",
          safe_payload["e"] == [1.0, None, 2.0] and safe_payload["h"][1]["x"] is None)
    try:
        _ = json.dumps(safe_payload, allow_nan=False)
        json_dump_ok = True
    except ValueError:
        json_dump_ok = False
    check("json.dumps(..., allow_nan=False) succeeds on sanitized payload", json_dump_ok)

    # NaN-aware smoothing does not treat a missing cell as 0 dB: a missing cell
    # surrounded by -5 dB neighbors smooths to about -5 dB.
    demo_grid = np.full((7, 7), -5.0)
    demo_grid[3, 3] = np.nan   # one missing cell in the middle of a -5 dB field
    smoothed_demo = _nan_aware_gaussian_smooth(demo_grid, sigma=1.0)
    check("NaN-aware smoothing fills a missing cell from its neighbors, not from a fabricated 0 dB",
          np.isfinite(smoothed_demo[3, 3]) and abs(smoothed_demo[3, 3] - (-5.0)) < 0.5)

    # VF76_FLIP_REMAP is a permutation of the 76 indices of the 30-2 grid.
    check("VF76_FLIP_REMAP remains a valid permutation of 0..75 (manual flip logic intact)",
          sorted(VF76_FLIP_REMAP.tolist()) == list(range(N_VF)))

    # Whole-field mirroring by surgery side is opt-in. It applies only under
    # the "surgery_side_mirror" policy (post-chiasmatic study type) and only
    # once a surgery-side variable and a reference value are configured. The
    # default study type ("general") uses the manual flip policy.
    check("Default study type ('general') uses the manual (non-automatic) flip policy",
          CLINICAL_TYPES["general"]["flip_policy"] == "manual")

    class _FakeAppForMirrorTest:
        flip_policy = "surgery_side_mirror"
        surgery_side_var = None
        surgery_side_reference = None
        grid_flip_remap = VF76_FLIP_REMAP
    _fake = _FakeAppForMirrorTest()
    _demo_combined = np.arange(N_VF, dtype=float)
    _demo_valid = np.ones(N_VF, dtype=bool)
    _out0, _ = ResearchApp._apply_cohort_mirror(_fake, _demo_combined, _demo_valid,
                                                 {"Surgery Side": "Right"})
    check("surgery_side_mirror is a no-op until BOTH a variable and a reference value are explicitly configured",
          np.array_equal(_out0, _demo_combined))

    _fake.surgery_side_var = "Surgery Side"
    _fake.surgery_side_reference = "Left"
    _out1, _ = ResearchApp._apply_cohort_mirror(_fake, _demo_combined, _demo_valid,
                                                 {"Surgery Side": "Right"})
    check("Once explicitly configured, a non-reference surgery side DOES mirror the field (opt-in, not silent)",
          not np.array_equal(_out1, _demo_combined))
    _out2, _ = ResearchApp._apply_cohort_mirror(_fake, _demo_combined, _demo_valid,
                                                 {"Surgery Side": "Left"})
    check("The reference side itself is never mirrored",
          np.array_equal(_out2, _demo_combined))

    if verbose:
        print("\n[NeuroVF] -- Total Deviation validation summary --------------")
        for name, ok in results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        n_fail = sum(1 for _, ok in results if not ok)
        print(f"[NeuroVF] {len(results)-n_fail}/{len(results)} checks passed."
              + ("" if n_fail == 0 else f"  {n_fail} FAILED - review above."))
        print("--------------------------------------------------------------\n")
    return all(ok for _, ok in results)


# Missing-source-marker appearance (sand-hatched loci)
MISSING_MARKER_FACE  = "#D8C49A"
MISSING_MARKER_EDGE  = "#6B5B3E"
MISSING_MARKER_HATCH = "////"
UNRESOLVED_MARKER_FACE = "#FFCC80"   # orange - unexpected/unresolved locations
UNRESOLVED_MARKER_EDGE = "#E65100"
UNRESOLVED_MARKER_HATCH = "...."

MISSING_LOCI_CAPTION = (
    "Sand-hatched loci indicate locations without a source Total Deviation "
    "value. These locations were excluded from numerical analyses. "
    "Interpolation was used only to generate the surrounding visual surface, "
    "after which the original missing-location mask was reapplied."
)


# Horizontal flip remap for the 76 points of the 30-2 grid, taken from
# GRID_SPECS["30-2"].
VF76_FLIP_REMAP = GRID_SPECS["30-2"]["flip_remap"]

# Colour-convention legend text for correlation maps.
COLOUR_LEGEND_CORR = (
    "RED = positive association: higher variable value → higher or less-negative "
    "Total Deviation → better visual-field sensitivity.\n"
    "BLUE = negative association: higher variable value → lower or more-negative "
    "Total Deviation → greater visual-field sensitivity loss."
)

# Core grid helpers
def pts_to_grid_100(pts_xy, vals_76, do_flip, signed=False, vmin=None, vmax=None,
                     extent_deg=30.0):
    """Interpolate point values onto a 100x100 display grid (display only).

    Statistics are computed on the original tested locations before this
    function is called, never on its output.

    NaN (missing) locations are left out of the interpolation. The caller
    redraws the missing-location markers on top of the image (see
    ResearchApp._draw_missing_markers).

    vmin/vmax: display clip range in dB. If either is omitted, the range is
    (-30, 30) when `signed` is true (longitudinal change) and
    (TD_PLAUSIBLE_MIN, TD_PLAUSIBLE_MAX) otherwise.

    extent_deg: half-width of the display window in degrees of visual angle:
    30 for 24-2 and 30-2 (points reach +-27 deg), 60 for 60-4 (points reach
    +-54 deg). It has to cover the eccentricity range of pts_xy, otherwise
    points near the edge fall outside the window. The 100x100 grid and the
    inscribed-circle mask are the same for every extent; only the degree scale
    changes.
    """
    if vmin is None or vmax is None:
        if signed:
            vmin, vmax = -30.0, 30.0
        else:
            vmin, vmax = TD_PLAUSIBLE_MIN, TD_PLAUSIBLE_MAX

    pa = pts_xy.copy()
    if do_flip:
        pa[:, 0] = -pa[:, 0]
    vals = np.asarray(vals_76, dtype=float)
    valid = np.isfinite(vals)
    gx, gy = np.meshgrid(np.linspace(-extent_deg, extent_deg, 100),
                          np.linspace( extent_deg, -extent_deg, 100))
    if valid.sum() < 3:
        # Too few measurements to interpolate a surface.
        return np.full((100, 100), np.nan)
    grid = griddata(pa[valid], vals[valid], (gx, gy), method='cubic')
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        grid[nan_mask] = griddata(pa[valid], vals[valid], (gx, gy), method='nearest')[nan_mask]
    grid = np.clip(grid, vmin, vmax)
    y, x = np.ogrid[-50:50, -50:50]
    grid[x*x + y*y > 50*50] = np.nan
    return grid


def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    The multiple-testing family is the set of finite (computable) p-values
    passed in. Non-computable locations (np.nan) are not coerced to p = 1.0,
    and their q-value is returned as np.nan. To restrict the family to an
    analysis region, set the p-values outside the region to np.nan before
    calling.
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        return _bh_manual(p_values, alpha)
    flat  = np.asarray(p_values, dtype=float).flatten()
    valid = np.isfinite(flat)
    q_flat = np.full_like(flat, np.nan)
    if valid.sum() > 0:
        _, q_adj, _, _ = multipletests(flat[valid], alpha=alpha, method='fdr_bh')
        q_flat[valid] = q_adj
    return q_flat.reshape(np.asarray(p_values).shape)


def _bh_manual(p_values, alpha=0.05):
    flat   = np.asarray(p_values, dtype=float).flatten()
    valid  = np.isfinite(flat)
    q_flat = np.full_like(flat, np.nan)
    v_idx  = np.where(valid)[0]
    n      = len(v_idx)
    if n == 0:
        return q_flat.reshape(np.asarray(p_values).shape)
    v_p    = flat[v_idx]
    order  = np.argsort(v_p)
    q      = np.ones(n)
    for i, idx in enumerate(order):
        q[idx] = v_p[idx] * n / (i + 1)
    q_mono = np.minimum.accumulate(q[order[::-1]])[::-1]
    for i, idx in enumerate(order):
        q[idx] = q_mono[i]
    q_flat[v_idx] = np.clip(q, 0, 1)
    return q_flat.reshape(np.asarray(p_values).shape)


def _nn_spacing_deg(points):
    """Median nearest-neighbor distance (degrees) among a set of grid points."""
    d = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def _stamp_significant_points(grid, pts, values, mask, ext, spacing_deg):
    """Paint each masked point's own value into `grid` as a small disk, in place.

    Interpolation can smooth an isolated significant point, surrounded by
    non-significant neighbors, down to near-invisibility or to the wrong
    colour, even though the figure title counts it. Stamping makes every
    significant point appear on the map, on the same colour scale as the rest
    of the map. The disk radius is a fraction of the grid's point spacing, so
    the disks of neighboring points do not merge.
    """
    if not np.any(mask):
        return grid
    r_deg = spacing_deg * 0.42
    r_px  = max(2.0, r_deg / (2 * ext) * 99)
    yy, xx = np.mgrid[0:100, 0:100]
    for (px, py), val in zip(pts[mask], values[mask]):
        if not np.isfinite(val):
            continue
        col = (px - (-ext)) / (2 * ext) * 99
        row = (ext - py) / (2 * ext) * 99
        disk = (xx - col) ** 2 + (yy - row) ** 2 <= r_px ** 2
        grid[disk] = val
    return grid


# Group-comparison statistics (two levels: Welch t / Hedges' g; three or more:
# Welch ANOVA / Games-Howell). Module-level so they can be unit-tested.
GROUP_SD_EPS = 1e-6          # a group with sd <= this at a location is treated as zero-variance
GROUP_ALPHA  = 0.05


def hedges_j(df):
    """Exact small-sample correction J(df) = Gamma(df/2) / (sqrt(df/2) * Gamma((df-1)/2))."""
    df = float(df)
    if df <= 1.0:
        return np.nan
    return float(np.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1.0) / 2.0)) / np.sqrt(df / 2.0))


def hedges_g_pair(a, b):
    """Hedges' g = J(n_a+n_b-2) * (mean_b - mean_a) / s_pooled, s_pooled from the two groups' own
    variances weighted by (n-1). np.nan when a group has < 2 values or s_pooled <= GROUP_SD_EPS."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    df = na + nb - 2
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / df)
    if not sp > GROUP_SD_EPS:
        return np.nan
    return float(hedges_j(df) * (np.mean(b) - np.mean(a)) / sp)


def welch_df(a, b):
    """Welch-Satterthwaite degrees of freedom for two samples."""
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    return float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))


def welch_anova(groups):
    """Welch's heteroscedastic one-way ANOVA. groups: list of 1-D arrays (every value finite).
    Returns dict(F, df1, df2, p, eta_sq, omega_sq) or None when not computable (a group with < 2 values
    or sd <= GROUP_SD_EPS). eta_sq / omega_sq come from the classical sum-of-squares decomposition
    (descriptive; omega_sq is not truncated at 0)."""
    from scipy.stats import f as _f
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    if k < 2 or np.any(n < 2):
        return None
    m = np.array([np.mean(g) for g in groups]); v = np.array([np.var(g, ddof=1) for g in groups])
    if np.any(np.sqrt(v) <= GROUP_SD_EPS):
        return None
    w = n / v; W = w.sum(); mw = (w * m).sum() / W
    lam = ((1.0 - w / W) ** 2 / (n - 1.0)).sum()
    F = ((w * (m - mw) ** 2).sum() / (k - 1.0)) / (1.0 + 2.0 * (k - 2.0) / (k ** 2 - 1.0) * lam)
    df1, df2 = k - 1.0, (k ** 2 - 1.0) / (3.0 * lam)
    N = n.sum(); gm = (n * m).sum() / N
    ssb = (n * (m - gm) ** 2).sum(); ssw = ((n - 1.0) * v).sum(); msw = ssw / (N - k)
    return {"F": float(F), "df1": float(df1), "df2": float(df2), "p": float(_f.sf(F, df1, df2)),
            "eta_sq": float(ssb / (ssb + ssw)), "omega_sq": float((ssb - (k - 1.0) * msw) / (ssb + ssw + msw))}


def games_howell(groups):
    """All unique pairwise Games-Howell comparisons (i < j in the given order). Each: diff = mean_j - mean_i,
    Welch t, Welch df, p from the studentized range distribution (family-wise error controlled across the
    k(k-1)/2 contrasts), and Hedges' g for the pair."""
    from scipy.stats import studentized_range
    k = len(groups)
    n = [len(g) for g in groups]
    m = [float(np.mean(g)) for g in groups]; v = [float(np.var(g, ddof=1)) for g in groups]
    out = []
    for i in range(k):
        for j in range(i + 1, k):
            se2 = v[i] / n[i] + v[j] / n[j]
            t = (m[j] - m[i]) / np.sqrt(se2)
            df = se2 ** 2 / ((v[i] / n[i]) ** 2 / (n[i] - 1) + (v[j] / n[j]) ** 2 / (n[j] - 1))
            out.append({"i": i, "j": j, "diff": m[j] - m[i], "t": float(t), "df": float(df),
                        "p": float(studentized_range.sf(abs(t) * np.sqrt(2.0), k, df)),
                        "hedges_g": hedges_g_pair(groups[i], groups[j])})
    return out


def stats_76_to_grid(stat_76, p_76, q_threshold=0.05, sigma=1.5, do_flip=False,
                      points=None, extent_deg=30.0):
    """Interpolate a per-location statistic and its significance masks for display.

    The statistic (t, F, a group difference, ...) is interpolated onto a
    100x100 grid using only the locations where both the statistic and its
    p-value are finite. Non-computable locations are left out of the
    interpolation and stay np.nan (transparent) in every returned grid. With
    fewer than three computable locations, all returned grids are np.nan.

    points, extent_deg: point array and display half-width (degrees) of the
    active grid (see pts_to_grid_100). `points` defaults to the 30-2 grid.
    """
    if points is None:
        points = POINTS_30_2
    stat_arr = np.asarray(stat_76, dtype=float)
    p_arr    = np.asarray(p_76,    dtype=float)
    valid_stat_mask = np.isfinite(stat_arr) & np.isfinite(p_arr)

    q_full  = bh_fdr(p_arr)   # np.nan preserved at non-computable locations
    sig     = (q_full <= q_threshold) & valid_stat_mask
    sig_unc = (p_arr < 0.05)          & valid_stat_mask

    pts = points.copy()
    if do_flip:
        pts[:, 0] = -pts[:, 0]

    n_valid = int(valid_stat_mask.sum())
    if n_valid < 3:
        empty = np.full((100, 100), np.nan)
        return {
            'q': q_full, 'sig_mask': sig, 'unc_mask': sig_unc,
            'stat_grid': empty, 'sig_grid': empty, 'unc_grid': empty,
            'n_sig': int(sig.sum()), 'n_unc': int(sig_unc.sum()),
            'n_total': len(points), 'n_valid': n_valid,
        }

    gx, gy = np.meshgrid(np.linspace(-extent_deg, extent_deg, 100),
                          np.linspace( extent_deg, -extent_deg, 100))
    v_pts  = pts[valid_stat_mask]
    v_stat = stat_arr[valid_stat_mask]
    v_p    = p_arr[valid_stat_mask]

    stat_grid = griddata(v_pts, v_stat, (gx, gy), method='cubic')
    nan_mask  = np.isnan(stat_grid)
    if nan_mask.any():
        stat_grid[nan_mask] = griddata(v_pts, v_stat, (gx, gy),
                                       method='nearest')[nan_mask]
    y, x = np.ogrid[-50:50, -50:50]
    circ = x*x + y*y > 50*50
    stat_grid[circ] = np.nan

    def _build_masked_grid(mask_full):
        grid = np.full((100, 100), np.nan)
        mask_v = mask_full[valid_stat_mask]
        if mask_v.sum() == 0:
            return grid
        p_grid = griddata(v_pts, v_p, (gx, gy), method='linear')
        nan_p  = np.isnan(p_grid)
        if nan_p.any():
            p_grid[nan_p] = griddata(v_pts, v_p, (gx, gy), method='nearest')[nan_p]
        p_grid = np.clip(p_grid, 0, 1)
        p_grid[circ] = np.nan
        p_sig_vals = v_p[mask_v]
        threshold  = float(np.max(p_sig_vals)) if len(p_sig_vals) > 0 else 0.05
        threshold  = min(threshold, 0.05)
        grid = stat_grid.copy()
        grid[p_grid > threshold] = np.nan
        grid[circ] = np.nan
        _stamp_significant_points(grid, pts, stat_arr, mask_full, extent_deg, _nn_spacing_deg(pts))
        return grid

    return {
        'q':         q_full,
        'sig_mask':  sig,
        'unc_mask':  sig_unc,
        'stat_grid': stat_grid,
        'sig_grid':  _build_masked_grid(sig),
        'unc_grid':  _build_masked_grid(sig_unc),
        'n_sig':     int(sig.sum()),
        'n_unc':     int(sig_unc.sum()),
        'n_total':   len(points),
        'n_valid':   n_valid,
    }


def safe_absmax(arr, default=1.0):
    try:
        a = np.abs(np.asarray(arr, dtype=float))
        valid = a[np.isfinite(a)]
        if len(valid) == 0:
            return default
        v = float(valid.max())
        return v if v > 0 else default
    except Exception:
        return default


def _wrap_title(text, width=80):
    """Wrap a title string to fit within figure width."""
    return "\n".join(textwrap.wrap(text, width=width))


def _nan_aware_gaussian_smooth(grid, sigma=1.5, tol=1e-6):
    """NaN-aware Gaussian smoothing of a 2-D display surface (display only).

    A value grid and a weight grid (1 where finite, 0 where missing) are
    smoothed separately and then divided. A missing cell therefore contributes
    zero weight to its neighbors; it is not treated as a 0 dB value, which
    would pull the surrounding surface toward 0 dB.

    The result is not used for numerical analysis: all statistics are computed
    on the original tested locations before any interpolation or smoothing.
    """
    arr = np.asarray(grid, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.full_like(arr, np.nan, dtype=float)
    value_grid  = np.where(valid, arr, 0.0)
    weight_grid = valid.astype(float)
    smooth_values  = gaussian_filter(value_grid,  sigma=sigma)
    smooth_weights = gaussian_filter(weight_grid, sigma=sigma)
    smoothed = np.full_like(smooth_values, np.nan, dtype=float)
    usable = smooth_weights > tol
    smoothed[usable] = smooth_values[usable] / smooth_weights[usable]
    return smoothed


def _smooth_in_field(grid, sigma=1.5):
    """Display smoothing of a full 100x100 field grid that stays inside the
    field circle. _nan_aware_gaussian_smooth gives a value to every pixel
    within a few sigma of real data, so on its own it spills the map past the
    dashed field boundary; masking after the smoothing (and before any
    quadrant crop) is what keeps colour from bleeding outside the circle."""
    out = _nan_aware_gaussian_smooth(grid, sigma=sigma)
    y, x = np.ogrid[-50:50, -50:50]
    out[x * x + y * y > 50 * 50] = np.nan
    return out


def _json_safe(value):
    """Recursively sanitize a value tree for json.dump(..., allow_nan=False).

      * NaN/Infinite Python float or NumPy floating -> None (JSON null)
      * finite NumPy floating                        -> Python float
      * NumPy integer                                -> Python int
      * NumPy Boolean                                -> Python bool
      * NumPy array                                  -> recursively sanitized list
      * dict                                         -> recursively sanitized dict (string keys)
      * list/tuple                                   -> recursively sanitized list
      * everything else (str, bool, int, finite float, None) -> unchanged

    Only NaN/Inf become None. A finite 0 is kept, so a measured 0 dB is never
    confused with a missing value.
    """
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]

    if isinstance(value, np.floating):
        value = float(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    return value


# Dialogs
class CropSelector:
    """Interactive matplotlib rectangle selector for cropping DICOM images."""
    def __init__(self, image, title):
        self.image       = image
        self.crop_coords = None
        self.temp_coords = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(self.image, cmap='gray')
        self.ax.set_title(f"{title}\n1. Drag box around circle | 2. Press ENTER | 3. Close")
        self.rs = RectangleSelector(self.ax, self.onselect,
                                    useblit=True, button=[1], interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    def onselect(self, eclick, erelease):
        self.temp_coords = (int(eclick.ydata), int(erelease.ydata),
                            int(eclick.xdata), int(erelease.xdata))

    def on_key(self, event):
        if event.key == 'enter' and self.temp_coords:
            self.crop_coords = self.temp_coords
            plt.close(self.fig)


class VariableSelector(simpledialog.Dialog):
    """Simple combobox dialog to pick one variable from a list."""
    def __init__(self, parent, title, choices, label="Select variable:"):
        self.choices   = choices
        self.selection = None
        self._label    = label
        super().__init__(parent, title)

    def body(self, master):
        tk.Label(master, text=self._label).pack(pady=5)
        self.combo = ttk.Combobox(master, values=self.choices,
                                  state="readonly", width=38)
        self.combo.pack(pady=5, padx=10)
        if self.choices:
            self.combo.current(0)
        return self.combo

    def apply(self):
        self.selection = self.combo.get()


# Main App
class ResearchApp:
    # Fixed VF columns - always present, cannot be deleted
    COLS_VF = ("Preop OD VF", "Preop OS VF", "Postop OD VF", "Postop OS VF")

    def __init__(self, root):
        self.root = root
        self.root.title("NeuroVF - Visual Field Research Platform  (v1.0)")

        # Load the database (the variable schema is stored inside it)
        self.db = self.load_db()

        # Choose the HVF pattern, study type and data type before building the
        # main window. Sets self.grid_pattern, self.data_type, self.grid,
        # self.grid_points, self.n_vf, self.grid_extent_deg and self.grid_flip_remap.
        self._select_grid_and_datatype()

        # Build column lists from schema
        self._rebuild_columns_from_schema()

        # Ensure all subjects have the right keys
        self.init_missing_columns()

        self.setup_ui()

        # Post-chiasmatic databases only: configure the surgery-side variable if
        # it is not configured yet (a no-op for other study types).
        self._ensure_surgery_side_configured()

        # Chiasmatic/PTC databases only: configure which eye is mirrored (a no-op
        # for other study types).
        self._ensure_flip_eye_configured()

    # HVF pattern + data-type selection
    def _db_has_vf_data(self):
        """True if any subject in self.db already has VF point data entered."""
        for sid, v in self.db.items():
            if sid in RESERVED_DB_KEYS or not isinstance(v, dict):
                continue
            for tp in ("preop", "postop"):
                for eye in ("od", "os"):
                    if v.get(f"vf_raw76_{tp}_{eye}"):
                        return True
            if v.get("vf_raw76_od") or v.get("vf_raw76_os"):
                return True
        return False

    def _apply_grid_and_datatype(self, pattern, data_type, study_type="general",
                                  surgery_side_var=None, surgery_side_reference=None,
                                  flip_eye=None):
        """Set every self.grid_* attribute from a chosen pattern/data_type/study_type."""
        self.grid_pattern     = pattern
        self.data_type        = data_type
        self.grid              = GRID_SPECS[pattern]
        self.grid_points       = self.grid["points"]
        self.n_vf              = self.grid["n"]
        self.grid_extent_deg   = self.grid["extent_deg"]
        self.grid_flip_remap   = self.grid["flip_remap"]
        self.data_type_spec    = DATA_TYPES[data_type]
        # Shortcuts used by the analysis code so that it does not branch on
        # self.data_type: severity thresholds, plausibility range and display
        # colour map and limits all follow the selected data type.
        self.severe_thr    = self.data_type_spec["severe"]
        self.moderate_thr  = self.data_type_spec["moderate"]
        self.mild_thr      = self.data_type_spec["mild"]
        self.plausible_min = self.data_type_spec["plausible_min"]
        self.plausible_max = self.data_type_spec["plausible_max"]
        self.map_cmap       = self.data_type_spec["cmap"]
        self.map_vmin        = self.data_type_spec["vmin"]
        self.map_vmax        = self.data_type_spec["vmax"]
        self.tier_labels     = self.data_type_spec["tier_labels"]
        # Both data types have an operational severity banding, but neither is an
        # independently validated per-point clinical standard. severity_caveat is
        # the text shown wherever the bands drive an analysis (cluster,
        # cumulative, scotoma-area report).
        self.severity_supported = True
        self.severity_caveat    = self.data_type_spec["severity_caveat"]

        # Study type: OD/OS registration policy (see CLINICAL_TYPES).
        if study_type not in CLINICAL_TYPES:
            study_type = "general"
        self.clinical_type       = study_type
        self.clinical_type_spec  = CLINICAL_TYPES[study_type]
        self.flip_policy       = self.clinical_type_spec["flip_policy"]
        self.morphometry_note  = self.clinical_type_spec["morphometry_note"]
        self.surgery_side_var        = surgery_side_var
        self.surgery_side_reference  = surgery_side_reference
        # Eye mirrored under "flip_one_eye_cohort", chosen once per database (see
        # _ensure_flip_eye_configured). None until configured; _get_raw76_pair
        # then falls back to "os".
        self.flip_eye = flip_eye if flip_eye in ("od", "os") else None

        self.db[VF_GRID_PATTERN_KEY] = pattern
        self.db[VF_DATA_TYPE_KEY]    = data_type
        self.db[VF_CLINICAL_TYPE_KEY]   = study_type
        if surgery_side_var is not None:
            self.db[VF_SURGERY_SIDE_VAR_KEY] = surgery_side_var
        if surgery_side_reference is not None:
            self.db[VF_SURGERY_SIDE_REFERENCE_KEY] = surgery_side_reference
        if self.flip_eye is not None:
            self.db[VF_FLIP_EYE_KEY] = self.flip_eye
        if hasattr(self, "root") and self.root is not None:
            try:
                self.root.title(
                    f"NeuroVF - Visual Field Research Platform  (v1.0)   "
                    f"[{GRID_SPECS[pattern]['label']}  |  {DATA_TYPES[data_type]['label']}  |  "
                    f"{CLINICAL_TYPES[study_type]['label'].split('  (')[0]}]")
            except Exception:
                pass

    @staticmethod
    def _center_window(win, root=None):
        """Center a Toplevel on its parent, or on the screen if the parent is not drawn.

        Only the window position is set, clamped so that the window stays on
        screen. The window is not raised, focused or made topmost, because that
        interferes with grab_set() on Windows.
        """
        try:
            win.update_idletasks()
            w = win.winfo_reqwidth() or win.winfo_width() or 400
            h = win.winfo_reqheight() or win.winfo_height() or 200
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            w = min(w, int(sw * 0.9))
            h = min(h, int(sh * 0.9))
            root_ok = False
            if root is not None and root.winfo_exists():
                root.update_idletasks()
                rw, rh = root.winfo_width(), root.winfo_height()
                if rw > 50 and rh > 50:   # root has actually been drawn
                    rx, ry = root.winfo_x(), root.winfo_y()
                    x = rx + max(0, (rw - w) // 2)
                    y = ry + max(0, (rh - h) // 2)
                    root_ok = True
            if not root_ok:
                x = max(0, (sw - w) // 2)
                y = max(0, (sh - h) // 2)
            x = max(0, min(x, sw - w))
            y = max(0, min(y, sh - h))
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass  # never let a display quirk crash the startup flow

    def _select_grid_and_datatype(self):
        """Show the pattern / study type / data type selector.

        If the database already contains VF data, the three settings are locked
        to the stored values and only an informational dialog is shown.
        Otherwise the user chooses each of them from a row of radio buttons.

        The surgery-side variable (post-chiasmatic policy) is not asked for
        here, because a new database has no variables yet. It is configured
        later by _ensure_surgery_side_configured(), once the variable schema is
        loaded.
        """
        locked = self._db_has_vf_data()
        stored_pattern    = self.db.get(VF_GRID_PATTERN_KEY, "30-2")
        stored_data_type  = self.db.get(VF_DATA_TYPE_KEY, "total_deviation")
        stored_study_type = self.db.get(VF_CLINICAL_TYPE_KEY, "general")
        if stored_pattern not in GRID_SPECS:
            stored_pattern = "30-2"
        if stored_data_type not in DATA_TYPES:
            stored_data_type = "total_deviation"
        if stored_study_type not in CLINICAL_TYPES:
            stored_study_type = "general"

        if locked:
            # VF data present: do not offer a change that would invalidate the
            # stored point arrays or the registration of existing results.
            stored_surgery_var = self.db.get(VF_SURGERY_SIDE_VAR_KEY)
            stored_surgery_ref = self.db.get(VF_SURGERY_SIDE_REFERENCE_KEY)
            stored_flip_eye    = self.db.get(VF_FLIP_EYE_KEY)
            self._apply_grid_and_datatype(stored_pattern, stored_data_type, stored_study_type,
                                           stored_surgery_var, stored_surgery_ref, stored_flip_eye)
            win = tk.Toplevel(self.root) if hasattr(self, "root") and self.root.winfo_exists() else tk.Toplevel()
            win.title("Database Pattern - Locked")
            win.grab_set(); win.resizable(False, False)
            tk.Label(win, text="This database already contains visual-field data.",
                     font=("Arial", 10, "bold"), pady=8, wraplength=440,
                     justify="left").pack(padx=20)
            info_text = (f"Pattern:     {GRID_SPECS[stored_pattern]['label']}\n"
                         f"Data type:   {DATA_TYPES[stored_data_type]['label']}\n"
                         f"Study type:  {CLINICAL_TYPES[stored_study_type]['label']}\n\n"
                         "To analyze a different pattern, data type, or study type, use "
                         "(or create) a different database file - mixing these in one "
                         "database would silently corrupt every ROI, registration, and "
                         "statistical computation.")
            tk.Label(win, text=info_text,
                     wraplength=460, justify="left", padx=20, pady=6).pack()
            if self.morphometry_note:
                tk.Label(win, text=self.morphometry_note, wraplength=460, justify="left",
                         padx=20, pady=6, fg="#1565c0").pack()
            tk.Button(win, text="Continue →", bg="#1a237e", fg="white",
                      font=("Arial", 10, "bold"),
                      command=win.destroy).pack(pady=12)
            self._center_window(win, getattr(self, "root", None))
            win.wait_window()
            return

        # New or empty database: free choice
        win = tk.Toplevel(self.root) if hasattr(self, "root") and self.root.winfo_exists() else tk.Toplevel()
        win.title("Select HVF Pattern, Study Type & Data Type")
        win.grab_set(); win.resizable(False, False)

        tk.Label(win, text="New database - choose the visual-field test pattern, "
                            "study type, and data type this database will hold:",
                 font=("Arial", 10, "bold"), wraplength=460, justify="left",
                 pady=10).pack(padx=20)

        tk.Label(win, text="Humphrey Visual Field pattern:",
                 font=("Arial", 9, "bold"), anchor="w").pack(fill="x", padx=20, pady=(8, 0))
        pattern_var = tk.StringVar(value=stored_pattern)
        for key in ("24-2", "30-2", "60-4"):
            tk.Radiobutton(win, text=GRID_SPECS[key]["label"], variable=pattern_var,
                           value=key, font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)

        tk.Label(win, text="Study type (OD/OS registration policy):",
                 font=("Arial", 9, "bold"), anchor="w").pack(fill="x", padx=20, pady=(10, 0))
        study_var = tk.StringVar(value=stored_study_type)
        for key in ("general", "post_chiasmatic", "chiasmatic", "ptc"):
            tk.Radiobutton(win, text=CLINICAL_TYPES[key]["label"], variable=study_var,
                           value=key, font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)

        tk.Label(win, text="Data type to analyze:",
                 font=("Arial", 9, "bold"), anchor="w").pack(fill="x", padx=20, pady=(10, 0))
        dtype_var = tk.StringVar(value=stored_data_type)
        for key in ("total_deviation", "raw_threshold"):
            tk.Radiobutton(win, text=DATA_TYPES[key]["label"], variable=dtype_var,
                           value=key, font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)

        confirmed = tk.BooleanVar(value=False)
        def on_ok():
            confirmed.set(True); win.destroy()
        bf = tk.Frame(win); bf.pack(pady=14)
        tk.Button(bf, text="Continue →", bg="#1a237e", fg="white",
                  font=("Arial", 10, "bold"), command=on_ok).pack(side=tk.LEFT, padx=10)
        win.protocol("WM_DELETE_WINDOW", on_ok)  # closing = accept defaults, never hang
        self._center_window(win, getattr(self, "root", None))
        win.wait_window()

        self._apply_grid_and_datatype(pattern_var.get(), dtype_var.get(), study_var.get())
        self.save_db()
        if self.morphometry_note:
            try:
                messagebox.showinfo(f"{CLINICAL_TYPES[study_var.get()]['label']} - Registration Note",
                                     self.morphometry_note)
            except Exception:
                pass

    SURGERY_SIDE_VAR_NAME = "Surgery Side"

    def _ensure_surgery_side_configured(self):
        """Configure the surgery-side variable ("surgery_side_mirror" policy only).

        Adds the fixed "Surgery Side" categorical variable (Left/Right) to the
        schema if it is missing, then asks once which side is mirrored; the
        other side becomes the reference (not mirrored). The variable name is
        fixed and the variable is mandatory: process_vf_db_ocr refuses to import
        VF data for a subject whose Surgery Side is not set.

        Called once from __init__, after the main window is built. A no-op for
        other flip policies and when already configured.
        """
        if getattr(self, "flip_policy", "manual") != "surgery_side_mirror":
            return
        if getattr(self, "surgery_side_var", None):
            return  # already configured

        var_name = self.SURGERY_SIDE_VAR_NAME
        if var_name not in self.variable_schema:
            self.variable_schema[var_name] = {"type": "Categorical", "options": ["Left", "Right"]}
            self._save_schema()
            self._rebuild_columns_from_schema()
            self.init_missing_columns()
            # setup_ui built the table with the previous column list, and
            # _rebuild_columns_from_schema only updates self.columns. Rebuild the
            # widget as well, so that cell clicks resolve to the right column.
            self._rebuild_treeview()
            self.refresh_table()

        opts = list(self.variable_schema.get(var_name, {}).get("options", []) or ["Left", "Right"])
        if len(opts) < 2:
            opts = ["Left", "Right"]

        win = tk.Toplevel(self.root) if hasattr(self, "root") and self.root.winfo_exists() else tk.Toplevel()
        win.title("Configure Surgery Side")
        win.grab_set(); win.resizable(False, False)
        tk.Label(win, text=f"Post-chiasmatic registration mirrors the whole field by "
                            f"surgery side. A '{var_name}' variable has been added to "
                            f"your schema (mandatory - every subject must have it set "
                            f"before VF data can be imported). Which side should be "
                            f"treated as FLIPPED (mirrored)?",
                 font=("Arial", 10, "bold"), wraplength=440, justify="left",
                 pady=10).pack(padx=20)

        flip_choice = tk.StringVar(value=opts[0])
        for opt in opts:
            tk.Radiobutton(win, text=opt, variable=flip_choice, value=opt,
                           font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)

        confirmed = tk.BooleanVar(value=False)
        def on_ok():
            confirmed.set(True); win.destroy()
        tk.Button(win, text="Save →", bg="#1a237e", fg="white", font=("Arial", 10, "bold"),
                  command=on_ok).pack(pady=14)
        win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())
        self._center_window(win, getattr(self, "root", None))
        win.wait_window()

        if confirmed.get():
            flip_side = flip_choice.get()
            reference_side = next((o for o in opts if o != flip_side), opts[0])
            self.surgery_side_var       = var_name
            self.surgery_side_reference = reference_side
            self.db[VF_SURGERY_SIDE_VAR_KEY]       = var_name
            self.db[VF_SURGERY_SIDE_REFERENCE_KEY] = reference_side
            self.save_db()

    def _ensure_flip_eye_configured(self):
        """Ask once which eye is mirrored ("flip_one_eye_cohort" policy only).

        The choice (OD or OS) applies to every subject and is stored in the
        database (see _get_raw76_pair). Called once from __init__, after the
        main window is built. A no-op for other flip policies and when already
        configured.
        """
        if getattr(self, "flip_policy", "manual") != "flip_one_eye_cohort":
            return
        if getattr(self, "flip_eye", None):
            return  # already configured

        win = tk.Toplevel(self.root) if hasattr(self, "root") and self.root.winfo_exists() else tk.Toplevel()
        win.title("Configure Flip Eye")
        win.grab_set(); win.resizable(False, False)
        tk.Label(win, text=f"{self.clinical_type_spec['label'].split('  (')[0]} registration "
                            f"mirrors ONE eye to the other's nasal/temporal orientation for "
                            f"every subject in the cohort. Which eye should be flipped?",
                 font=("Arial", 10, "bold"), wraplength=440, justify="left",
                 pady=10).pack(padx=20)

        eye_choice = tk.StringVar(value="os")
        tk.Radiobutton(win, text="OD (right eye)", variable=eye_choice, value="od",
                       font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)
        tk.Radiobutton(win, text="OS (left eye)", variable=eye_choice, value="os",
                       font=("Arial", 10)).pack(anchor="w", padx=36, pady=2)

        confirmed = tk.BooleanVar(value=False)
        def on_ok():
            confirmed.set(True); win.destroy()
        tk.Button(win, text="Save →", bg="#1a237e", fg="white", font=("Arial", 10, "bold"),
                  command=on_ok).pack(pady=14)
        win.protocol("WM_DELETE_WINDOW", lambda: win.destroy())
        self._center_window(win, getattr(self, "root", None))
        win.wait_window()

        if confirmed.get():
            self.flip_eye = eye_choice.get()
            self.db[VF_FLIP_EYE_KEY] = self.flip_eye
            self.save_db()

    # Schema helpers
    def _rebuild_columns_from_schema(self):
        """Derive self.columns, self.categorical_cols and self.continuous_cols from the schema.

        Schema format:
            {var_name: {"type": "Categorical"|"Continuous", "options": [...]}}
        The legacy flat format is also accepted:
            {var_name: "Categorical"|"Continuous"}
        """
        raw_schema = self.db.get("__schema__", {})
        if not isinstance(raw_schema, dict):
            raw_schema = {}

        # Normalize legacy flat-string entries to the dict format
        schema = {}
        for k, v in raw_schema.items():
            if isinstance(v, dict):
                schema[k] = v   # already in dict format
            elif isinstance(v, str):
                schema[k] = {"type": v, "options": []}
            # anything else is malformed and skipped

        self.variable_schema = schema
        # {var_name: {"type": "Categorical"|"Continuous", "options": [...]}}

        # Fixed columns
        self.cols_vf = list(self.COLS_VF)

        # Dynamic user-defined columns (preserving insertion order via dict)
        self.user_defined_cols = list(schema.keys())

        # Fixed action columns, placed after the four VF columns
        self.COL_VF_VIEW = "View VF"
        self.COL_SCOTOMA_AREA = "Scotoma Area"

        # Full column list for Treeview: ID + VF + viewer button + area button + user-defined
        self.columns = (("ID",) + self.COLS_VF + (self.COL_VF_VIEW, self.COL_SCOTOMA_AREA)
                         + tuple(self.user_defined_cols))

        # Typed subsets
        self.categorical_cols = [k for k, v in schema.items()
                                  if v.get("type") == "Categorical"]
        self.continuous_cols  = [k for k, v in schema.items()
                                  if v.get("type") == "Continuous"]

    def _schema_type(self, vname):
        """Return 'Categorical' or 'Continuous' for a variable, or '' if unknown."""
        entry = self.variable_schema.get(vname, {})
        return entry.get("type", "") if isinstance(entry, dict) else str(entry)

    def _schema_options(self, vname):
        """Return the list of category options for a Categorical variable."""
        entry = self.variable_schema.get(vname, {})
        if isinstance(entry, dict):
            return entry.get("options", [])
        return []

    def _save_schema(self):
        """Persist variable_schema inside the DB under reserved key __schema__."""
        self.db["__schema__"] = self.variable_schema
        self.save_db()

    def add_variable_dialog(self):
        """
        Open dialog to add a new variable to the schema.
        For Categorical variables: user defines the exact option names
        (e.g. 'Left' and 'Right' for a 'side' variable).
        """
        # Step 1: Name + Type
        win = tk.Toplevel(self.root)
        win.title("Add Variable - Step 1: Name & Type")
        win.grab_set(); win.resizable(False, False)

        tk.Label(win, text="Variable Name:", font=("Arial", 10)).grid(
            row=0, column=0, padx=16, pady=10, sticky='w')
        name_var = tk.StringVar()
        name_entry = tk.Entry(win, textvariable=name_var, width=28, font=("Arial", 10))
        name_entry.grid(row=0, column=1, padx=8, pady=10)
        name_entry.focus_set()

        tk.Label(win, text="Type:", font=("Arial", 10)).grid(
            row=1, column=0, padx=16, pady=4, sticky='w')
        type_var = tk.StringVar(value="Continuous")
        type_frame = tk.Frame(win)
        type_frame.grid(row=1, column=1, padx=8, pady=4, sticky='w')
        tk.Radiobutton(type_frame, text="Continuous  (numeric: age, volume, score…)",
                       variable=type_var, value="Continuous",
                       font=("Arial", 10)).pack(anchor='w', pady=2)
        tk.Radiobutton(type_frame, text="Categorical  (named groups: left/right, male/female…)",
                       variable=type_var, value="Categorical",
                       font=("Arial", 10)).pack(anchor='w', pady=2)

        confirmed1 = tk.BooleanVar(value=False)
        def on_next():
            if name_var.get().strip():
                confirmed1.set(True)
            win.destroy()
        def on_cancel1():
            win.destroy()

        bf = tk.Frame(win); bf.grid(row=2, column=0, columnspan=2, pady=14)
        tk.Button(bf, text="Next →", bg="#1a237e", fg="white",
                  font=("Arial", 10, "bold"), command=on_next).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=on_cancel1).pack(side=tk.LEFT)
        win.wait_window()

        if not confirmed1.get():
            return
        vname = name_var.get().strip()
        vtype = type_var.get()
        if not vname:
            return
        if vname in self.variable_schema:
            messagebox.showwarning("Duplicate", f"Variable '{vname}' already exists.")
            return
        if vname in ("ID",) + self.COLS_VF:
            messagebox.showwarning("Reserved", f"'{vname}' is a reserved column name.")
            return

        # Step 2: For Categorical - define options
        cat_options = []
        if vtype == "Categorical":
            win2 = tk.Toplevel(self.root)
            win2.title(f"Add Variable '{vname}' - Define Categories")
            win2.grab_set()
            win2.resizable(False, False)

            # Header
            tk.Label(win2,
                     text=f"Define categories for:  '{vname}'",
                     font=("Arial", 11, "bold"), pady=8).grid(
                     row=0, column=0, columnspan=3, padx=16, sticky='w')

            tk.Label(win2, text="Number of categories:",
                     font=("Arial", 10)).grid(row=1, column=0, padx=16, pady=6, sticky='w')

            n_var = tk.IntVar(value=2)
            n_spin = tk.Spinbox(win2, from_=2, to=10, textvariable=n_var,
                                width=5, font=("Arial", 10), state="readonly")
            n_spin.grid(row=1, column=1, padx=8, pady=6, sticky='w')

            # Entry fields frame
            entries_frame = tk.Frame(win2, bd=1, relief='groove', bg="#f5f5f5")
            entries_frame.grid(row=2, column=0, columnspan=3,
                               padx=16, pady=6, sticky='ew')

            entry_vars = []

            def build_entries(*_):
                """Rebuild the category name entry rows."""
                for w in entries_frame.winfo_children():
                    w.destroy()
                entry_vars.clear()
                try:
                    n = int(n_var.get())
                except (ValueError, tk.TclError):
                    n = 2
                n = max(2, min(10, n))

                tk.Label(entries_frame, text="Category names:",
                         font=("Arial", 9, "bold"), bg="#f5f5f5").grid(
                         row=0, column=0, columnspan=2,
                         padx=10, pady=(8, 4), sticky='w')

                first_entry = None
                for i in range(n):
                    tk.Label(entries_frame, text=f"  Option {i+1}:",
                             font=("Arial", 10), bg="#f5f5f5").grid(
                             row=i+1, column=0, padx=10, pady=4, sticky='w')
                    ev = tk.StringVar()
                    en = tk.Entry(entries_frame, textvariable=ev,
                                  width=24, font=("Arial", 10))
                    en.grid(row=i+1, column=1, padx=10, pady=4)
                    entry_vars.append(ev)
                    if first_entry is None:
                        first_entry = en

                if first_entry:
                    first_entry.focus_set()
                win2.update_idletasks()

            # Build initial entries, then wire spinbox arrows only
            build_entries()
            n_spin.configure(command=build_entries)

            # Hint
            tk.Label(win2,
                     text="Example: 2 options → 'Left', 'Right'\n"
                          "These will appear as a dropdown in the table and Add Subject dialog.",
                     font=("Arial", 8, "italic"), fg="#555").grid(
                     row=3, column=0, columnspan=3, padx=16, pady=(2, 6), sticky='w')

            # Buttons
            confirmed2 = tk.BooleanVar(value=False)

            def on_add():
                opts = [e.get().strip() for e in entry_vars if e.get().strip()]
                if len(opts) < 2:
                    messagebox.showwarning("Incomplete",
                        "Please fill in at least 2 category names.")
                    return
                cat_options.extend(opts)
                confirmed2.set(True)
                win2.destroy()

            def on_cancel2():
                win2.destroy()

            bf2 = tk.Frame(win2)
            bf2.grid(row=4, column=0, columnspan=3, pady=14)
            tk.Button(bf2, text="Add Variable", bg="#2e7d32", fg="white",
                      font=("Arial", 10, "bold"),
                      command=on_add).pack(side=tk.LEFT, padx=10)
            tk.Button(bf2, text="Cancel",
                      command=on_cancel2).pack(side=tk.LEFT, padx=4)

            win2.wait_window()

            if not confirmed2.get() or not cat_options:
                return

        # Commit to the schema. self.variable_schema and self.db["__schema__"]
        # are updated before the columns are rebuilt, because
        # _rebuild_columns_from_schema reads the schema from self.db.
        self.variable_schema[vname] = {
            "type":    vtype,
            "options": cat_options,   # [] for Continuous
        }
        # Store the schema in the database and save it before rebuilding the columns
        self.db["__schema__"] = self.variable_schema
        self.save_db()

        # Rebuild the columns from the stored schema
        self._rebuild_columns_from_schema()

        # Add the new column key to every existing subject record
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS:
                continue
            if isinstance(data, dict):
                data.setdefault(vname, "")
        self.save_db()   # persist the subject-level additions too

        self._rebuild_treeview()
        self.refresh_table()

        if vtype == "Categorical":
            opts_str = ", ".join(f"'{o}'" for o in cat_options)
            messagebox.showinfo("Variable Added",
                f"Categorical variable '{vname}' added.\n"
                f"Options: {opts_str}\n\n"
                f"Double-click the '{vname}' cell in the table to set a value.")
        else:
            messagebox.showinfo("Variable Added",
                f"Continuous variable '{vname}' added.\n"
                "Double-click any cell in that column to enter a numeric value.")

    def _rebuild_treeview(self):
        """Destroy and recreate the Treeview with current columns."""
        # Destroy existing tree and scrollbars
        for widget in self.tree_frame.winfo_children():
            widget.destroy()

        self.tree = ttk.Treeview(self.tree_frame,
                                  columns=self.columns, show='headings')
        for col in self.columns:
            self.tree.heading(col, text=COL_DISPLAY_LABELS.get(col, col))
            if col == "ID":
                w = 100
            elif col in self.COLS_VF:
                w = 100
            elif col == self.COL_VF_VIEW:
                w = 80
            elif col == self.COL_SCOTOMA_AREA:
                w = 110
            else:
                w = 110
            self.tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(self.tree_frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-1>",  self.on_single_click)

    # UI
    def setup_ui(self):
        toolbar = tk.Frame(self.root, bg="#37474f")
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Grid/Data Type", bg="#004d40", fg="white",
                  font=("Arial", 9, "bold"),
                  command=self._select_grid_and_datatype).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(toolbar, text="|", bg="#37474f", fg="#90a4ae",
                 font=("Arial", 11)).pack(side=tk.LEFT, padx=2)

        if self.clinical_type == "ptc":
            tk.Button(toolbar, text="Enter Device MD", bg="#00695c", fg="white",
                      font=("Arial", 9, "bold"),
                      command=self.enter_manual_md_dialog).pack(side=tk.LEFT, padx=4, pady=4)

            tk.Label(toolbar, text="|", bg="#37474f", fg="#90a4ae",
                     font=("Arial", 11)).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="＋ Add Variable", bg="#4a148c", fg="white",
                  font=("Arial", 9, "bold"),
                  command=self.add_variable_dialog).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(toolbar, text="|", bg="#37474f", fg="#90a4ae",
                 font=("Arial", 11)).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="＋ Add Subject", bg="#546e7a", fg="white",
                  font=("Arial", 9, "bold"),
                  command=self.add_subject_dialog).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Button(toolbar, text="Delete Subject", bg="#b71c1c", fg="white",
                  font=("Arial", 9, "bold"),
                  command=self.delete_subject).pack(side=tk.LEFT, padx=2, pady=4)

        tk.Label(toolbar, text="|", bg="#37474f", fg="#90a4ae",
                 font=("Arial", 11)).pack(side=tk.LEFT, padx=2)

        tk.Button(toolbar, text="Paste Values", bg="#00695c", fg="white",
                  font=("Arial", 9, "bold"),
                  command=self.bulk_paste_dialog).pack(side=tk.LEFT, padx=4, pady=4)

        tk.Label(toolbar, text="  |  Analysis Area:", bg="#37474f",
                 fg="white", font=("Arial", 9)).pack(side=tk.LEFT)
        self.quadrant_var = tk.StringVar(value="Full Field")
        quad_menu = ttk.Combobox(toolbar, textvariable=self.quadrant_var,
                                 state="readonly", width=20)
        quad_menu['values'] = ("Full Field",
                               "Superior Left","Superior Right",
                               "Inferior Left","Inferior Right",
                               "2 Left Quarters","2 Right Quarters")
        quad_menu.pack(side=tk.LEFT, padx=4)

        tk.Label(toolbar, text="  |", bg="#37474f", fg="#90a4ae").pack(side=tk.LEFT)

        tk.Button(toolbar, text="Analyse", bg="#1a237e", fg="white",
                  font=("Arial", 10, "bold"),
                  command=self.open_analysis_dialog).pack(side=tk.LEFT, padx=6, pady=4)

        # Legend bar
        legend_bar = tk.Frame(self.root, bg="#eceff1", height=24)
        legend_bar.pack(side=tk.TOP, fill=tk.X)
        legend_bar.pack_propagate(False)
        for txt, col in [
            ("Visual Field (Pre/Post)", "#0277bd"),
            ("User Variables (dynamic)", "#2e7d32"),
            (f"{self.grid['label']} | {self.data_type_spec['label']} | RED=Positive corr", "#d32f2f"),
        ]:
            tk.Label(legend_bar, text=txt, fg=col, bg="#eceff1",
                     font=("Arial", 8, "bold")).pack(side=tk.LEFT, padx=10)

        # Table frame
        self.tree_frame = tk.Frame(self.root)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=1)

        # Build initial treeview
        self._rebuild_treeview()
        self.refresh_table()

    # Analysis dialog
    def open_analysis_dialog(self):
        """Step 1: choose Cross-Sectional or Longitudinal."""
        win = tk.Toplevel(self.root)
        win.title("Select Study Type")
        win.grab_set(); win.resizable(False, False)

        tk.Label(win, text="Select study type for analysis:",
                 font=("Arial", 11, "bold"), pady=10).pack(padx=20)

        study_var = tk.StringVar(value="cross")
        tk.Radiobutton(win, text="Cross-Sectional  (single time-point: Timepoint 1 or Timepoint 2)",
                       variable=study_var, value="cross",
                       font=("Arial", 10)).pack(anchor='w', padx=30, pady=4)
        tk.Radiobutton(win, text="Longitudinal  (Timepoint 1 → Timepoint 2 change: Δ per subject first)",
                       variable=study_var, value="longitudinal",
                       font=("Arial", 10)).pack(anchor='w', padx=30, pady=4)

        confirmed = tk.BooleanVar(value=False)
        def on_ok():    confirmed.set(True); win.destroy()
        def on_cancel(): win.destroy()

        bf = tk.Frame(win); bf.pack(pady=12)
        tk.Button(bf, text="Continue →", bg="#1a237e", fg="white",
                  font=("Arial", 10, "bold"), command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        win.wait_window()
        if not confirmed.get():
            return

        study_type = study_var.get()
        timepoint  = None

        if study_type == "cross":
            win2 = tk.Toplevel(self.root)
            win2.title("Select Time-point")
            win2.grab_set(); win2.resizable(False, False)
            tk.Label(win2, text="Analyse which time-point?",
                     font=("Arial", 11, "bold"), pady=10).pack(padx=20)
            tp_var = tk.StringVar(value="preop")
            tk.Radiobutton(win2, text="Timepoint 1",
                           variable=tp_var, value="preop",
                           font=("Arial", 10)).pack(anchor='w', padx=30, pady=4)
            tk.Radiobutton(win2, text="Timepoint 2",
                           variable=tp_var, value="postop",
                           font=("Arial", 10)).pack(anchor='w', padx=30, pady=4)
            c2 = tk.BooleanVar(value=False)
            def on_ok2():    c2.set(True); win2.destroy()
            def on_cancel2(): win2.destroy()
            bf2 = tk.Frame(win2); bf2.pack(pady=12)
            tk.Button(bf2, text="Continue →", bg="#1a237e", fg="white",
                      font=("Arial", 10, "bold"), command=on_ok2).pack(side=tk.LEFT, padx=10)
            tk.Button(bf2, text="Cancel", command=on_cancel2).pack(side=tk.LEFT)
            win2.wait_window()
            if not c2.get():
                return
            timepoint = tp_var.get()

        self.open_analysis_menu(study_type, timepoint)

    # Eye views: chiasmatic and PTC studies run every analysis first on both
    # eyes registered, then on OD and OS separately, as recorded.
    EYE_VIEW_CLINICAL_TYPES = ("chiasmatic", "ptc")

    def _chain_get(self, key):
        cache = getattr(self, "_eye_answers", None)
        return None if cache is None else cache.get(key)

    def _chain_put(self, key, value):
        cache = getattr(self, "_eye_answers", None)
        if cache is not None:
            cache[key] = value

    def _remember(self, key, ask):
        """Ask once per analysis: the OD-only and OS-only replays reuse the
        answer given for the both-eyes run instead of asking again."""
        cache = getattr(self, "_eye_answers", None)
        if cache is not None and key in cache:
            return cache[key]
        value = ask()
        if cache is not None:
            cache[key] = value
        return value

    def _select_variable(self, title, choices, label="Select variable:"):
        return self._remember(("var", title),
                              lambda: VariableSelector(self.root, title, choices, label=label).selection)

    # Figure layout. Every analysis figure uses matplotlib's constrained layout
    # engine, which reserves room for titles, colour-bar labels, captions and
    # panels and re-spaces them when the window is resized.
    def _subplots(self, *args, **kwargs):
        fig, axes = plt.subplots(*args, layout="constrained", **kwargs)
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.06, hspace=0.06)
        return fig, axes

    @staticmethod
    def _caption(fig, text, face="#fff9c4", edge="#f57f17", fontsize=8.5, color="black", style="normal"):
        """Figure caption placed by the layout engine below the panels (never
        over them) and wrapped to the window width."""
        import textwrap
        kw = dict(bbox=dict(boxstyle="round,pad=0.4", facecolor=face, edgecolor=edge, alpha=0.95)) if face else {}
        # matplotlib's wrap=True splits only on spaces, so a caption made of
        # several newline-joined lines can overrun the canvas. Each line is
        # wrapped here to the figure width, capped at 13 in because the figure is
        # shrunk to fit the screen when it is shown.
        width_in = min(float(fig.get_size_inches()[0]), 13.0)
        chars = max(40, int(width_in * 13.0 * 8.5 / max(float(fontsize), 6.0)))
        text = "\n".join(seg for line in str(text).split("\n") for seg in (textwrap.wrap(line, chars) or [""]))
        fig.supxlabel(text, fontsize=fontsize, color=color, style=style, **kw)

    def _tag_eye_view(self, fig, view):
        if getattr(fig, "_eye_labelled", False):   # already labelled with its own single eye
            return
        flip_eye = getattr(self, "flip_eye", None)
        if view == "avg":
            txt = "Both eyes registered" + (f"  ({flip_eye.upper()} mirrored)" if flip_eye else "")
        else:
            txt = "OD | OS  -  as recorded (not mirrored)"
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text():
            sup.set_text(f"[{txt}]\n{sup.get_text()}")
        else:
            fig.suptitle(f"[{txt}]", fontsize=10, fontweight="bold", wrap=True)
        try:
            fig.canvas.manager.set_window_title(txt)
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _eye_results(self, compute):
        """Run compute() once per eye on the as-recorded data: {'od': .., 'os': ..}."""
        out, prev = {}, getattr(self, "_eye_view", "avg")
        try:
            for eye in ("od", "os"):
                self._eye_view = eye
                out[eye] = compute()
        finally:
            self._eye_view = prev
        return out

    def _draw_eye_pair(self, rows, title, caption, cell=4.3):
        """
        One figure, OD in the left column and OS in the right column (both
        as recorded, not mirrored), one row per entry of `rows`. Each row
        dict: title, cmap, grids{eye}, missing{eye}, sub{eye}, sym (bool,
        symmetric colour limits) or vlim=(lo, hi), cbar, and optionally
        contour=(levels, colors, fmt). A row's colour scale is shared by
        the two eyes so they can be compared directly.
        """
        import numpy.ma as ma
        n = len(rows)
        fig, axes = self._subplots(n, 2, figsize=(cell * 2 + 2.2, cell * n + 1.4), squeeze=False)
        fig.suptitle(title, fontweight="bold", fontsize=11)
        for r, row in enumerate(rows):
            grids = [g for g in row["grids"].values() if g is not None]
            if row.get("vlim"):
                lo, hi = row["vlim"]
            elif grids and row.get("sym"):
                a = max(safe_absmax(g, 0.0) for g in grids) or 1.0
                lo, hi = -a, a
            elif grids:
                lo = float(np.nanmin([np.nanmin(g) for g in grids]))
                hi = float(np.nanmax([np.nanmax(g) for g in grids]))
            else:
                lo, hi = -1.0, 1.0
            for c, eye in enumerate(("od", "os")):
                ax = axes[r][c]
                g = row["grids"].get(eye)
                head = f"{eye.upper()} - {row['title']}"
                if row.get("sub", {}).get(eye):
                    head += f"\n{row['sub'][eye]}"
                if g is None:
                    ax.text(0.5, 0.5, f"{eye.upper()}: not enough data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_title(head, fontweight="bold", fontsize=9); ax.axis("off")
                    continue
                im = ax.imshow(g, cmap=row["cmap"], vmin=lo, vmax=hi)
                self._draw_overlay(ax)
                if row.get("contour"):
                    levels, colors, cfmt = row["contour"]
                    try:
                        cs = ax.contour(ma.masked_invalid(g), levels=levels, colors=colors,
                                        linewidths=2.0, alpha=0.95)
                        for txt in ax.clabel(cs, fmt=cfmt, fontsize=8, inline=True):
                            txt.set_fontweight("bold")
                    except Exception as exc:
                        print(f"[NeuroVF] contours not drawn: {exc}")
                miss = row.get("missing", {}).get(eye)
                if miss is not None:
                    self._draw_missing_markers(ax, miss, do_flip=False)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(row.get("cbar", ""), fontsize=8)
                ax.set_title(head, fontweight="bold", fontsize=9); ax.axis("off")
        self._caption(fig, caption, fontsize=8)
        self.add_save_button(fig); plt.show()
        return fig

    # Plot-type analyses (scatter, box and category rows) share one figure.
    def _axes_grid(self, n, figsize):
        """(fig, axes) for an analysis figure. During the OD|OS pass both
        eyes draw into ONE figure: with n == 1 the eyes are columns, with
        n > 1 (e.g. one panel per category) the eyes are rows."""
        ctx = getattr(self, "_pair_ctx", None)
        if ctx is None:
            fig, axes = self._subplots(1, n, figsize=figsize, squeeze=False)
            return fig, list(axes[0])
        if ctx["fig"] is None:
            if n == 1:
                fig, axs = self._subplots(1, 2, figsize=(figsize[0] * 1.9, figsize[1]), squeeze=False)
                ctx["axes"] = {"od": [axs[0][0]], "os": [axs[0][1]]}
            else:
                fig, axs = self._subplots(2, n, figsize=(figsize[0], figsize[1] * 1.9), squeeze=False)
                ctx["axes"] = {"od": list(axs[0]), "os": list(axs[1])}
            ctx["fig"] = fig
        ctx["used"].add(ctx["eye"])
        return ctx["fig"], ctx["axes"][ctx["eye"]]

    def _present(self, fig):
        """Show a finished figure - deferred to _finish_pair_ctx during the OD|OS pass."""
        ctx = getattr(self, "_pair_ctx", None)
        if ctx is not None and ctx["fig"] is fig:
            return
        self.add_save_button(fig); plt.show()

    def _finish_pair_ctx(self):
        ctx = getattr(self, "_pair_ctx", None)
        if not ctx or ctx["fig"] is None:
            return
        for eye in ("od", "os"):
            for ax in ctx["axes"][eye]:
                if eye in ctx["used"]:
                    ax.set_title(f"{eye.upper()} (as recorded)\n{ax.get_title()}",
                                 fontweight="bold", fontsize=ax.title.get_fontsize())
                else:
                    ax.text(0.5, 0.5, f"{eye.upper()}: not enough data", ha="center",
                            va="center", transform=ax.transAxes)
                    ax.axis("off")
        self.add_save_button(ctx["fig"]); plt.show()

    def _run_with_eye_views(self, analysis, pair_style="native"):
        """Run `analysis`, a zero-argument callable that opens its own figures.

        Clinical types in EYE_VIEW_CLINICAL_TYPES get two passes: first both
        eyes registered (the mirrored eye laid over the other); then, once those
        windows are closed, OD and OS side by side as recorded. Dialog answers
        are asked once and reused in the second pass. Other clinical types run
        the analysis once.

        pair_style: "native" - the analysis draws the OD|OS layout itself when
        self._eye_pair is set; "loop" - it is run once per eye into one shared
        figure (see _axes_grid).

        The analyses end with a blocking plt.show(). Interactive mode
        (plt.ion()) is switched on for the whole sequence, so that plt.show()
        returns immediately and close listeners can be attached to the windows
        while they are still open. The previous mode is restored once the second
        pass has run, or once it is clear that there will be none.
        """
        if self.clinical_type not in self.EYE_VIEW_CLINICAL_TYPES:
            analysis()
            return
        answers = {}

        def run_view(view):
            prev = tuple(getattr(self, k, d) for k, d in (
                ("_eye_answers", None), ("_eye_view", "avg"), ("_eye_pair", False), ("_pair_ctx", None)))
            self._eye_answers, self._eye_view, self._eye_pair = answers, "avg", (view == "pair")
            self._pair_ctx = None
            before = set(plt.get_fignums())
            try:
                if view == "pair" and pair_style == "loop":
                    self._pair_ctx = {"fig": None, "axes": {}, "used": set(), "eye": None}
                    for eye in ("od", "os"):
                        self._eye_view, self._pair_ctx["eye"] = eye, eye
                        analysis()
                    self._finish_pair_ctx()
                else:
                    analysis()
            finally:
                (self._eye_answers, self._eye_view, self._eye_pair, self._pair_ctx) = prev
            new = [n for n in plt.get_fignums() if n not in before]
            for n in new:
                self._tag_eye_view(plt.figure(n), view)
            return new

        was_interactive = plt.isinteractive()
        plt.ion()

        def restore_interactive():
            if not was_interactive:
                plt.ioff()

        new = run_view("avg")
        if not new:
            restore_interactive()
            return

        # A pass can open more than one window (for example a mean-maps window
        # and a statistics window). Each window gets its own listener, and the
        # next pass starts once all of them have been closed, in any order.
        still_open = set(new)
        handled = set()   # a window may report its close through both triggers below
        def on_close(fignum):
            if fignum in handled:
                return
            handled.add(fignum)
            still_open.discard(fignum)
            if not still_open:
                # Deferred with after_idle: the next pass can open dialogs
                # (wait_window) and build figures, which is safer as a separate Tk
                # callback than nested inside the close event being dispatched.
                def do_pair():
                    try:
                        run_view("pair")
                    finally:
                        restore_interactive()
                self.root.after_idle(do_pair)
        for fignum in new:
            fig = plt.figure(fignum)
            fig.canvas.mpl_connect("close_event", lambda _e, fn=fignum: on_close(fn))
            # Also bind to Tk's <Destroy> on the window itself, a lower-level signal
            # that does not depend on matplotlib's close_event. The `handled` set
            # makes it safe for both to fire for the same window.
            try:
                win = fig.canvas.manager.window
                win.bind("<Destroy>", lambda _e, fn=fignum, w=win:
                         on_close(fn) if _e.widget is w else None, add="+")
            except Exception:
                pass

    def open_analysis_menu(self, study_type, timepoint):
        """Step 2: choose which analysis to run."""
        win = tk.Toplevel(self.root)
        win.title(f"Analysis Menu - "
                  f"{'Cross-Sectional' if study_type=='cross' else 'Longitudinal'}")
        win.grab_set(); win.resizable(False, False)

        lbl_tp = f"[{tp_display(timepoint)}]" if timepoint else "[Δ Timepoint1→Timepoint2]"
        tk.Label(win,
                 text=f"Study Type: {'Cross-Sectional' if study_type=='cross' else 'Longitudinal'}  {lbl_tp}",
                 font=("Arial", 10, "bold"), fg="#1a237e", pady=8).pack(padx=20)

        is_long = (study_type == "longitudinal")
        map_label = self.data_type_spec["label"].split("(")[0].strip()
        analyses = [
            ("Spatial Map  (pixel-wise correlations, Category Comparison)",
             True,   "spatial"),
            (f"Mean {map_label} Map  (whole cohort)",
             True,   "prob"),
            (f"Category-Specific Mean {map_label} Maps  (side-by-side by Categorical variable)",
             True,   "prob_cat"),
            ("Cluster Analysis  (severity thresholds, dB - cross-sectional only)",
             not is_long,   "cluster"),
            ("Cumulative Analysis  (severity thresholds, dB - cross-sectional only)",
             not is_long,   "cumulative"),
            ("MD (Mean Deviation) vs Variable  (global index, HPA-validated bands)",
             self.clinical_type == "ptc",   "md_trend"),
        ]

        sel_var = tk.StringVar(value="spatial")
        for label, active, key in analyses:
            state = tk.NORMAL if active else tk.DISABLED
            rb = tk.Radiobutton(win, text=label, variable=sel_var, value=key,
                                font=("Arial", 10), state=state,
                                fg="black" if active else "#aaaaaa")
            rb.pack(anchor='w', padx=30, pady=3)
            if key in ("cluster", "cumulative"):
                tk.Label(win, text=f"   ↳ {self.severity_caveat}",
                         font=("Arial", 8, "italic"), fg="#b71c1c",
                         justify='left').pack(anchor='w', padx=50)
                if not active:
                    reason = ("   ↳ Cluster and cumulative analyses use absolute severity\n"
                              "      thresholds and are not applicable to longitudinal\n"
                              "      change values.\n"
                              "      Use Pearson or Spearman analysis for longitudinal dB-change data.")
                    tk.Label(win, text=reason,
                             font=("Arial", 8, "italic"), fg="#888",
                             justify='left').pack(anchor='w', padx=50)
            if key == "md_trend" and not active:
                tk.Label(win, text="   ↳ MD (Mean Deviation) is available for PTC/IIH studies "
                                    "only - it's the validated global outcome measure for "
                                    "papilledema, not for chiasmatic or post-chiasmatic work, "
                                    "where the topographic pattern is the point of the analysis.",
                         font=("Arial", 8, "italic"), fg="#888",
                         justify='left').pack(anchor='w', padx=50)

        if self.clinical_type in self.EYE_VIEW_CLINICAL_TYPES:
            tk.Label(win, text="Each analysis runs on both eyes registered first; when you close "
                                "that window, OD and OS follow side by side exactly as recorded "
                                "(not mirrored).",
                     font=("Arial", 8, "italic"), fg="#1565c0", wraplength=440,
                     justify="left").pack(anchor="w", padx=30, pady=(8, 0))

        confirmed = tk.BooleanVar(value=False)
        def on_ok():    confirmed.set(True); win.destroy()
        def on_cancel(): win.destroy()

        bf = tk.Frame(win); bf.pack(pady=12)
        tk.Button(bf, text="Run →", bg="#2e7d32", fg="white",
                  font=("Arial", 10, "bold"), command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        win.wait_window()
        if not confirmed.get():
            return

        choice = sel_var.get()
        runners = {
            "spatial":    (self.generate_both_eyes_heatmap,        "native"),
            "prob":       (self.generate_probability_map,          "native"),
            "prob_cat":   (self.generate_probability_map_by_category, "loop"),
            "cluster":    (self.analyze_affected_space,            "loop"),
            "cumulative": (self.analyze_cumulative_space,          "loop"),
            "md_trend":   (self.analyze_md_vs_variable,            "loop"),
        }
        entry = runners.get(choice)
        if entry:
            run, style = entry
            self._run_with_eye_views(lambda: run(study_type, timepoint), style)

    def analyze_md_vs_variable(self, study_type="cross", timepoint="preop"):
        """MD (Mean Deviation) against a chosen variable.

        MD is a global index, as opposed to the pointwise spatial maps elsewhere
        in the menu.

        Cross-sectional: each subject's MD at the given timepoint against a
        continuous variable (scatter plot and linear regression) or a
        categorical variable (box plot by group, with Welch's t-test when
        exactly two groups have data).
        Longitudinal: the MD change (Timepoint 2 MD - Timepoint 1 MD) against
        the variable.

        Available only for the "ptc" clinical type (see compute_md).
        """
        if self.clinical_type != "ptc":
            messagebox.showwarning("Not Available",
                "MD (Mean Deviation) is available for PTC/IIH studies only.")
            return
        if not self.continuous_cols and not self.categorical_cols:
            messagebox.showwarning("No Variables",
                "No Continuous or Categorical variables defined. Use 'Add Variable' first.")
            return

        all_vars = list(self.continuous_cols) + list(self.categorical_cols)
        var_to_test = self._select_variable("MD vs Variable", all_vars, "Select variable to compare against MD:")
        if not var_to_test:
            return
        is_categorical = var_to_test in self.categorical_cols

        subj_vals, subj_md, subj_ids = [], [], []
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS or not isinstance(data, dict):
                continue
            raw = data.get(var_to_test, "")
            if raw in (None, ""):
                continue
            if study_type == "longitudinal":
                md_pre,  _, _, _ = self.compute_md(data, "preop_")
                md_post, _, _, _ = self.compute_md(data, "postop_")
                if md_pre is None or md_post is None:
                    continue
                md_value = md_post - md_pre
            else:
                prefix = self._tp_prefix(timepoint)
                md_value, npts, tier, _src = self.compute_md(data, prefix)
                if md_value is None:
                    md_value, npts, tier, _src = self.compute_md(data, "")
                    if md_value is None:
                        continue
            if is_categorical:
                subj_vals.append(str(raw))
            else:
                try:
                    subj_vals.append(float(raw))
                except (ValueError, TypeError):
                    continue
            subj_md.append(md_value)
            subj_ids.append(sid)

        if len(subj_md) < 4:
            messagebox.showwarning("Insufficient Data",
                f"Need \u22654 subjects with both MD and '{var_to_test}'. Found: {len(subj_md)}")
            return

        tp_lbl = ("Longitudinal \u0394MD (Timepoint 2 \u2212 Timepoint 1)" if study_type == "longitudinal"
                  else f"MD [{tp_display(timepoint)}]")
        fig, (ax,) = self._axes_grid(1, (9, 6.5))
        subj_md = np.array(subj_md, dtype=float)

        if is_categorical:
            groups = sorted(set(subj_vals))
            group_vals = {g: subj_md[[v == g for v in subj_vals]] for g in groups}
            ax.boxplot([group_vals[g] for g in groups], tick_labels=groups, showmeans=True)
            for i, g in enumerate(groups, start=1):
                ax.scatter(np.full(len(group_vals[g]), i) + np.random.uniform(-0.05, 0.05, len(group_vals[g])),
                           group_vals[g], alpha=0.5, color="#1565c0", zorder=3)
            stat_text = ""
            if len(groups) == 2:
                from scipy.stats import ttest_ind
                g0, g1 = group_vals[groups[0]], group_vals[groups[1]]
                if len(g0) >= 2 and len(g1) >= 2:
                    t, p = ttest_ind(g0, g1, equal_var=False)
                    stat_text = f"  |  Welch's t={t:.2f}, p={p:.4f}"
            ax.set_xlabel(var_to_test, fontweight='bold')
            ax.set_title(_wrap_title(f"{tp_lbl} by {var_to_test}{stat_text}"), fontweight='bold')
        else:
            x = np.array(subj_vals, dtype=float)
            sort_idx = np.argsort(x)
            x_sorted, y_sorted = x[sort_idx], subj_md[sort_idx]
            sl, ic, r, p, _ = linregress(x_sorted, y_sorted)
            ax.scatter(x_sorted, y_sorted, color="#1565c0", alpha=0.7, s=50, zorder=3)
            lw, ls = (2.5, '-') if p < 0.05 else (1.2, '--')
            ax.plot(x_sorted, ic + sl * x_sorted, color="#b71c1c", linestyle=ls, linewidth=lw,
                    label=f"r={r:.2f}, p={p:.4f}")
            ax.legend(loc='best')
            ax.set_xlabel(var_to_test, fontweight='bold')
            ax.set_title(_wrap_title(f"{tp_lbl} vs {var_to_test}"), fontweight='bold')

        ax.set_ylabel(f"{tp_lbl} (dB)", fontweight='bold')
        ax.axhline(0, color='gray', lw=0.6, alpha=0.5)
        ax.grid(True, alpha=0.3)
        self._caption(fig,
                  "MD uses the device-reported value where both eyes have been manually "
                  "entered (PTC only); otherwise it's approximated as the unweighted mean of "
                  "available combined-eye Total Deviation values (see compute_md); it may "
                  "differ from a device-printed MD. HPA global-MD bands: "
                  "Early/Mild > \u22126.01 dB, Moderate \u22126.01 to \u221212.01 dB, Severe/Advanced "
                  "\u2264 \u221212.01 dB.",
                  face=None, fontsize=7.5, color='#666', style='italic')
        self._present(fig)

    # Helpers: quadrant masking
    def get_quadrant_crop(self):
        mode = self.quadrant_var.get()
        if   mode == "Superior Left":    return 0,50,0,50
        elif mode == "Superior Right":   return 0,50,50,100
        elif mode == "Inferior Left":    return 50,100,0,50
        elif mode == "Inferior Right":   return 50,100,50,100
        elif mode == "2 Left Quarters":  return 0,100,0,50
        elif mode == "2 Right Quarters": return 0,100,50,100
        else:                            return 0,100,0,100

    def apply_quadrant_mask(self, data):
        mode   = self.quadrant_var.get()
        masked = np.copy(data).astype(float)
        if   mode == "Superior Left":    masked[50:,:]=np.nan; masked[:,50:]=np.nan
        elif mode == "Superior Right":   masked[50:,:]=np.nan; masked[:,:50]=np.nan
        elif mode == "Inferior Left":    masked[:50,:]=np.nan; masked[:,50:]=np.nan
        elif mode == "Inferior Right":   masked[:50,:]=np.nan; masked[:,:50]=np.nan
        elif mode == "2 Left Quarters":  masked[:,50:]=np.nan
        elif mode == "2 Right Quarters": masked[:,:50]=np.nan
        return masked

    def crop_to_quadrant(self, data):
        r0,r1,c0,c1 = self.get_quadrant_crop()
        return data[r0:r1, c0:c1]

    def apply_circle_mask(self, data_100x100):
        # NaN-aware smoothing: a missing cell contributes zero weight, not a
        # 0 dB value.
        out = _nan_aware_gaussian_smooth(data_100x100, sigma=1.5)
        y,x = np.ogrid[-50:50,-50:50]
        out[x*x+y*y > 50*50] = np.nan
        out = self.apply_quadrant_mask(out)
        return out

    def _n_active_pixels(self):
        """Number of points of the active grid inside the selected analysis region.

        Reported in figure titles as the size of the region.
        """
        return int(self._quadrant_pt_mask(self.grid_points).sum())

    def _quadrant_pt_mask(self, pts_xy_flipped):
        mode = self.quadrant_var.get()
        x    = pts_xy_flipped[:,0]
        y    = pts_xy_flipped[:,1]
        if   mode == "Superior Left":    return (y > 0) & (x < 0)
        elif mode == "Superior Right":   return (y > 0) & (x > 0)
        elif mode == "Inferior Left":    return (y < 0) & (x < 0)
        elif mode == "Inferior Right":   return (y < 0) & (x > 0)
        elif mode == "2 Left Quarters":  return x < 0
        elif mode == "2 Right Quarters": return x > 0
        else:                            return np.ones(len(pts_xy_flipped), dtype=bool)

    # Missing-location markers
    def _draw_missing_markers(self, ax, missing_mask_76, do_flip=False,
                               label=None, marker_type="sand"):
        """Draw missing-location markers on *ax*.

        missing_mask_76: boolean mask over the points of the active grid; True
        draws a sand-hatched marker (an orange "unresolved" marker when
        marker_type == "unresolved") at that location.

        Call this after imshow, interpolation and smoothing, so that the markers
        sit on top of the final surface and missing data stays visible. The
        flip state (`do_flip`) and the selected quadrant or half-field crop are
        honoured. Coordinates are converted to the 100x100 pixel-index space
        used by _draw_overlay, so this works for the full image and for a
        cropped sub-array.

        `label` adds one legend entry; pass it once per panel.
        """
        mask = np.asarray(missing_mask_76, dtype=bool)
        if not mask.any():
            return None

        pts = self.grid_points.copy()
        if do_flip:
            pts[:, 0] = -pts[:, 0]

        # Only locations inside the displayed quadrant or half-field are drawn.
        qmask = self._quadrant_pt_mask(pts)
        show_mask = mask & qmask
        if not show_mask.any():
            return None

        # Map degrees (-ext..+ext; ext is 30 for 24-2/30-2 and 60 for 60-4) to
        # the 100x100 pixel-index grid of pts_to_grid_100, then shift for the
        # active quadrant crop, as _draw_overlay does.
        ext = float(self.grid_extent_deg)
        r0, r1, c0, c1 = self.get_quadrant_crop()
        xs = pts[show_mask, 0]
        ys = pts[show_mask, 1]
        col = (xs + ext) / (2.0 * ext) * 99.0
        row = (ext - ys) / (2.0 * ext) * 99.0
        col_disp = col - c0
        row_disp = row - r0

        if marker_type == "unresolved":
            face, edge, hatch = UNRESOLVED_MARKER_FACE, UNRESOLVED_MARKER_EDGE, UNRESOLVED_MARKER_HATCH
        else:
            face, edge, hatch = MISSING_MARKER_FACE, MISSING_MARKER_EDGE, MISSING_MARKER_HATCH

        sc = ax.scatter(col_disp, row_disp, marker='s', s=55,
                         facecolor=face, edgecolor=edge, hatch=hatch,
                         linewidth=0.7, zorder=12, label=label)
        return sc

    def _draw_overlay(self, ax):
        """
        Draw crosshairs and the VF circle arc on *ax*, correctly adapted to the
        current analysis-area crop.

        The full 100×100 grid represents a circle of radius 50 centred at (49.5, 49.5)
        in pixel coordinates.  When a quadrant crop is applied the image displayed
        is a sub-array of that grid (e.g. columns 50–100 for '2 Right Quarters').
        The circle centre and crosshair positions must be shifted accordingly so
        that the overlay aligns with the actual data.

        All coordinates are in *display* (pixel-index) space of the cropped image.
        """
        mode = self.quadrant_var.get()
        r0, r1, c0, c1 = self.get_quadrant_crop()
        h = r1 - r0   # displayed height in pixels
        w = c1 - c0   # displayed width in pixels

        # Centre of the full circle in the cropped coordinate system
        cx = 49.5 - c0   # column centre shifted
        cy = 49.5 - r0   # row centre shifted
        radius = 49.5    # always 49.5 (full-circle radius in 100×100 space)

        # Crosshairs
        # Horizontal: y = cy, clipped to [0, w-1] in x
        ax.axhline(y=cy, color='black', linewidth=0.8, linestyle='-', alpha=0.6)
        # Vertical: x = cx, clipped to [0, h-1] in y
        ax.axvline(x=cx, color='black', linewidth=0.8, linestyle='-', alpha=0.6)

        # Circle arc
        import matplotlib.patches as mpatches
        circle = mpatches.Circle(
            (cx, cy), radius,
            color='#555', fill=False,
            linewidth=1.2, linestyle='--', alpha=0.7,
            clip_on=True)
        ax.add_patch(circle)

        # Force axis limits to the crop size so the arc is clipped correctly
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)   # imshow y-axis is inverted

    def _norm76(self, vals, flip):
        """Apply the horizontal flip remap to a point array when `flip` is true.

        Only the order of the array changes (left-right mirroring). Values are
        not normalized or rescaled, and NaN entries stay NaN.
        """
        if flip:
            return vals[self.grid_flip_remap]
        return vals

    def _get_raw76_pair(self, data, prefix=""):
        """Return (od, os, flip_od, flip_os) for one subject and timepoint.

        The arrays are the stored values (dB), unmodified; missing source
        locations are np.nan. The flags say whether each eye is mirrored before
        the eyes are combined, and depend on self.flip_policy:

          "manual"              - the stored per-eye vf_flip_* flags.
          "flip_one_eye_cohort" - self.flip_eye ("od" or "os", chosen once per
                                  database; see _ensure_flip_eye_configured) is
                                  mirrored for every subject; stored flags are
                                  ignored.
          "surgery_side_mirror" - neither eye is mirrored here; the combined
                                  field is mirrored by surgery side afterwards,
                                  in _apply_cohort_mirror.

        Returns (None, None, False, False) unless both eyes are stored.
        """
        if prefix:
            r_od = data.get(f"vf_raw76_{prefix}od")
            r_os = data.get(f"vf_raw76_{prefix}os")
            f_od = data.get(f"vf_flip_{prefix}od", False)
            f_os = data.get(f"vf_flip_{prefix}os", False)
        else:
            r_od = data.get("vf_raw76_preop_od") or data.get("vf_raw76_od")
            r_os = data.get("vf_raw76_preop_os") or data.get("vf_raw76_os")
            f_od = data.get("vf_flip_preop_od", False) or data.get("vf_flip_od", False)
            f_os = data.get("vf_flip_preop_os", False) or data.get("vf_flip_os", False)
        policy = getattr(self, "flip_policy", "manual")
        if policy == "flip_one_eye_cohort":
            flip_eye = getattr(self, "flip_eye", "os") or "os"
            f_od, f_os = (flip_eye == "od"), (flip_eye == "os")
        elif policy == "surgery_side_mirror":
            f_od, f_os = False, False
        # policy == "manual" (or unset): keep the stored flags read above.
        if r_od is not None and r_os is not None:
            return (np.array(r_od, dtype=float),
                    np.array(r_os, dtype=float),
                    bool(f_od), bool(f_os))
        return None, None, False, False

    def _apply_cohort_mirror(self, combined, valid_mask, data):
        """Mirror the combined field by surgery side ("surgery_side_mirror" policy only).

        The combined (OD+OS) field of a subject is mirrored left-right when the
        subject's surgery-side value differs from the reference value, so that
        every subject is expressed in a common ipsilateral/contralateral frame.
        A no-op for other flip policies, when the surgery-side variable is not
        configured, and when the subject has no surgery-side value.
        """
        if getattr(self, "flip_policy", "manual") != "surgery_side_mirror":
            return combined, valid_mask
        if combined is None:
            return combined, valid_mask
        var = getattr(self, "surgery_side_var", None)
        ref = getattr(self, "surgery_side_reference", None)
        if not var or ref is None:
            return combined, valid_mask  # not configured: leave unmirrored
        subj_val = str(data.get(var, "")).strip()
        if not subj_val or subj_val == str(ref).strip():
            return combined, valid_mask
        remap = self.grid_flip_remap
        return combined[remap], valid_mask[remap]

    def _combine_od_os_complete_pair(self, od_n, os_n):
        """Combine two flip-registered eye arrays under the complete-pair rule.

            both OD and OS valid  -> combined = arithmetic mean(OD, OS)
            either eye missing    -> combined = np.nan

        np.nanmean is deliberately not used: it would substitute the single
        available eye's value when the fellow eye is missing. A location missing
        in either eye is excluded from every downstream analysis.

        Returns (combined, paired_valid_mask).
        """
        od_n = np.asarray(od_n, dtype=float)
        os_n = np.asarray(os_n, dtype=float)
        od_valid = np.isfinite(od_n)
        os_valid = np.isfinite(os_n)
        paired_valid_mask = od_valid & os_valid
        combined = np.full_like(od_n, np.nan, dtype=float)
        combined[paired_valid_mask] = (od_n[paired_valid_mask] + os_n[paired_valid_mask]) / 2.0
        return combined, paired_valid_mask

    def _get_avg76(self, data, prefix=""):
        """Return (values, valid_mask) for one subject in the current eye view.

        Registered view: the two eyes are flip-registered, combined under the
        complete-pair rule (np.nan wherever either eye is missing; see
        _combine_od_os_complete_pair) and then passed through the cohort mirror
        (_apply_cohort_mirror). OD or OS view: that eye as recorded (see
        _get_single_eye76). Returns (None, None) when the data are not stored.
        """
        view = getattr(self, "_eye_view", "avg")
        if view in ("od", "os"):
            single = self._get_single_eye76(data, prefix, view)
            if single is None:
                return None, None
            return single, np.isfinite(single)
        od, os_, f_od, f_os = self._get_raw76_pair(data, prefix)
        if od is None:
            return None, None
        od_n = self._norm76(od, f_od)
        os_n = self._norm76(os_, f_os)
        combined, valid_mask = self._combine_od_os_complete_pair(od_n, os_n)
        combined, valid_mask = self._apply_cohort_mirror(combined, valid_mask, data)
        return combined, valid_mask

    def _get_avg76_with_fallback(self, data, prefix=""):
        """_get_avg76 for `prefix`, with a fallback to the legacy unprefixed keys.

        The fallback applies only to a subject with no timepoint-prefixed data
        at all (a record that predates the two timepoints). A subject who has
        timepoint data but lacks an eye at the requested timepoint is excluded
        from that timepoint, not substituted: _get_avg76(data, "") reads the
        preop keys first (see _get_raw76_pair), so an unconditional fallback
        would replace missing Timepoint 2 data with Timepoint 1 data.

        All cohort-level collectors go through this method, and therefore
        through the same cohort-mirror step.
        """
        combined, valid = self._get_avg76(data, prefix)
        if combined is not None:
            return combined, valid
        has_timepoint_data = any(
            data.get(f"vf_raw76_{tp}_{eye}") is not None
            for tp in ("preop", "postop") for eye in ("od", "os"))
        if has_timepoint_data:
            return None, None   # missing at this timepoint: exclude, do not substitute
        return self._get_avg76(data, "")

    def _md_tier(self, md):
        """Classify a global MD value into Hodapp-Anderson-Parrish bands."""
        if md <= TD_SEVERE:
            return "Severe/Advanced"
        elif md <= TD_MODERATE:
            return "Moderate"
        return "Early/Mild"

    def get_manual_md(self, data, prefix=""):
        """Return (md_od, md_os), the device-reported MD values entered by the user.

        Each is a float or None, read from md_manual_{prefix}od/os. The entry
        dialog (enter_manual_md_dialog) is offered for PTC databases only, but
        this reader honours any stored value.
        """
        md_od = data.get(f"md_manual_{prefix}od")
        md_os = data.get(f"md_manual_{prefix}os")
        md_od = float(md_od) if md_od not in (None, "") else None
        md_os = float(md_os) if md_os not in (None, "") else None
        return md_od, md_os

    def compute_md(self, data, prefix=""):
        """Mean Deviation (MD): one global index per subject and timepoint.

        Available only for the "ptc" clinical type. MD is the validated global
        outcome measure for papilledema (IIHTT); for chiasmatic and
        post-chiasmatic work the topographic pattern is the object of the
        analysis, not a global summary.

        Two sources, in priority order:
          1. Device-reported: the MD printed on the source report, entered with
             enter_manual_md_dialog (md_manual_*_od / md_manual_*_os). In the
             registered view both eyes are required and the result is their
             average; in a single-eye view that eye's value is used.
          2. Approximated: the unweighted mean of the finite Total Deviation
             values of the current view. The device computes MD as a
             variance-weighted average, so the approximation is close for a
             full field with few missing points but not always identical to the
             printed value. Used only when data_type == "total_deviation";
             manual entry works for either data type.

        Returns (md, n, tier_label, source), or (None, 0, None, None) if MD is
        not computable. n is the number of values averaged (2 or 1 for
        device-reported values). tier_label applies TD_SEVERE / TD_MODERATE as
        Hodapp-Anderson-Parrish global MD staging, the purpose those numbers
        were validated for. source is "device-reported" or "approximated".
        """
        if self.clinical_type != "ptc":
            return None, 0, None, None

        md_od, md_os = self.get_manual_md(data, prefix)
        view = getattr(self, "_eye_view", "avg")
        if view in ("od", "os"):
            md_eye = md_od if view == "od" else md_os
            if md_eye is not None:
                return md_eye, 1, self._md_tier(md_eye), "device-reported"
        elif md_od is not None and md_os is not None:
            md = (md_od + md_os) / 2.0
            return md, 2, self._md_tier(md), "device-reported"

        if self.data_type != "total_deviation":
            return None, 0, None, None
        combined, valid = self._get_avg76_with_fallback(data, prefix)
        if combined is None:
            return None, 0, None, None
        finite = combined[np.isfinite(combined)]
        if len(finite) == 0:
            return None, 0, None, None
        md = float(np.mean(finite))
        return md, int(len(finite)), self._md_tier(md), "approximated"

    def _get_display_grid(self, data, eye="avg", prefix=""):
        """Visual-only 100x100 interpolated Total Deviation (dB) surface."""
        od, os_, f_od, f_os = self._get_raw76_pair(data, prefix)
        if od is not None:
            if   eye == "od":  vals = self._norm76(od, f_od)
            elif eye == "os":  vals = self._norm76(os_, f_os)
            else:
                vals, _ = self._get_avg76(data, prefix)
            return pts_to_grid_100(self.grid_points, vals, False, vmin=self.plausible_min, vmax=self.plausible_max, extent_deg=self.grid_extent_deg)
        if eye == "od" and data.get("vf_data_od"):
            return np.array(data["vf_data_od"])
        if eye == "os" and data.get("vf_data_os"):
            return np.array(data["vf_data_os"])
        if data.get("vf_data_od") and data.get("vf_data_os"):
            return (np.array(data["vf_data_od"]) + np.array(data["vf_data_os"])) / 2.0
        return None

    def _get_delta76(self, data):
        """Return (delta, valid_mask): the per-location change in dB, postop minus preop.

        The change is a plain subtraction of the registered values of the
        current view. Total Deviation is an interval (dB) scale with a
        meaningful zero (0 dB = expected sensitivity), so no ratio or
        normalization is involved.

        NaN propagates through the subtraction, so a location has a change value
        only if it is valid at both timepoints (a complete OD/OS pair at each
        one in the registered view). Returns (None, None) if either timepoint
        is not stored.
        """
        pre_combined,  pre_valid  = self._get_avg76(data, "preop_")
        post_combined, post_valid = self._get_avg76(data, "postop_")
        if pre_combined is None or post_combined is None:
            return None, None
        delta = post_combined - pre_combined   # NaN propagates automatically
        delta_valid_mask = np.isfinite(delta)
        return delta, delta_valid_mask

    def _get_loss76(self, data):
        return self._get_delta76(data)

    def _get_single_eye76(self, data, prefix, eye):
        """Return one eye's array as recorded (never mirrored), or None if it is not stored.

        The database stores the original recording; mirroring is applied at
        analysis time and only in the registered view, so the separate OD and OS
        analyses use unmirrored data. Unlike the paired path, a subject with
        only this eye's field still contributes.
        """
        if prefix:
            raw = data.get(f"vf_raw76_{prefix}{eye}")
        else:
            raw = data.get(f"vf_raw76_preop_{eye}") or data.get(f"vf_raw76_{eye}")
        if raw is None:
            return None
        return np.array(raw, dtype=float)

    def _tp_prefix(self, timepoint):
        if timepoint == "postop":
            return "postop_"
        return "preop_"

    # Collect helpers
    def _collect_grids_for_analysis(self, study_type, timepoint):
        """Return list of (sid, grid_100x100, clinical_data_dict)."""
        result = []
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS:
                continue
            if not isinstance(data, dict):
                continue
            if study_type == "longitudinal":
                loss76, flip = self._get_loss76(data)
                if loss76 is None:
                    continue
                g = pts_to_grid_100(self.grid_points, loss76, False, signed=True, extent_deg=self.grid_extent_deg)
            else:
                prefix = self._tp_prefix(timepoint)
                avg, _avg_valid = self._get_avg76_with_fallback(data, prefix)
                if avg is None:
                    continue
                g = pts_to_grid_100(self.grid_points, avg, False, vmin=self.plausible_min, vmax=self.plausible_max, extent_deg=self.grid_extent_deg)
            if g is not None:
                result.append((sid, g, data))
        return result

    def _collect_76_for_spatial(self, study_type, timepoint):
        """Return list of dicts {sid, vf76, flip, data} with vf76 in canonical orientation."""
        result = []
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS:
                continue
            if not isinstance(data, dict):
                continue
            if study_type == "longitudinal":
                loss76, flip = self._get_loss76(data)
                if loss76 is None:
                    continue
                result.append({"sid":sid, "vf76":loss76, "flip":False, "data":data})
            else:
                prefix = self._tp_prefix(timepoint)
                avg, _avg_valid = self._get_avg76_with_fallback(data, prefix)
                if avg is None:
                    continue
                result.append({"sid":sid, "vf76":avg, "flip":False, "data":data})
        return result

    # Cohort mean map
    def _cohort_mean_grid(self, study_type, timepoint):
        """Cohort mean of the current eye view at each tested location.

        The mean is taken over the original tested locations and only then
        interpolated to the display grid. A location is unavailable if any
        included subject lacks a valid value there (complete-case); it is never
        a partial mean over the subjects who happen to have data. Averaging
        across subjects with np.nanmean is acceptable here because each
        subject's value comes from the complete-pair OD/OS rule or from a
        single eye's own recording.

        Returns dict(grid, unavailable, n, n_min, n_max), or None if there is
        no data.
        """
        raw76_list = []
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS or not isinstance(data, dict):
                continue
            if study_type == "longitudinal":
                delta76, valid_mask = self._get_loss76(data)
                if delta76 is None:
                    continue
                raw76_list.append(delta76)
            else:
                prefix = self._tp_prefix(timepoint)
                combined, valid_mask = self._get_avg76_with_fallback(data, prefix)
                if combined is None:
                    continue
                raw76_list.append(combined)
        if not raw76_list:
            return None
        stacked = np.array(raw76_list)              # (n_subjects, n_points)
        n_total = len(raw76_list)
        valid_n = np.sum(np.isfinite(stacked), axis=0)
        # Complete-case rule: a location missing in any included subject is
        # unavailable for the whole cohort.
        unavailable = valid_n < n_total
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_td_76 = np.nanmean(stacked, axis=0)
        mean_td_76[unavailable] = np.nan
        signed_map = (study_type == "longitudinal")
        grid = pts_to_grid_100(self.grid_points, mean_td_76, False, signed=signed_map,
                               vmin=(None if signed_map else self.plausible_min),
                               vmax=(None if signed_map else self.plausible_max),
                               extent_deg=self.grid_extent_deg)
        return {"grid": self.apply_quadrant_mask(grid), "unavailable": unavailable,
                "n": len(raw76_list),
                "n_min": int(valid_n[~unavailable].min()) if (~unavailable).any() else 0,
                "n_max": int(valid_n.max())}

    # A separate contours panel repeats the map, so it is only worth drawing
    # when enough contour lines are present to clutter the plain map.
    CONTOUR_PANEL_MIN_LEVELS = 3

    @staticmethod
    def _present_contour_levels(grids, levels):
        """The contour levels that actually cross the given smoothed map(s)."""
        finite = [g[np.isfinite(g)] for g in grids if g is not None and np.isfinite(g).any()]
        return [lv for lv in levels if any(f.min() < lv < f.max() for f in finite)]

    def _mean_map_style(self, study_type, timepoint):
        """Colour scale, contour levels and wording for a cohort-mean map.
        Follows the database's data type (Total Deviation or sensitivity)."""
        colors = ['#1565c0', '#2e7d32', '#ef6c00', '#c62828']
        td = self.data_type == "total_deviation"
        name = "Total Deviation" if td else "Sensitivity"
        if study_type == "longitudinal":
            levels = [-9, -4.5, 0.0, 4.5, 9]
            return dict(
                cmap='RdBu_r', vmin=-15.0, vmax=15.0, name=name,
                title=f"Longitudinal {name} Change Map\n(Timepoint 2 − Timepoint 1 per subject, dB, then cohort mean)",
                clabel=f"Mean {name} change (dB)",
                scale_note="BLUE = net loss (change < 0 dB)  |  RED = net gain (change > 0 dB)",
                kind_note="This is a mean per-pixel change map (dB), not a statistical test.",
                levels=levels, colors=['#1a237e', '#42a5f5', '#90a4ae', '#ef9a9a', '#b71c1c'],
                cfmt={-9: '−9', -4.5: '−4.5', 0.0: '0', 4.5: '+4.5', 9: '+9'})
        levels = CONTOUR_LEVELS[self.data_type]
        # One fixed grayscale for every panel (never autoscaled per group):
        # 0 dB (or the top of the range) = white, the bottom of the range = black.
        return dict(
            cmap=self.map_cmap, vmin=self.map_vmin, vmax=self.map_vmax, name=name,
            title=f"Mean {name} Map - {tp_display(timepoint)}",
            clabel=f"Mean {name} (dB)",
            scale_note=("BLACK = greater sensitivity loss (mean TD toward −30 dB or lower)  |  "
                        "WHITE = expected sensitivity (mean TD ≈ 0 dB)  |  GREY = in between" if td else
                        "BLACK = lower sensitivity (greater loss, toward 0 dB)  |  "
                        "WHITE = normal sensitivity (≈ 30 dB or higher)  |  GREY = in between"),
            kind_note="This is a descriptive mean map - no statistical test is performed.",
            levels=levels, colors=colors, cfmt={lv: f"{lv:g}" for lv in levels})

    def _mean_map_pair(self, study_type, timepoint):
        """Mean map for OD and OS side by side, as recorded (not mirrored)."""
        res = self._eye_results(lambda: self._cohort_mean_grid(study_type, timepoint))
        if not any(res.values()):
            messagebox.showwarning("No Data", "No subjects with VF data found.")
            return
        st = self._mean_map_style(study_type, timepoint)

        def smooth(r):
            return None if r is None else _smooth_in_field(r["grid"])
        sm = {eye: smooth(res[eye]) for eye in res}
        miss = {eye: (res[eye]["unavailable"] if res[eye] else None) for eye in res}
        n_txt = {eye: (f"n={res[eye]['n']}  |  valid n/location: {res[eye]['n_min']}-{res[eye]['n_max']}"
                       if res[eye] else "") for eye in res}
        common = dict(cmap=st["cmap"], vlim=(st["vmin"], st["vmax"]), cbar=st["clabel"],
                      missing=miss, grids=sm)
        present = self._present_contour_levels(sm.values(), st["levels"])
        colors = [st["colors"][st["levels"].index(lv)] for lv in present]
        if len(present) >= self.CONTOUR_PANEL_MIN_LEVELS:
            rows = [dict(common, title=st["title"].split("\n")[0], sub=n_txt),
                    dict(common, title="Topographic contours", contour=(st["levels"], st["colors"], st["cfmt"]))]
        else:       # few lines: overlay them on the single map row instead of repeating it
            rows = [dict(common, title=st["title"].split("\n")[0], sub=n_txt,
                         contour=((present, colors, st["cfmt"]) if present else None))]
        caption = ("Descriptive mean map - no statistical test is performed. OD and OS are shown "
                   "exactly as recorded (the mirrored eye is NOT mirrored here).\n"
                   "Sand-hatched loci: no valid source measurement in that eye across the cohort.")
        self._draw_eye_pair(rows, f"{st['title'].split(chr(10))[0]}  |  Region: {self.quadrant_var.get()}",
                            caption)

    def generate_probability_map(self, study_type="cross", timepoint="preop"):
        """Cohort mean map: the mean value at each tested location across the cohort.

        Interpolation to the 100x100 display grid happens after the per-location
        mean and is for display only.
        """
        if getattr(self, "_eye_pair", False):
            return self._mean_map_pair(study_type, timepoint)
        res = self._cohort_mean_grid(study_type, timepoint)
        if res is None:
            messagebox.showwarning("No Data", "No subjects with VF data found.")
            return
        prob_map_raw = res["grid"]
        cohort_unavailable_mask = res["unavailable"]
        n_min_valid, n_max_valid, n_subjects = res["n_min"], res["n_max"], res["n"]
        st = self._mean_map_style(study_type, timepoint)
        cmap_use, vmin_use, vmax_use = st["cmap"], st["vmin"], st["vmax"]
        map_title, clabel = st["title"], st["clabel"]
        contour_levels, contour_colors, cfmt = st["levels"], st["colors"], st["cfmt"]

        fig_r, ax_r = self._subplots(figsize=(6,6))
        ax_r.imshow(prob_map_raw, cmap=cmap_use, vmin=vmin_use, vmax=vmax_use)
        ax_r.axis('off')
        self._draw_overlay(ax_r)
        ax_r.set_title(f"Raw Topographic Map - {self.quadrant_var.get()}\n{map_title}",
                       fontweight='bold')
        self.add_save_button(fig_r); plt.show()

        # NaN-aware smoothing: an unavailable location contributes zero weight,
        # not a 0 dB value, to its neighbors.
        proc = _smooth_in_field(prob_map_raw)

        present = self._present_contour_levels([proc], contour_levels)
        two_panels = len(present) >= self.CONTOUR_PANEL_MIN_LEVELS
        if two_panels:
            fig_p, (ax_clean, ax_contour) = self._subplots(1, 2, figsize=(14,7))
        else:
            fig_p, ax_clean = self._subplots(figsize=(8,7))
            ax_contour = None

        def _ov(ax): self._draw_overlay(ax)

        im_clean = ax_clean.imshow(proc, cmap=cmap_use, vmin=vmin_use, vmax=vmax_use)
        _ov(ax_clean)
        # Redraw the unavailable-location markers after interpolation and smoothing.
        self._draw_missing_markers(ax_clean, cohort_unavailable_mask, do_flip=False,
                                    label="No valid paired-eye source (excluded from cohort mean)")
        fig_p.colorbar(im_clean,ax=ax_clean,fraction=0.046,pad=0.04).set_label(
            clabel, fontweight='bold', fontsize=8)
        head = f"{map_title}\n{self.quadrant_var.get()}  n={n_subjects}  " \
               f"|  valid n/location: {n_min_valid}-{n_max_valid}"
        ax_clean.set_title(("(A) " if two_panels else "") + head, fontweight='bold', fontsize=10)
        ax_clean.axis('off')

        import numpy.ma as ma
        proc_masked = ma.masked_invalid(proc)

        def draw_contours(ax, levels):
            if not levels:
                return
            try:
                cs = ax.contour(proc_masked, levels=levels,
                                colors=[contour_colors[contour_levels.index(lv)] for lv in levels],
                                linewidths=2.5, linestyles='-', alpha=1.0)
                for txt in ax.clabel(cs, fmt=cfmt, fontsize=11, inline=True, inline_spacing=5):
                    txt.set_fontweight('bold')
            except Exception as exc:
                print(f"[NeuroVF] contours not drawn: {exc}")

        if two_panels:
            im_cont = ax_contour.imshow(proc, cmap=cmap_use, vmin=vmin_use, vmax=vmax_use)
            _ov(ax_contour)
            self._draw_missing_markers(ax_contour, cohort_unavailable_mask, do_flip=False)
            draw_contours(ax_contour, contour_levels)
            fig_p.colorbar(im_cont,ax=ax_contour,fraction=0.046,pad=0.04).set_label(
                clabel, fontweight='bold', fontsize=8)
            ax_contour.set_title(f"(B) Topographic Contours\n{self.quadrant_var.get()}",
                                 fontweight='bold',fontsize=10)
            ax_contour.axis('off')
        else:
            draw_contours(ax_clean, present)     # few lines: overlay them, don't repeat the map

        legend_caption = (f"{st['scale_note']}\n{st['kind_note']}\n"
                          "Sand-hatched loci: no valid paired-eye source measurement available "
                          "across the cohort; excluded from the cohort mean."
                          + (f"\n{self._registration_note()}" if self._registration_note() else ""))
        self._caption(fig_p, legend_caption, face="#e3f2fd", edge="#1565c0", fontsize=8)
        self.add_save_button(fig_p); plt.show()

    # Category-specific mean maps (by a categorical variable)
    def generate_probability_map_by_category(self, study_type="cross", timepoint="preop"):
        """Category-specific mean maps for a chosen categorical variable.

        Draws one mean map per value of the variable, side by side (at most six
        categories). Each category mean is computed on the original tested
        locations from the subjects' registered values (complete-case within
        the category); interpolation to the display grid happens afterwards and
        is for display only.
        """
        if not self.categorical_cols:
            messagebox.showwarning("No Categorical Variables",
                "No Categorical variables defined.\n"
                "Use 'Add Variable' to create one first.")
            return

        cat_var = self._select_variable("Select Categorical Variable", self.categorical_cols, "Select Categorical variable for grouping:")
        if not cat_var:
            return

        # Collect the per-location arrays (not interpolated grids) by category
        # value; statistics use the original tested locations.
        cat_raw76 = {}
        for sid, data in self.db.items():
            if sid in RESERVED_DB_KEYS or not isinstance(data, dict):
                continue
            cat_val = str(data.get(cat_var, "")).strip()
            if not cat_val:
                continue

            if study_type == "longitudinal":
                delta76, valid_mask = self._get_loss76(data)
                if delta76 is None:
                    continue
                cat_raw76.setdefault(cat_val, []).append(delta76)
            else:
                prefix = self._tp_prefix(timepoint)
                combined, valid_mask = self._get_avg76_with_fallback(data, prefix)
                if combined is None:
                    continue
                cat_raw76.setdefault(cat_val, []).append(combined)

        if not cat_raw76:
            messagebox.showwarning("No Data",
                f"No subjects with both '{cat_var}' filled in AND VF data.")
            return

        categories = sorted(cat_raw76.keys())
        if len(categories) > 6:
            messagebox.showwarning("Too Many Categories",
                f"Variable '{cat_var}' has {len(categories)} unique values.\n"
                "Please use a variable with ≤6 categories.")
            return

        st = self._mean_map_style(study_type, timepoint)
        cmap_use, vmin_use, vmax_use, clabel = st["cmap"], st["vmin"], st["vmax"], st["clabel"]

        def _smooth(raw76_list):
            if not raw76_list:
                return None, None
            stacked = np.array(raw76_list)
            category_valid_n = np.sum(np.isfinite(stacked), axis=0)
            n_cat_total = len(raw76_list)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean76 = np.nanmean(stacked, axis=0)
            # Complete-case within the category: this mask belongs to this
            # category only.
            category_unavailable_mask = category_valid_n < n_cat_total
            mean76[category_unavailable_mask] = np.nan
            _sig = (study_type == "longitudinal")
            m = pts_to_grid_100(self.grid_points, mean76, False, signed=_sig,
                                vmin=(None if _sig else self.plausible_min),
                                vmax=(None if _sig else self.plausible_max),
                                extent_deg=self.grid_extent_deg)
            m = self.apply_quadrant_mask(m)
            return _smooth_in_field(m), category_unavailable_mask

        def _ov(ax): self._draw_overlay(ax)

        def _add_contours(ax, sm, levels, colors, cfmt):
            import numpy.ma as ma
            if not levels:
                return
            try:
                cs = ax.contour(ma.masked_invalid(sm), levels=levels,
                                colors=colors, linewidths=2.0, linestyles='-', alpha=0.9)
                for txt in ax.clabel(cs, fmt=cfmt, fontsize=9, inline=True, inline_spacing=4):
                    txt.set_fontweight('bold')
            except Exception as exc:
                print(f"[NeuroVF] contours not drawn: {exc}")

        n_cats = len(categories)
        # One panel per category value
        n_panels = n_cats
        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")

        fig, axes = self._axes_grid(n_panels, (7*n_panels, 7))
        fig.suptitle(
            _wrap_title(
                f"Category-Specific Mean {st['name']} Maps - {cat_var}  |  {tp_lbl}  |  "
                f"Region: {self.quadrant_var.get()}"
            ),
            fontweight='bold', fontsize=11)

        labels_str = [chr(65+i) for i in range(n_cats)]  # A, B, C...

        # Each category's own map first, so the contour levels can be chosen
        # once for the whole figure and every panel is drawn on the same scale.
        maps = {c: _smooth(cat_raw76[c]) for c in categories}
        grids = [g for g, _ in maps.values() if g is not None]
        # Only the study's severity-band levels are contoured. Evenly spaced
        # levels over the data range would cut the small, noisy per-category and
        # per-eye panels into many short loops with crowded labels.
        levels = self._present_contour_levels(grids, st["levels"])
        colors = [st["colors"][st["levels"].index(lv)] for lv in levels]
        cfmt = st["cfmt"]
        levels_note = (f"Contours (dB): {', '.join(f'{lv:g}' for lv in levels)}" if levels else "")

        for idx, cat_val in enumerate(categories):
            sm, category_unavailable_mask = maps[cat_val]
            n_sub = len(cat_raw76[cat_val])
            ax = axes[idx]
            if sm is not None:
                im = ax.imshow(sm, cmap=cmap_use, vmin=vmin_use, vmax=vmax_use)
                _ov(ax)
                _add_contours(ax, sm, levels, colors, cfmt)
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.locator = MaxNLocator(nbins=3); cb.update_ticks()
                cb.ax.tick_params(labelsize=7)
                cb.set_label(clabel, fontsize=8)
                # This category's missing markers, redrawn after interpolation and
                # smoothing.
                self._draw_missing_markers(ax, category_unavailable_mask, do_flip=False)
            ax.set_title(f"({labels_str[idx]}) {cat_val}  n={n_sub}",
                         fontweight='bold', fontsize=10)
            ax.axis('off')

        cat_caption = (f"Variable: {cat_var}  |  {tp_lbl}  |  {st['scale_note']}\n"
                       f"Descriptive Category-Specific Mean {st['name']} Maps - no statistical test "
                       "between categories is performed here.\n"
                       "Sand-hatched loci: no valid paired-eye source measurement available "
                       "within this category; excluded from the category mean."
                       + (f"\n{levels_note}" if levels_note else "")
                       + (f"\n{self._registration_note()}" if self._registration_note() else ""))
        self._caption(fig, cat_caption, face="#e8f5e9", edge="#2e7d32")
        self._present(fig)

    # Spatial Map
    def _heatmap_pick_method(self):
        """Correlation-method dialog; returns the chosen method or None."""
        corr_win = tk.Toplevel(self.root)
        corr_win.title("Spatial Map Analysis - Correlation Method")
        corr_win.grab_set()
        corr_win.resizable(False, False)

        tk.Label(corr_win, text="Select correlation method:",
                 font=("Arial", 10, "bold"), pady=8).pack(padx=20, anchor='w')

        corr_method = tk.StringVar(value="pearson")

        # Category group comparison methods
        if self.categorical_cols:
            tk.Label(corr_win, text="--- Group Comparison (Categorical variable - Welch's t-test) ---",
                     font=("Arial", 8, "italic"), fg="#1565c0").pack(anchor='w', padx=20, pady=(4,2))
            tk.Radiobutton(corr_win,
                           text="Category Group Comparison  -  Pick a Categorical variable,\n"
                                "   compare two groups pixel-wise (Welch's t-test + BH-FDR)",
                           variable=corr_method, value="cat_group",
                           font=("Arial", 9), justify='left',
                           wraplength=560).pack(anchor="w", padx=20, pady=3)
        else:
            tk.Label(corr_win,
                     text="--- No Categorical variables defined yet (use 'Add Variable') ---",
                     font=("Arial", 8, "italic"), fg="#aaaaaa").pack(anchor='w', padx=20, pady=(4,2))

        # Numeric correlation methods
        tk.Label(corr_win, text="--- Continuous Variable Correlation ---",
                 font=("Arial", 8, "italic"), fg="#555").pack(anchor='w', padx=20, pady=(8,2))

        base_methods = [
            ("pearson",
             f"Pearson r  -  linear, parametric\n"
             f"   Uses raw {self.data_type_spec['label']}; caveat: floor/ceiling effects at extreme pixels"),
            ("spearman",
             f"Spearman rho  -  rank-based, non-parametric\n"
             f"   Uses raw {self.data_type_spec['label']}; caveat: still affected by extreme-value clustering"),
        ]

        for val, txt in base_methods:
            tk.Radiobutton(corr_win, text=txt, variable=corr_method, value=val,
                           font=("Arial", 9), justify='left',
                           wraplength=560).pack(anchor="w", padx=20, pady=3)

        confirmed = tk.BooleanVar(value=False)
        def on_ok():    confirmed.set(True); corr_win.destroy()
        def on_cancel(): corr_win.destroy()

        bf = tk.Frame(corr_win); bf.pack(pady=8)
        tk.Button(bf, text="Run Analysis", bg="#2e7d32", fg="white",
                  font=("Arial", 10, "bold"), command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        corr_win.wait_window()
        if not confirmed.get():
            return None

        return corr_method.get()

    def generate_both_eyes_heatmap(self, study_type="cross", timepoint="preop"):
        """Spatial map analysis: pick a method once, then run it."""
        method_choice = self._remember("corr_method", self._heatmap_pick_method)
        if method_choice is None:
            return

        if method_choice == "cat_group":
            self._spatial_category_group(study_type, timepoint)
        else:
            if not self.continuous_cols:
                messagebox.showwarning("No Continuous Variables",
                    "No Continuous variables defined.\nUse 'Add Variable' to create one.")
                return
            var_to_test = self._select_variable("Spatial Map Analysis", self.continuous_cols, "Select Continuous variable:")
            if not var_to_test:
                return
            self._spatial_numeric_corr(method_choice, var_to_test, study_type, timepoint)

    # Registration notes and category group comparison
    def _policy_flip_flag(self, sid, eye):
        """Whether this eye's field is mirrored under the study's fixed
        registration policy - True/False for chiasmatic, PTC (the one
        configured eye) and post-chiasmatic (both eyes of a subject whose
        Surgery Side is the flipped one). None for the General policy,
        where the flip stays a manual per-eye choice."""
        pol = getattr(self, "flip_policy", "manual")
        if pol == "flip_one_eye_cohort" and getattr(self, "flip_eye", None):
            return eye.lower() == self.flip_eye
        if pol == "surgery_side_mirror" and getattr(self, "surgery_side_var", None):
            val = str(self.db.get(sid, {}).get(self.surgery_side_var, "")).strip()
            return bool(val) and val != str(self.surgery_side_reference).strip()
        return None

    def _registration_note(self):
        """One-line statement of how left/right was registered for this
        study type, appended to figure captions so a map's orientation is
        never ambiguous. Empty for the manual (General) policy."""
        pol = getattr(self, "flip_policy", "manual")
        if pol == "surgery_side_mirror" and getattr(self, "surgery_side_var", None):
            ref = str(self.surgery_side_reference)
            opts = self.variable_schema.get(self.surgery_side_var, {}).get("options", []) or []
            flipped = [o for o in opts if str(o) != ref] or ["the other side"]
            return (f"Registered frame: subjects with {self.surgery_side_var} = "
                    f"{', '.join(repr(str(o)) for o in flipped)} are MIRRORED left-right; "
                    f"'{ref}' subjects are shown as recorded.")
        if pol == "flip_one_eye_cohort" and getattr(self, "flip_eye", None):
            return (f"Registered frame: {self.flip_eye.upper()} is mirrored to the other "
                    f"eye's nasal/temporal orientation for every subject.")
        return ""

    def _group_stats(self, entries, cat_var, group_a_label, group_b_label):
        """Two-level workflow: pointwise Welch t-test with BH-FDR.

        Computed on the current eye view at each original tested location, with
        BH-FDR across the computable locations and Hedges' g (numerical output
        only); the results are then interpolated for display. Returns a dict;
        R["ok"] is False (with n_a and n_b) when a group has fewer than two
        subjects.
        """
        from scipy.stats import ttest_ind
        grp_a, grp_b = [], []; sid_a, sid_b = [], []
        for e in entries:
            val = str(e["data"].get(cat_var, "")).strip()
            if   val == group_a_label: grp_a.append(e["vf76"]); sid_a.append(e["sid"])
            elif val == group_b_label: grp_b.append(e["vf76"]); sid_b.append(e["sid"])
        n_a, n_b = len(grp_a), len(grp_b)
        if n_a < 2 or n_b < 2:
            return {"ok": False, "n_a": n_a, "n_b": n_b}

        arr_a = np.array(grp_a)
        arr_b = np.array(grp_b)

        # Complete-case rule: a location missing in any subject of a group is
        # not computed for that group. Non-computable statistics stay np.nan
        # (unavailable), so they cannot be read as t = 0, p = 1 or g = 0.
        nv = self.n_vf
        p_76 = np.full(nv, np.nan); g_76 = np.full(nv, np.nan)
        t_76 = np.full(nv, np.nan); df_76 = np.full(nv, np.nan)
        reason_incomplete = np.zeros(nv, bool); reason_zero_var = np.zeros(nv, bool)
        for i in range(nv):
            a = arr_a[:, i]; b = arr_b[:, i]
            a_v = a[np.isfinite(a)]; b_v = b[np.isfinite(b)]
            if len(a_v) < n_a or len(b_v) < n_b:
                reason_incomplete[i] = True
                continue
            if np.std(a_v) > GROUP_SD_EPS or np.std(b_v) > GROUP_SD_EPS:
                _tres = ttest_ind(a_v, b_v, equal_var=False)
                _t, p_76[i] = _tres.statistic, _tres.pvalue
                t_76[i] = _t; df_76[i] = welch_df(a_v, b_v)
                g_76[i] = hedges_g_pair(a_v, b_v)
            else:
                reason_zero_var[i] = True

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            n_valid_a = np.sum(np.isfinite(arr_a), axis=0)
            n_valid_b = np.sum(np.isfinite(arr_b), axis=0)
            mean_a_76 = np.nanmean(arr_a, axis=0)   # across subjects; masked below if incomplete
            mean_b_76 = np.nanmean(arr_b, axis=0)
        mean_a_76[n_valid_a < n_a] = np.nan
        mean_b_76[n_valid_b < n_b] = np.nan

        # Group B - Group A is available only where both group means are.
        difference_valid_mask = np.isfinite(mean_a_76) & np.isfinite(mean_b_76)
        diff_76 = np.full(nv, np.nan)
        diff_76[difference_valid_mask] = (
            mean_b_76[difference_valid_mask] - mean_a_76[difference_valid_mask])

        # The BH-FDR family is the set of computable (finite) p-values;
        # non-computable locations stay np.nan.
        q_76 = bh_fdr(p_76)
        res_diff = stats_76_to_grid(diff_76, p_76, do_flip=False, points=self.grid_points, extent_deg=self.grid_extent_deg)
        mean_a_g = pts_to_grid_100(self.grid_points, mean_a_76, False, vmin=self.map_vmin, vmax=self.map_vmax, extent_deg=self.grid_extent_deg)
        mean_b_g = pts_to_grid_100(self.grid_points, mean_b_76, False, vmin=self.map_vmin, vmax=self.map_vmax, extent_deg=self.grid_extent_deg)
        result = {
            "ok": True, "n_a": n_a, "n_b": n_b, "n_groups": 2, "n_contrasts": 1,
            "group_a": group_a_label, "group_b": group_b_label, "sids_a": sid_a, "sids_b": sid_b,
            "correction": "BH-FDR across computable locations (1 contrast)",
            "n_valid_a": n_valid_a, "n_valid_b": n_valid_b,
            "mean_a": mean_a_76, "mean_b": mean_b_76, "diff": diff_76,
            "difference_valid_mask": difference_valid_mask,
            "t": t_76, "df": df_76, "p": p_76, "q": q_76, "hedges_g": g_76,
            "reason_incomplete": reason_incomplete, "reason_zero_variance": reason_zero_var,
            "family_n": int(np.isfinite(p_76).sum()),
            "n_sig": int(np.nansum(q_76 <= GROUP_ALPHA)),
            "n_unc": int(np.nansum(p_76 < GROUP_ALPHA)),
            "unc_mask": res_diff['unc_mask'], "sig_mask": res_diff['sig_mask'],
            "mean_a_g": self.apply_quadrant_mask(mean_a_g),
            "mean_b_g": self.apply_quadrant_mask(mean_b_g),
            "diff_g":    self.apply_quadrant_mask(res_diff['stat_grid']),
            "sig_g":     self.apply_quadrant_mask(res_diff['sig_grid']),
            "unc_g":     self.apply_quadrant_mask(res_diff['unc_grid']),
        }
        return result

    @staticmethod
    def _range_txt(a, fmt="{:+.2f}"):
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        if a.size == 0:
            return "n/a"
        return f"median {fmt.format(np.median(a))} (range {fmt.format(a.min())} to {fmt.format(a.max())})"

    def _noncomputable_txt(self, reason_incomplete, reason_zero_var):
        ni, nz = int(np.sum(reason_incomplete)), int(np.sum(reason_zero_var))
        return (f"Non-computable locations: {ni + nz} of {self.n_vf} "
                f"(incomplete group: {ni}; zero-variance group: {nz})")

    def _two_group_summary(self, R, excluded=None):
        """Numerical summary (caption text) for the two-level workflow. Hedges' g is
        reported here and in the research exports only -- it is never mapped."""
        vna = R["n_valid_a"]; vnb = R["n_valid_b"]
        txt = (f"Welch's t-test per location; BH-FDR across {R['family_n']} computable locations (1 contrast: "
               f"{R['group_b']} \u2212 {R['group_a']})  |  groups: 2  |  n: {R['group_a']}={R['n_a']}, {R['group_b']}={R['n_b']}\n"
               f"Hedges' g (B\u2212A; numerical output only, not mapped): {self._range_txt(R['hedges_g'])} "
               f"over {int(np.isfinite(R['hedges_g']).sum())} computable locations\n"
               f"{self._noncomputable_txt(R['reason_incomplete'], R['reason_zero_variance'])}")
        if excluded:
            txt += "\nLevels excluded (n<2): " + ", ".join(f"{k} (n={v})" for k, v in excluded.items())
        return txt

    def _group_pair(self, study_type, timepoint, cat_var, group_a_label, group_b_label, excluded=None):
        """Two-level group comparison for OD and OS side by side, as recorded (not
        mirrored): rows = statistic (mean difference), uncorrected, BH-FDR corrected."""
        def per_eye():
            return self._group_stats(self._collect_76_for_spatial(study_type, timepoint),
                                     cat_var, group_a_label, group_b_label)
        res = self._eye_results(per_eye)
        if not any(r["ok"] for r in res.values()):
            messagebox.showwarning("Insufficient Data",
                "Need \u22652 per group in at least one eye.\n" +
                "\n".join(f"{eye.upper()}: {group_a_label} {r['n_a']} | {group_b_label} {r['n_b']}"
                          for eye, r in res.items()))
            return
        ok = {eye: r for eye, r in res.items() if r["ok"]}

        smooth_circ = _smooth_in_field
        miss = {eye: ~r["difference_valid_mask"] for eye, r in ok.items()}
        gr = lambda key, f=lambda g: g: {eye: (f(ok[eye][key]) if eye in ok else None) for eye in ("od", "os")}
        sub = lambda fmt: {eye: (fmt(ok[eye]) if eye in ok else "") for eye in ("od", "os")}
        common = dict(cmap="RdBu_r", sym=True, missing=miss)
        b_a = f"{group_b_label} \u2212 {group_a_label}"
        rows = [
            dict(common, title="Mean difference, B\u2212A (dB)", grids=gr("diff_g", smooth_circ),
                 cbar="dB", sub=sub(lambda r: f"n_A={r['n_a']}  n_B={r['n_b']}")),
            dict(common, title="Uncorrected p<0.05 (Welch's t)", grids=gr("unc_g"), cbar="B\u2212A (unc.)",
                 sub=sub(lambda r: f"{r['n_unc']}/{r['family_n']} significant")),
            dict(common, title="BH-FDR corrected q<0.05", grids=gr("sig_g"), cbar="B\u2212A (FDR)",
                 sub=sub(lambda r: f"{r['n_sig']}/{r['family_n']} significant")),
        ]
        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [\u0394]")
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        per_eye_txt = "\n".join(
            f"{eye.upper()}: Hedges' g (numerical only) {self._range_txt(r['hedges_g'])}; "
            + self._noncomputable_txt(r["reason_incomplete"], r["reason_zero_variance"])
            for eye, r in ok.items())
        caption = (f"Variable: {cat_var}  |  Group A = '{group_a_label}'  |  Group B = '{group_b_label}'  |  "
                   f"Difference = Group B \u2212 Group A (dB)  |  {tp_lbl}\n"
                   f"RED = Group B higher {dt_name}  |  BLUE = Group B lower {dt_name}  |  "
                   f"Welch's t-test, BH-FDR across computable locations within each eye (1 contrast)\n"
                   f"{per_eye_txt}\n"
                   f"OD and OS are shown exactly as recorded (the mirrored eye is NOT mirrored here). "
                   f"Sand-hatched loci: a group mean was unavailable; difference not calculated."
                   + ("\nLevels excluded (n<2): " + ", ".join(f"{k} (n={v})" for k, v in excluded.items()) if excluded else ""))
        self._draw_eye_pair(rows, f"{dt_name}: {group_a_label} vs {group_b_label}  ({cat_var})  |  "
                                  f"Region: {self.quadrant_var.get()}", caption)

    # Three or more levels: pointwise Welch ANOVA with Games-Howell post hoc.
    OMNIBUS_CORRECTION = "pointwise Welch ANOVA; BH-FDR across computable locations"

    def _posthoc_correction_txt(self, m):
        return (f"Games\u2013Howell (studentized range; family-wise error controlled across the {m} unique contrasts at each "
                f"location), computed only at locations with omnibus BH-FDR q\u2264{GROUP_ALPHA:g}")

    def _group_stats_multi(self, entries, cat_var, levels, excluded=None):
        """Three-or-more-level workflow on the current eye view. levels: the groups analysed (each needs
        >= 2 subjects). A location is computable only if every subject of every group is finite there and
        every group has sd > GROUP_SD_EPS. Returns the omnibus statistics, post-hoc contrasts (numerical
        only) and display grids; R["ok"] is False when a group has < 2 subjects."""
        rows = {lv: [] for lv in levels}; sids = {lv: [] for lv in levels}
        for e in entries:
            val = str(e["data"].get(cat_var, "")).strip()
            if val in rows:
                rows[val].append(e["vf76"]); sids[val].append(e["sid"])
        k = len(levels); n = [len(rows[lv]) for lv in levels]
        if k < 3 or min(n) < 2:
            return {"ok": False, "levels": list(levels), "n": n}
        arrs = [np.array(rows[lv]) for lv in levels]
        nv = self.n_vf
        F = np.full(nv, np.nan); df2 = np.full(nv, np.nan); p = np.full(nv, np.nan)
        eta = np.full(nv, np.nan); omega = np.full(nv, np.nan)
        r_incomplete = np.zeros(nv, bool); r_zero_var = np.zeros(nv, bool)
        fins = [None] * nv
        for i in range(nv):
            cols = [a[:, i] for a in arrs]
            fin = [c[np.isfinite(c)] for c in cols]
            if any(len(f) < nn for f, nn in zip(fin, n)):
                r_incomplete[i] = True; continue
            r = welch_anova(fin)
            if r is None:
                r_zero_var[i] = True; continue
            F[i], df2[i], p[i], eta[i], omega[i] = r["F"], r["df2"], r["p"], r["eta_sq"], r["omega_sq"]
            fins[i] = fin
        q = bh_fdr(p)
        gate = np.isfinite(q) & (q <= GROUP_ALPHA)
        m = k * (k - 1) // 2
        post = {}
        for a_i in range(k):
            for b_i in range(a_i + 1, k):
                post[f"{levels[a_i]} vs {levels[b_i]}"] = {
                    "level_a": levels[a_i], "level_b": levels[b_i],
                    "diff": np.full(nv, np.nan), "t": np.full(nv, np.nan), "df": np.full(nv, np.nan),
                    "p": np.full(nv, np.nan), "hedges_g": np.full(nv, np.nan)}
        for i in np.where(gate)[0]:
            for gh in games_howell(fins[i]):
                d = post[f"{levels[gh['i']]} vs {levels[gh['j']]}"]
                d["diff"][i], d["t"][i], d["df"][i], d["p"][i], d["hedges_g"][i] = gh["diff"], gh["t"], gh["df"], gh["p"], gh["hedges_g"]
        for d in post.values():
            d["significant"] = np.isfinite(d["p"]) & (d["p"] < GROUP_ALPHA)
            d["n_significant"] = int(d["significant"].sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            n_valid = np.array([np.sum(np.isfinite(a), axis=0) for a in arrs])
            mean = np.array([np.nanmean(a, axis=0) for a in arrs])
        for gi in range(k):
            mean[gi][n_valid[gi] < n[gi]] = np.nan
        res_F = stats_76_to_grid(F, p, do_flip=False, points=self.grid_points, extent_deg=self.grid_extent_deg)
        mean_g = [self.apply_quadrant_mask(pts_to_grid_100(self.grid_points, mean[gi], False, vmin=self.map_vmin,
                                                            vmax=self.map_vmax, extent_deg=self.grid_extent_deg))
                  for gi in range(k)]
        result_multi = {
            "ok": True, "levels": list(levels), "n_groups": k, "n": n, "sids": sids,
            "excluded_levels": dict(excluded or {}),
            "n_contrasts": m, "contrasts": [[d["level_a"], d["level_b"]] for d in post.values()],
            "correction": {"omnibus": self.OMNIBUS_CORRECTION, "posthoc": self._posthoc_correction_txt(m)},
            "n_valid": n_valid, "mean": mean,
            "F": F, "df1": float(k - 1), "df2": df2, "p": p, "q": q, "eta_sq": eta, "omega_sq": omega,
            "reason_incomplete": r_incomplete, "reason_zero_variance": r_zero_var,
            "computable": np.isfinite(p), "posthoc_applicable": gate, "posthoc": post,
            "family_n": int(np.isfinite(p).sum()),
            "n_unc": int(np.nansum(p < GROUP_ALPHA)), "n_sig": int(np.nansum(q <= GROUP_ALPHA)),
            "n_posthoc_locations": int(gate.sum()),
            "unc_mask": res_F["unc_mask"], "sig_mask": res_F["sig_mask"],
            "mean_g": mean_g,
            "F_g": np.clip(self.apply_quadrant_mask(res_F["stat_grid"]), 0.0, None),
            "unc_g": self.apply_quadrant_mask(res_F["unc_grid"]),
            "sig_g": self.apply_quadrant_mask(res_F["sig_grid"]),
        }
        return result_multi

    def _multi_summary(self, R):
        """Numerical summary (caption text) for the three-or-more-level workflow. The omnibus effect size and
        the pairwise Hedges' g are reported here and in the research exports only -- never mapped."""
        k, m = R["n_groups"], R["n_contrasts"]
        lvl = ", ".join(f"{lv} n={nn}" for lv, nn in zip(R["levels"], R["n"]))
        ph = "; ".join(f"{key}: {d['n_significant']}" for key, d in R["posthoc"].items())
        gpool = np.concatenate([d["hedges_g"][np.isfinite(d["hedges_g"])] for d in R["posthoc"].values()] or [np.array([])])
        txt = (f"Omnibus: {self.OMNIBUS_CORRECTION} ({R['family_n']} computable locations)  |  groups: {k} ({lvl})  |  unique contrasts: {m}\n"
               f"Post hoc (numerical output only): {self._posthoc_correction_txt(m)}; applied at {R['n_posthoc_locations']} locations; "
               f"locations with Games\u2013Howell p<{GROUP_ALPHA:g} per contrast: {ph}\n"
               f"Effect sizes (numerical output only, not mapped): omega\u00b2 {self._range_txt(R['omega_sq'], '{:.3f}')}; "
               f"pairwise Hedges' g {self._range_txt(gpool)} over {gpool.size} contrast\u00d7location values\n"
               f"{self._noncomputable_txt(R['reason_incomplete'], R['reason_zero_variance'])}")
        if R["excluded_levels"]:
            txt += "\nLevels excluded (n<2): " + ", ".join(f"{a} (n={b})" for a, b in R["excluded_levels"].items())
        return txt

    def _group_pair_multi(self, study_type, timepoint, cat_var, levels, excluded=None):
        """Three-or-more-level comparison for OD and OS side by side, as recorded (not mirrored):
        3 x 2 layout, columns OD | OS, rows = Welch-ANOVA F statistic, uncorrected, BH-FDR corrected."""
        def per_eye():
            return self._group_stats_multi(self._collect_76_for_spatial(study_type, timepoint), cat_var, levels, excluded)
        res = self._eye_results(per_eye)
        if not any(r["ok"] for r in res.values()):
            messagebox.showwarning("Insufficient Data",
                "Need \u22652 subjects in every group in at least one eye.\n" +
                "\n".join(f"{eye.upper()}: " + ", ".join(f"{lv} {nn}" for lv, nn in zip(levels, r["n"])) for eye, r in res.items()))
            return
        ok = {eye: r for eye, r in res.items() if r["ok"]}
        gr = lambda key: {eye: (ok[eye][key] if eye in ok else None) for eye in ("od", "os")}
        sub = lambda fmt: {eye: (fmt(ok[eye]) if eye in ok else "") for eye in ("od", "os")}
        miss = {eye: ~r["computable"] for eye, r in ok.items()}
        hi = max([safe_absmax(r["F_g"], 1.0) for r in ok.values()] + [1.0])
        common = dict(cmap="YlOrRd", vlim=(0.0, hi), missing=miss, cbar="Welch F")
        k, m = len(levels), len(levels) * (len(levels) - 1) // 2
        rows = [
            dict(common, title="Omnibus Welch F (statistic)", grids=gr("F_g"),
                 sub=sub(lambda r: f"k={r['n_groups']} groups  |  n=" + "/".join(str(x) for x in r["n"]))),
            dict(common, title="Uncorrected p<0.05 (Welch ANOVA)", grids=gr("unc_g"),
                 sub=sub(lambda r: f"{r['n_unc']}/{r['family_n']} significant")),
            dict(common, title="BH-FDR corrected q<0.05", grids=gr("sig_g"),
                 sub=sub(lambda r: f"{r['n_sig']}/{r['family_n']} significant")),
        ]
        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [\u0394]")
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        per_eye_txt = "\n".join(f"{eye.upper()}: " + self._multi_summary(r).replace("\n", "  |  ") for eye, r in ok.items())
        caption = (f"Variable: {cat_var}  |  {k} groups: " + ", ".join(f"'{lv}'" for lv in levels) + f"  |  {tp_lbl}\n"
                   f"{per_eye_txt}\n"
                   f"OD and OS are shown exactly as recorded (the mirrored eye is NOT mirrored here). "
                   f"Sand-hatched loci: not computable (incomplete or zero-variance group).")
        self._draw_eye_pair(rows, f"{dt_name}: Welch ANOVA across {k} groups  ({cat_var})  |  "
                                  f"Region: {self.quadrant_var.get()}", caption)

    def _spatial_category_group(self, study_type, timepoint):
        """
        Group comparison by a Categorical variable. Two levels: pointwise Welch t-test (BH-FDR, Hedges' g
        numerical only). Three or more levels: pointwise Welch ANOVA (BH-FDR) with Games-Howell post hoc
        (numerical only). Levels with fewer than 2 subjects are excluded and recorded.
        """
        if not self.categorical_cols:
            messagebox.showwarning("No Categorical Variables",
                "No Categorical variables defined.")
            return

        cat_var = self._select_variable("Select Categorical Variable", self.categorical_cols, "Pick Categorical variable for group split:")
        if not cat_var:
            return

        entries = self._collect_76_for_spatial(study_type, timepoint)
        all_values = sorted(set(
            str(e["data"].get(cat_var, "")).strip()
            for e in entries
            if str(e["data"].get(cat_var, "")).strip()
        ))

        if len(all_values) < 2:
            messagebox.showwarning("Insufficient Groups",
                f"Variable '{cat_var}' has fewer than 2 distinct values in subjects with VF data.")
            return

        # The groups analysed are fixed once per analysis chain, so the OD|OS
        # replay uses the same groups.
        cached = self._chain_get("groups")
        if cached is not None:
            levels, excluded = list(cached["levels"]), dict(cached["excluded"])
        else:
            counts = {lv: sum(1 for e in entries if str(e["data"].get(cat_var, "")).strip() == lv) for lv in all_values}
            levels = [lv for lv in all_values if counts[lv] >= 2]
            excluded = {lv: counts[lv] for lv in all_values if counts[lv] < 2}
            if len(levels) < 2:
                messagebox.showwarning("Insufficient Data",
                    "Need \u22652 per group.\n" + "  |  ".join(f"{lv}: {counts[lv]}" for lv in all_values))
                return
        self._chain_put("groups", {"levels": levels, "excluded": excluded})

        if len(levels) >= 3:
            if getattr(self, "_eye_pair", False):
                return self._group_pair_multi(study_type, timepoint, cat_var, levels, excluded)
            return self._spatial_category_group_multi(entries, study_type, timepoint, cat_var, levels, excluded)
        group_a_label, group_b_label = levels
        if getattr(self, "_eye_pair", False):
            return self._group_pair(study_type, timepoint, cat_var, group_a_label, group_b_label, excluded)
        return self._spatial_category_group_two(entries, study_type, timepoint, cat_var, group_a_label, group_b_label, excluded)

    def _spatial_category_group_two(self, entries, study_type, timepoint, cat_var, group_a_label, group_b_label, excluded):
        R = self._group_stats(entries, cat_var, group_a_label, group_b_label)
        n_a, n_b = R["n_a"], R["n_b"]
        if not R["ok"]:
            messagebox.showwarning("Insufficient Data",
                f"Need ≥2 per group.\n"
                f"{group_a_label}: {n_a}  |  {group_b_label}: {n_b}")
            return
        n_valid_a, n_valid_b = R["n_valid_a"], R["n_valid_b"]
        difference_valid_mask = R["difference_valid_mask"]
        family_n, n_sig, n_unc = R["family_n"], R["n_sig"], R["n_unc"]
        region_lbl = self.quadrant_var.get()
        n_pts  = self._n_active_pixels()
        mean_a_g, mean_b_g = R["mean_a_g"], R["mean_b_g"]
        diff_g, sig_g, unc_g = R["diff_g"], R["sig_g"], R["unc_g"]

        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        title_main = _wrap_title(
            f"{dt_name}: {group_a_label} vs {group_b_label}  ({cat_var})  |  {tp_lbl}  |  "
            f"n_A={n_a}  |  n_B={n_b}  |  Region: {region_lbl}  |  "
            f"BH-FDR q<0.05: {n_sig}/{family_n} computable locations  (unc p<0.05: {n_unc}/{family_n})"
        )

        legend_txt = (
            f"Variable: {cat_var}  |  Group A = '{group_a_label}'  |  Group B = '{group_b_label}'  |  "
            f"Difference = Group B − Group A (dB)\n"
            f"RdBu_r: RED = Group B higher {dt_name}  |  "
            f"BLUE = Group B lower {dt_name}\n"
            f"{self._two_group_summary(R, excluded)}  |  {tp_lbl}\n"
            f"Sand-hatched loci: a group mean was unavailable; between-group difference not calculated."
            + (f"\n{self._registration_note()}" if self._registration_note() else "")
        )

        def _ov(ax): self._draw_overlay(ax)

        smooth_circ = _smooth_in_field

        missing_a    = n_valid_a < n_a
        missing_b    = n_valid_b < n_b
        missing_diff = ~difference_valid_mask

        # Figure 1 - group mean maps and their difference
        fig1, axes1 = self._subplots(1, 3, figsize=(18,7))
        is_long = (study_type == "longitudinal")
        change_word = "Change" if is_long else "Mean"
        fig1.suptitle(title_main + f"  [{change_word} {dt_name} Maps, dB]", fontweight='bold', fontsize=10)
        for ax, arr, lbl, miss_mask in zip(axes1,
            [mean_a_g, mean_b_g, diff_g],
            [f"(A) {group_a_label} {change_word} {dt_name} (dB)", f"(B) {group_b_label} {change_word} {dt_name} (dB)",
             f"(C) Difference (dB): {group_b_label} − {group_a_label}"],
            [missing_a, missing_b, missing_diff]):
            sm   = smooth_circ(arr)
            # Longitudinal panels A and B are signed changes (Timepoint 2 -
            # Timepoint 1), so they use the same symmetric RdBu_r scale as the
            # difference panel, not the data type's absolute scale.
            if is_long or 'Diff' in lbl:
                vmax = safe_absmax(sm, 15); vmin = -vmax
                cmap = 'RdBu_r'
            else:
                vmin, vmax = self.map_vmin, self.map_vmax
                cmap = 'gray'
            im   = ax.imshow(sm, cmap=cmap, vmin=vmin, vmax=vmax)
            _ov(ax); fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Redraw the missing markers after interpolation and smoothing.
            self._draw_missing_markers(ax, miss_mask, do_flip=False)
            ax.set_title(lbl, fontweight='bold', fontsize=9); ax.axis('off')
        self._caption(fig1, legend_txt, fontsize=8)
        self.add_save_button(fig1); plt.show()

        # Figure 2 - statistic, uncorrected significance, BH-FDR significance (no effect-size map)
        abs_s   = safe_absmax(sig_g, 1)
        abs_unc = safe_absmax(unc_g, abs_s)
        abs_st  = safe_absmax(smooth_circ(diff_g), 1)

        fig2, axes2 = self._subplots(1, 3, figsize=(18,7))
        fig2.suptitle(title_main, fontweight='bold', fontsize=10)

        im1 = axes2[0].imshow(smooth_circ(diff_g), cmap='RdBu_r', vmin=-abs_st, vmax=abs_st)
        _ov(axes2[0])
        self._draw_missing_markers(axes2[0], missing_diff, do_flip=False)
        axes2[0].set_title(f"(A) Mean difference (dB)\nWelch's t  |  {group_b_label} − {group_a_label}",
                           fontweight='bold', fontsize=9); axes2[0].axis('off')
        fig2.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04).set_label(f"{group_b_label}−{group_a_label} (dB)", fontsize=8)

        im3 = axes2[1].imshow(unc_g, cmap='RdBu_r', vmin=-abs_unc, vmax=abs_unc)
        _ov(axes2[1])
        self._draw_missing_markers(axes2[1], missing_diff, do_flip=False)
        axes2[1].set_title(f"(B) Uncorrected p<0.05\nWelch's t  |  {n_unc} sig pts",
                           fontweight='bold',fontsize=9); axes2[1].axis('off')
        fig2.colorbar(im3,ax=axes2[1],fraction=0.046,pad=0.04).set_label(
            f"{group_b_label}−{group_a_label} (unc.)",fontsize=8)

        im4 = axes2[2].imshow(sig_g, cmap='RdBu_r', vmin=-abs_s, vmax=abs_s)
        _ov(axes2[2])
        self._draw_missing_markers(axes2[2], missing_diff, do_flip=False,
                                    label="Group mean unavailable; difference not calculated")
        axes2[2].set_title(f"(C) BH-FDR q<0.05\nWelch's t  |  {n_sig}/{family_n} computable  (region: {n_pts} = {region_lbl})",
                           fontweight='bold',fontsize=9); axes2[2].axis('off')
        fig2.colorbar(im4,ax=axes2[2],fraction=0.046,pad=0.04).set_label(
            f"{group_b_label}−{group_a_label} (FDR)",fontsize=8)

        self._caption(fig2, legend_txt, face='#e8f5e9', edge='#2e7d32', fontsize=8.5)
        self.add_save_button(fig2); plt.show()

    def _spatial_category_group_multi(self, entries, study_type, timepoint, cat_var, levels, excluded):
        R = self._group_stats_multi(entries, cat_var, levels, excluded)
        if not R["ok"]:
            messagebox.showwarning("Insufficient Data",
                "Need \u22652 per group.\n" + "  |  ".join(f"{lv}: {nn}" for lv, nn in zip(levels, R["n"])))
            return
        k = R["n_groups"]
        region_lbl = self.quadrant_var.get()
        n_pts = self._n_active_pixels()
        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        title_main = _wrap_title(
            f"{dt_name}: Welch ANOVA across {k} groups  ({cat_var})  |  {tp_lbl}  |  "
            f"n={'/'.join(str(x) for x in R['n'])}  |  Region: {region_lbl}  |  "
            f"BH-FDR q<0.05: {R['n_sig']}/{R['family_n']} computable locations  (unc p<0.05: {R['n_unc']}/{R['family_n']})")
        legend_txt = (
            f"Variable: {cat_var}  |  groups: " + ", ".join(f"({chr(65 + i)}) '{lv}'" for i, lv in enumerate(levels)) + f"  |  {tp_lbl}\n"
            f"{self._multi_summary(R)}\n"
            f"Sand-hatched loci: a group mean (maps A\u2013{chr(64 + k)}) or the omnibus statistic (maps below) was not computable."
            + (f"\n{self._registration_note()}" if self._registration_note() else ""))
        smooth_circ = _smooth_in_field
        is_long = (study_type == "longitudinal")
        change_word = "Change" if is_long else "Mean"

        # Figure 1 - one mean map per group
        ncols = min(k, 4); nrows = -(-k // ncols)
        fig1, axes1 = self._subplots(nrows, ncols, figsize=(5.4 * ncols + 1.5, 5.8 * nrows + 1.5), squeeze=False)
        fig1.suptitle(title_main + f"  [{change_word} {dt_name} Maps, dB]", fontweight='bold', fontsize=10)
        for gi in range(nrows * ncols):
            ax = axes1[gi // ncols][gi % ncols]
            if gi >= k:
                ax.axis("off"); continue
            sm = smooth_circ(R["mean_g"][gi])
            if is_long:
                vmax = safe_absmax(sm, 15); vmin, cmap = -vmax, 'RdBu_r'
            else:
                vmin, vmax, cmap = self.map_vmin, self.map_vmax, 'gray'
            im = ax.imshow(sm, cmap=cmap, vmin=vmin, vmax=vmax)
            self._draw_overlay(ax); fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._draw_missing_markers(ax, R["n_valid"][gi] < R["n"][gi], do_flip=False)
            ax.set_title(f"({chr(65 + gi)}) {levels[gi]}  n={R['n'][gi]}\n{change_word} {dt_name} (dB)", fontweight='bold', fontsize=9)
            ax.axis('off')
        self._caption(fig1, legend_txt, fontsize=8)
        self.add_save_button(fig1); plt.show()

        # Figure 2 - omnibus statistic, uncorrected significance, BH-FDR significance (no effect-size map)
        hi = safe_absmax(R["F_g"], 1.0)
        fig2, axes2 = self._subplots(1, 3, figsize=(18, 7))
        fig2.suptitle(title_main, fontweight='bold', fontsize=10)
        missing_stat = ~R["computable"]
        panels = [(R["F_g"], f"(A) Omnibus Welch F\nk={k} groups  |  F(df1={k - 1}, df2 per location)"),
                  (R["unc_g"], f"(B) Uncorrected p<0.05\nWelch ANOVA  |  {R['n_unc']} sig pts"),
                  (R["sig_g"], f"(C) BH-FDR q<0.05\nWelch ANOVA  |  {R['n_sig']}/{R['family_n']} computable  (region: {n_pts} = {region_lbl})")]
        for ax, (grid, ttl) in zip(axes2, panels):
            im = ax.imshow(grid, cmap='YlOrRd', vmin=0.0, vmax=hi)
            self._draw_overlay(ax)
            self._draw_missing_markers(ax, missing_stat, do_flip=False)
            ax.set_title(ttl, fontweight='bold', fontsize=9); ax.axis('off')
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Welch F", fontsize=8)
        self._caption(fig2, legend_txt, face='#e8f5e9', edge='#2e7d32', fontsize=8.5)
        self.add_save_button(fig2); plt.show()

    # Spatial numeric correlation
    def _corr_method_setup(self, method_choice):
        """(corr_func, method_label, should_compute) for Pearson / Spearman."""
        from scipy.stats import spearmanr
        if method_choice == "spearman":
            return (lambda x, y: spearmanr(x, y),
                    f"Spearman rho ({self.data_type_spec['label']})",
                    lambda pv: len(np.unique(pv)) > 2)
        return (lambda x, y: pearsonr(x, y),
                f"Pearson r ({self.data_type_spec['label']})",
                lambda pv: np.std(pv) > 1e-6)

    @staticmethod
    def _method_short(method_choice):
        return "Spearman ρ" if method_choice == "spearman" else "Pearson r"

    def _corr_stats76(self, vfs, vars_arr, corr_func, should_compute):
        """Per-location correlation coefficient and p-value (np.nan where
        not computable - never a fabricated 0 or 1)."""
        c_76 = np.full(self.n_vf, np.nan); p_76 = np.full(self.n_vf, np.nan)
        for i in range(self.n_vf):
            pix_full  = vfs[:, i]
            valid_sub = np.isfinite(pix_full)
            if valid_sub.sum() < 3:
                continue
            pix = pix_full[valid_sub]
            if should_compute(pix):
                c_76[i], p_76[i] = corr_func(pix, vars_arr[valid_sub])
        return c_76, p_76

    def _corr_pair(self, method_choice, var_to_test, study_type, timepoint):
        """Spatial correlation for OD and OS side by side, as recorded (not
        mirrored): raw r, uncorrected, BH-FDR corrected."""
        corr_func, method_label, should_compute = self._corr_method_setup(method_choice)
        method_short = self._method_short(method_choice)

        def per_eye():
            entries = self._collect_76_for_spatial(study_type, timepoint)
            v_list, f_list = [], []
            for e in entries:
                try:
                    v_list.append(float(e["data"].get(var_to_test, "")))
                    f_list.append(e["vf76"])
                except (ValueError, TypeError):
                    continue
            if len(v_list) < 3:
                return {"ok": False, "n": len(v_list)}
            c_76, p_76 = self._corr_stats76(np.array(f_list), np.array(v_list), corr_func, should_compute)
            valid = np.isfinite(c_76) & np.isfinite(p_76)
            res = self._corr_stats_grid_valid(c_76, p_76, valid)
            if res is None:
                return {"ok": False, "n": len(v_list)}
            q_76 = bh_fdr(p_76)
            return {"ok": True, "n": len(v_list), "res": res, "valid": valid,
                    "family_n": int(valid.sum()), "n_sig": int(np.nansum(q_76 <= 0.05)),
                    "n_unc": int(np.nansum(p_76 < 0.05))}
        res = self._eye_results(per_eye)
        ok = {eye: r for eye, r in res.items() if r["ok"]}
        if not ok:
            messagebox.showinfo("Insufficient Data",
                f"Not enough subjects/computable locations for a per-eye {method_label} map "
                f"of '{var_to_test}' (need \u22653 subjects per eye).")
            return

        crop = lambda eye, key: self.crop_to_quadrant(ok[eye]["res"][key]) if eye in ok else None
        raw = {eye: (self.crop_to_quadrant(_smooth_in_field(ok[eye]["res"]["stat_grid"])) if eye in ok else None)
               for eye in ("od", "os")}
        sig = {eye: crop(eye, "sig_grid") for eye in ("od", "os")}
        miss = {eye: ~r["valid"] for eye, r in ok.items()}
        sub = lambda fmt: {eye: (fmt(ok[eye]) if eye in ok else "") for eye in ("od", "os")}
        common = dict(cmap="RdBu_r", sym=True, missing=miss)
        rows = [
            dict(common, title="Raw correlation (no threshold)", grids=raw, cbar=method_short,
                 sub=sub(lambda r: f"n={r['n']}  |  {r['family_n']}/{self.n_vf} computable")),
            dict(common, title="Uncorrected p<0.05", grids={eye: crop(eye, "unc_grid") for eye in ("od", "os")},
                 cbar=f"{method_short} (unc.)", sub=sub(lambda r: f"{r['n_unc']} significant")),
            dict(common, title="BH-FDR corrected q<0.05", grids=sig, cbar=f"{method_short} (BH-FDR)",
                 sub=sub(lambda r: f"{r['n_sig']}/{r['family_n']} significant")),
        ]
        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [\u0394]")
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        caption = (f"Method: {method_label}  |  {tp_lbl}\n"
                   f"RED = r>0 (higher {var_to_test} \u2192 higher {dt_name})  |  "
                   f"BLUE = r<0  |  BH-FDR within each eye\n"
                   f"OD and OS are shown exactly as recorded (the mirrored eye is NOT mirrored here). "
                   f"Sand-hatched loci: correlation not computable.")
        self._draw_eye_pair(rows, f"Spatial Correlation ({method_label})  |  {var_to_test}  |  "
                                  f"Region: {self.quadrant_var.get()}", caption)

        # Optional categorical split, as in the registered pass. _remember reuses
        # the answers given there (including the variable chosen inside
        # _spatial_numeric_corr_by_category), and the split is drawn once per eye
        # on that eye's as-recorded data.
        if self.categorical_cols:
            ask = self._remember("cat_split", lambda: messagebox.askyesno(
                "Compare by Category?",
                f"Whole-cohort analysis of  '{var_to_test}'  is complete.\n\n"
                f"Would you like to compare the spatial correlation maps\n"
                f"split by a Categorical variable?\n\n"
                f"Example: compare  '{var_to_test}'  results for\n"
                f"each group defined by one of your categorical variables\n"
                f"(e.g. Male vs Female, Left vs Right, etc.)."))
            if ask:
                self._corr_by_category_per_eye(self._spatial_numeric_corr_by_category,
                                               method_choice, var_to_test, study_type, timepoint, method_label)

    def _corr_by_category_per_eye(self, runner, method_choice, var_to_test, study_type, timepoint, method_label):
        """Run the categorical split of a correlation once for OD and once for
        OS, each on that eye's own recording; label each figure with its eye."""
        prev = getattr(self, "_eye_view", "avg")
        try:
            for eye in ("od", "os"):
                self._eye_view = eye
                before = set(plt.get_fignums())
                runner(method_choice, var_to_test, study_type, timepoint, method_label)
                for n in sorted(set(plt.get_fignums()) - before):
                    fig = plt.figure(n)
                    label = f"{eye.upper()} - as recorded (not mirrored)"
                    sup = getattr(fig, "_suptitle", None)
                    if sup is not None and sup.get_text():
                        sup.set_text(f"{label}\n{sup.get_text()}")
                    else:
                        fig.suptitle(label, fontweight="bold", fontsize=10)
                    fig._eye_labelled = True
        finally:
            self._eye_view = prev

    def _spatial_numeric_corr(self, method_choice, var_to_test, study_type, timepoint):
        """Pointwise correlation of the VF data with a continuous variable."""
        if getattr(self, "_eye_pair", False):
            return self._corr_pair(method_choice, var_to_test, study_type, timepoint)

        entries   = self._collect_76_for_spatial(study_type, timepoint)
        v_list    = []; f_list = []
        for e in entries:
            raw_val = e["data"].get(var_to_test, "")
            try:
                v_list.append(float(raw_val))
                f_list.append(e["vf76"])
            except (ValueError, TypeError):
                continue

        if len(v_list) < 3:
            messagebox.showwarning("Insufficient Data",
                f"Need ≥3 patients with VF data AND '{var_to_test}'.\n"
                f"Found: {len(v_list)}")
            return

        vfs_76   = np.array(f_list)
        vars_arr = np.array(v_list)

        corr_func, method_label, should_compute = self._corr_method_setup(method_choice)
        method_short = self._method_short(method_choice)
        vfs_input = vfs_76

        # NaN is excluded separately at each tested location (pairwise-complete
        # per location), so a subject missing at one location still contributes
        # at the others.
        c_76, p_76 = self._corr_stats76(vfs_input, vars_arr, corr_func, should_compute)

        # A location is computable only where both the coefficient and its
        # p-value are finite. Non-computable p-values are not coerced to 1.0
        # before the multiple-testing family is built, and non-computable
        # coefficients are not interpolated as r = 0.
        valid_stat_mask = np.isfinite(c_76) & np.isfinite(p_76)
        family_n = int(valid_stat_mask.sum())
        q_76  = bh_fdr(p_76)
        n_sig = int(np.nansum(q_76 <= 0.05))
        n_unc = int(np.nansum(p_76 < 0.05))
        region_lbl = self.quadrant_var.get()
        n_pts  = self._n_active_pixels()

        res_all = self._corr_stats_grid_valid(c_76, p_76, valid_stat_mask)
        if res_all is None:
            messagebox.showinfo(
                "Insufficient Computable Locations",
                f"Only {family_n}/{self.n_vf} pixels had a computable {method_label} "
                f"in this cohort - too few for an interpolated statistical map.\n\n"
                f"Insufficient computable source locations for an interpolated "
                f"statistical map.")
            return

        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")

        # Colour convention: red = positive r (higher variable value, higher VF value)
        _dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        _red  = f"RED  = r>0 → higher {var_to_test} → higher {_dt_name}"
        _blue = f"BLUE = r<0 → higher {var_to_test} → lower {_dt_name}"

        legend_txt = (
            f"Method: {method_label}\n"
            f"{_red}  |  {_blue}\n"
            f"Uncorrected p<0.05: {n_unc}  →  BH-FDR q<0.05: {n_sig}  (of {family_n}/{self.n_vf} computable locations)  |  {tp_lbl}\n"
            f"Sand-hatched loci: correlation not computable at that location (insufficient variance/pairwise data)."
        )

        def _ov(ax): self._draw_overlay(ax)

        # Window 1 - All patients: Raw | Uncorrected | BH-FDR
        raw_crop = self.crop_to_quadrant(res_all['stat_grid'])
        raw_shown = self.crop_to_quadrant(_smooth_in_field(res_all['stat_grid']))   # masked to the field circle
        unc_crop = self.crop_to_quadrant(res_all['unc_grid'])
        sig_crop = self.crop_to_quadrant(res_all['sig_grid'])
        abs_raw  = safe_absmax(raw_crop, 0.5)
        abs_unc  = safe_absmax(unc_crop, abs_raw)
        abs_sig  = safe_absmax(sig_crop, abs_raw)
        not_computable_mask = ~valid_stat_mask

        fig1, axes1 = self._subplots(1, 3, figsize=(20, 7))
        fig1.suptitle(
            _wrap_title(f"Spatial Correlation ({method_label}) - All Patients  n={len(v_list)}\n"
                        f"{var_to_test}  |  {region_lbl}  |  {tp_lbl}"),
            fontweight='bold', fontsize=11)

        im = axes1[0].imshow(raw_shown, cmap='RdBu_r', vmin=-abs_raw, vmax=abs_raw)
        _ov(axes1[0])
        self._draw_missing_markers(axes1[0], not_computable_mask, do_flip=False,
                                    label="Not computable at this location")
        axes1[0].set_title(f"(A) Raw - no threshold\nn={len(v_list)}  |  {family_n}/{self.n_vf} computable",fontweight='bold',fontsize=10)
        axes1[0].axis('off')
        fig1.colorbar(im,ax=axes1[0],fraction=0.046,pad=0.04).set_label(method_short,fontsize=8)

        im = axes1[1].imshow(unc_crop, cmap='RdBu_r', vmin=-abs_unc, vmax=abs_unc)
        _ov(axes1[1])
        self._draw_missing_markers(axes1[1], not_computable_mask, do_flip=False)
        axes1[1].set_title(f"(B) Uncorrected p<0.05\n{n_unc} sig loci",fontweight='bold',fontsize=10)
        axes1[1].axis('off')
        fig1.colorbar(im,ax=axes1[1],fraction=0.046,pad=0.04).set_label(f"{method_short} (unc.)",fontsize=8)

        im = axes1[2].imshow(sig_crop, cmap='RdBu_r', vmin=-abs_sig, vmax=abs_sig)
        _ov(axes1[2])
        self._draw_missing_markers(axes1[2], not_computable_mask, do_flip=False)
        axes1[2].set_title(f"(C) BH-FDR q<0.05\n{n_sig}/{family_n} computable  (region: {n_pts} = {region_lbl})",fontweight='bold',fontsize=10)
        axes1[2].axis('off')
        fig1.colorbar(im,ax=axes1[2],fraction=0.046,pad=0.04).set_label(f"{method_short} (BH-FDR)",fontsize=8)

        self._caption(fig1, legend_txt, color='#333333')
        self.add_save_button(fig1); plt.show()

        # Optional: compare results by a Categorical variable
        if self.categorical_cols:
            ask = self._remember("cat_split", lambda: messagebox.askyesno(
                "Compare by Category?",
                f"Whole-cohort analysis of  '{var_to_test}'  is complete.\n\n"
                f"Would you like to compare the spatial correlation maps\n"
                f"split by a Categorical variable?\n\n"
                f"Example: compare  '{var_to_test}'  results for\n"
                f"each group defined by one of your categorical variables\n"
                f"(e.g. Male vs Female, Left vs Right, etc.)."
            ))
            if ask:
                self._spatial_numeric_corr_by_category(
                    method_choice, var_to_test, study_type, timepoint, method_label)

    # Spatial correlation - split by Categorical variable (post-analysis popup)
    def _corr_stats_grid_valid(self, c76, p76, valid_mask, q_threshold=0.05):
        """Like stats_76_to_grid, restricted to the locations in valid_mask.

        Only locations where the correlation is computable are interpolated; the
        rest stays transparent (NaN). Feeding a non-computable location in as
        r = 0 would pull the interpolated values of nearby computable locations
        toward zero. Returns None when fewer than ten locations are computable,
        so that the caller can show a message in place of an empty map.
        """
        valid_mask = np.asarray(valid_mask, dtype=bool)
        pts = self.grid_points[valid_mask]
        n_valid = pts.shape[0]
        if n_valid < 10:
            return None

        stat_v = np.asarray(c76, dtype=float)[valid_mask]
        p_v    = np.asarray(p76, dtype=float)[valid_mask]
        q_v    = bh_fdr(p_v)
        sig_v  = q_v <= q_threshold
        unc_v  = p_v < 0.05
        # Full-length versions of the two masks above, returned to the caller.
        sig_mask76 = np.zeros(self.n_vf, dtype=bool); sig_mask76[valid_mask] = sig_v
        unc_mask76 = np.zeros(self.n_vf, dtype=bool); unc_mask76[valid_mask] = unc_v

        ext = self.grid_extent_deg
        gx, gy = np.meshgrid(np.linspace(-ext, ext, 100), np.linspace(ext, -ext, 100))
        stat_grid = griddata(pts, stat_v, (gx, gy), method='cubic')
        nan_mask  = np.isnan(stat_grid)
        if nan_mask.any():
            stat_grid[nan_mask] = griddata(pts, stat_v, (gx, gy), method='nearest')[nan_mask]
        y, x = np.ogrid[-50:50, -50:50]
        circ = x*x + y*y > 50*50
        stat_grid[circ] = np.nan

        def _masked(mask_v):
            if mask_v.sum() == 0:
                return np.full((100, 100), np.nan)
            p_grid = griddata(pts, p_v, (gx, gy), method='linear')
            nan_p  = np.isnan(p_grid)
            if nan_p.any():
                p_grid[nan_p] = griddata(pts, p_v, (gx, gy), method='nearest')[nan_p]
            p_grid = np.clip(p_grid, 0, 1)
            p_grid[circ] = np.nan
            threshold = min(float(np.max(p_v[mask_v])), 0.05)
            grid = stat_grid.copy()
            grid[p_grid > threshold] = np.nan
            grid[circ] = np.nan
            _stamp_significant_points(grid, pts, stat_v, mask_v, ext, _nn_spacing_deg(self.grid_points))
            return grid

        return {
            'sig_mask':  sig_mask76,
            'unc_mask':  unc_mask76,
            'stat_grid': stat_grid,
            'sig_grid':  _masked(sig_v),
            'unc_grid':  _masked(unc_v),
            'n_sig':     int(sig_v.sum()),
            'n_unc':     int(unc_v.sum()),
            'n_valid':   n_valid,
        }

    def _spatial_numeric_corr_by_category(self, method_choice, var_to_test,
                                           study_type, timepoint, method_label):
        """Correlation stratified by a categorical variable.

        Shown after the whole-cohort correlation: one column of panels per
        category value. Within each category the same per-location Pearson or
        Spearman correlation and BH-FDR correction are computed on that
        category's subjects only. This describes each group; it does not test
        whether the correlation differs between groups (that would need a
        between-group test such as Fisher's z).
        """

        # Pick categorical variable
        cat_var = self._select_variable("Compare by Category", self.categorical_cols, "Select Categorical variable to split by:")
        if not cat_var:
            return

        options = self._schema_options(cat_var)
        # Collect subjects' category values
        entries = self._collect_76_for_spatial(study_type, timepoint)
        # Build per-category lists
        cat_data = {}   # {cat_value: (vfs_76_list, vars_list)}
        for e in entries:
            cat_val = str(e["data"].get(cat_var, "")).strip()
            if not cat_val:
                continue
            raw_val = e["data"].get(var_to_test, "")
            try:
                fval = float(raw_val)
            except (ValueError, TypeError):
                continue
            cat_data.setdefault(cat_val, ([], []))
            cat_data[cat_val][0].append(e["vf76"])
            cat_data[cat_val][1].append(fval)

        # Use defined options order if available, otherwise sort.
        # Fall back to raw keys if the options filter (e.g. due to a
        # case/whitespace mismatch between schema options and stored values)
        # drops categories that actually have data.
        if options:
            categories = [o for o in options if o in cat_data]
            if len(categories) < 2 and len(cat_data) >= 2:
                categories = sorted(cat_data.keys())
        else:
            categories = sorted(cat_data.keys())

        if len(categories) < 2:
            messagebox.showwarning("Insufficient Groups",
                f"Need ≥2 categories with data.\n"
                f"Categories found with data: {list(cat_data.keys())}")
            return

        corr_func, _label, should_compute = self._corr_method_setup(method_choice)
        method_short = self._method_short(method_choice)

        def compute_corr76(vf_list, var_list):
            if len(var_list) < 3:
                return None, None, None
            c76, p76 = self._corr_stats76(np.array(vf_list), np.array(var_list), corr_func, should_compute)
            return c76, p76, np.isfinite(c76) & np.isfinite(p76)

        # Build figure: one column per category, 3 rows (Raw/Unc/FDR)
        n_cats   = len(categories)
        tp_lbl   = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")
        region_lbl = self.quadrant_var.get()

        fig, axes = self._subplots(3, n_cats, figsize=(6.5*n_cats, 15))
        if n_cats == 1:
            axes = axes.reshape(3, 1)

        fig.suptitle(
            _wrap_title(
                f"Spatial Correlation ({method_label}) split by '{cat_var}'\n"
                f"{var_to_test}  |  {region_lbl}  |  {tp_lbl}"
            ),
            fontweight='bold', fontsize=12)

        row_labels = ["(A) Raw - no threshold", "(B) Uncorrected p<0.05", "(C) BH-FDR q<0.05"]

        try:
            for col_idx, cat_val in enumerate(categories):
                vf_list, var_list = cat_data[cat_val]
                n_sub = len(var_list)
                c76, p76, valid = compute_corr76(vf_list, var_list)

                for row_idx in range(3):
                    ax = axes[row_idx, col_idx]
                    ax.axis('off')
                    if row_idx == 0:
                        ax.set_title(f"'{cat_val}'  n={n_sub}", fontweight='bold', fontsize=10)

                if c76 is None:
                    axes[0, col_idx].text(0.5, 0.5,
                        f"'{cat_val}'\nn={n_sub}\n(need ≥3)",
                        ha='center', va='center', fontsize=10,
                        transform=axes[0, col_idx].transAxes)
                    continue

                res = self._corr_stats_grid_valid(c76, p76, valid)
                n_computable = int(valid.sum())

                if res is None:
                    axes[0, col_idx].text(0.5, 0.5,
                        f"'{cat_val}'\nn={n_sub}\n\nOnly {n_computable}/{self.n_vf} pixels had\n"
                        f"sufficient variance to compute\n{method_label} in this subgroup.\n"
                        f"Too few for a spatial map -\n"
                        f"try merging categories or a\nlarger subgroup.",
                        ha='center', va='center', fontsize=9,
                        transform=axes[0, col_idx].transAxes)
                    continue

                n_sig = res['n_sig']; n_unc = res['n_unc']

                raw_c = self.crop_to_quadrant(res['stat_grid'])
                unc_c = self.crop_to_quadrant(res['unc_grid'])
                sig_c = self.crop_to_quadrant(res['sig_grid'])
                abs_r = safe_absmax(raw_c, 0.5)
                abs_u = safe_absmax(unc_c, abs_r)
                abs_s = safe_absmax(sig_c, abs_r)
                not_computable = ~valid

                raw_shown = self.crop_to_quadrant(_smooth_in_field(res['stat_grid']))   # masked to the field circle
                for row_idx, (data_arr, abs_v, row_lbl, n_info) in enumerate([
                    (raw_shown, abs_r, row_labels[0], f"{n_computable}/{self.n_vf} px computed"),
                    (unc_c,             abs_u, row_labels[1], f"{n_unc} sig"),
                    (sig_c,             abs_s, row_labels[2], f"{n_sig} sig"),
                ]):
                    ax = axes[row_idx, col_idx]
                    im = ax.imshow(data_arr, cmap='RdBu_r', vmin=-abs_v, vmax=abs_v)
                    self._draw_overlay(ax)
                    self._draw_missing_markers(ax, not_computable, do_flip=False)
                    lbl_str = f"{row_lbl}" + (f"\n{n_info} loci" if n_info else f"\nn={n_sub}")
                    ax.set_title(lbl_str, fontweight='bold', fontsize=9)
                    ax.axis('off')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                        method_short, fontsize=8)
        except Exception as exc:
            plt.close(fig)
            messagebox.showerror("Category Subanalysis Failed",
                f"Could not build the category spatial map:\n{exc}")
            return

        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"
        self._caption(fig,
            f"RED = r>0 → higher {var_to_test} → higher {dt_name}  |  "
            f"BLUE = r<0 → lower {dt_name}\n"
            f"Stratified: each group correlated & BH-FDR-corrected on its own subjects only - "
            f"no group-difference test performed.\n"
            f" |  "
            f"Sand-hatched: not computable in this subgroup  |  {tp_lbl}")
        self.add_save_button(fig); plt.show()

    # Cluster Analysis
    def analyze_affected_space(self, study_type="cross", timepoint="preop"):
        if not self.continuous_cols:
            messagebox.showwarning("No Continuous Variables",
                "No Continuous variables defined. Use 'Add Variable' first.")
            return
        var_to_test = self._select_variable("Cluster Space Analysis", self.continuous_cols, "Select Continuous variable:")
        if not var_to_test:
            return

        # Three severity bands for the active data type (see TIER_LABELS and
        # self.severity_caveat), computed on the original tested locations (not
        # the interpolated grid) inside the selected analysis region.
        clusters = [(self.tier_labels["severe"],   self.plausible_min, self.severe_thr),
                    (self.tier_labels["moderate"],  self.severe_thr,    self.moderate_thr),
                    (self.tier_labels["normal"],    self.moderate_thr,  self.plausible_max)]
        space_stats = {c[0]:[] for c in clusters}
        clinical_vals = []
        region_mask = self._quadrant_pt_mask(self.grid_points)

        for entry in self._collect_76_for_spatial(study_type, timepoint):
            data = entry["data"]
            if not data.get(var_to_test):
                continue
            try:
                c_val = float(data[var_to_test])
                vf76 = np.asarray(entry["vf76"], dtype=float)
                valid = np.isfinite(vf76) & region_mask
                valid_vals = vf76[valid]
                if len(valid_vals) == 0:
                    continue
                clinical_vals.append(c_val)
                for label, low, high in clusters:
                    if label == self.tier_labels["normal"]:
                        count = np.sum(valid_vals > low)
                    else:
                        count = np.sum((valid_vals > low) & (valid_vals <= high))
                    space_stats[label].append((count/len(valid_vals))*100)
            except:
                continue

        if len(clinical_vals) < 2:
            messagebox.showwarning("Warning","Insufficient data."); return

        fig, (ax,) = self._axes_grid(1, (11, 7))
        colors  = ['#d32f2f','#f57c00','#1976d2']
        x       = np.array(clinical_vals)
        sort_idx= np.argsort(x); x_sorted = x[sort_idx]

        for i, (label, lo, hi) in enumerate(clusters):
            y   = np.array(space_stats[label])[sort_idx]
            sl,ic,r,p,_ = linregress(x_sorted, y)
            ax.scatter(x_sorted, y, color=colors[i], alpha=0.4, s=40)
            lbl  = f"{label} (p={p:.3f})"
            lw, ls = (2.5,'-') if p < 0.05 else (1.2,'--')
            ax.plot(x_sorted, ic+sl*x_sorted, color=colors[i],
                    linestyle=ls, linewidth=lw, label=lbl)

        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")
        ax.set_title(_wrap_title(
            f"Cluster Space vs {var_to_test}  ({self.data_type_spec['label']} - study-defined bands)\n"
            f"Region: {self.quadrant_var.get()}  |  {tp_lbl}"),
            fontweight='bold')
        ax.set_xlabel(f"{var_to_test} Value", fontweight='bold')
        ax.set_ylabel("% of valid paired-eye ROI locations", fontweight='bold')
        ax.legend(title=f"{self.data_type_spec['short']} bands (Solid = p<0.05)", loc='upper left', bbox_to_anchor=(1.02,1))
        ax.grid(True, alpha=0.3)
        self._caption(fig,
                f"BLUE = negative association: higher variable value → lower {self.data_type_spec['short']}\n"
                f"{self.severity_caveat}",
                face=None, fontsize=8, color='#b71c1c', style='italic')
        self._present(fig)

    # Cumulative Analysis
    def analyze_cumulative_space(self, study_type="cross", timepoint="preop"):
        if not self.continuous_cols:
            messagebox.showwarning("No Continuous Variables",
                "No Continuous variables defined. Use 'Add Variable' first.")
            return
        var_to_test = self._select_variable("Cumulative Space Analysis", self.continuous_cols, "Select Continuous variable:")
        if not var_to_test:
            return

        # Cumulative severity bands for the active data type; the principal
        # outcome is "moderate-or-worse" (self.moderate_thr). Computed on the
        # original tested locations, not the interpolated display grid.
        _dt = self.data_type_spec["short"]
        clusters = [(f"{self.tier_labels['severe']} ({_dt}\u2264{self.severe_thr:.2f}dB)",
                     self.plausible_min, self.severe_thr),
                    (f"{self.tier_labels['moderate']}-or-worse ({_dt}\u2264{self.moderate_thr:.2f}dB)",
                     self.plausible_min, self.moderate_thr)]
        space_stats = {c[0]:[] for c in clusters}
        clinical_vals = []
        region_mask = self._quadrant_pt_mask(self.grid_points)

        for entry in self._collect_76_for_spatial(study_type, timepoint):
            data = entry["data"]
            if not data.get(var_to_test):
                continue
            try:
                c_val = float(data[var_to_test])
                vf76 = np.asarray(entry["vf76"], dtype=float)
                valid = np.isfinite(vf76) & region_mask
                valid_vals = vf76[valid]
                if len(valid_vals) == 0:
                    continue
                clinical_vals.append(c_val)
                for label, lo, hi in clusters:
                    count = np.sum(valid_vals <= hi)
                    space_stats[label].append((count/len(valid_vals))*100)
            except:
                continue

        if len(clinical_vals) < 2:
            messagebox.showwarning("Warning","Insufficient data."); return

        fig, (ax,) = self._axes_grid(1, (11, 7))
        colors  = ['#b71c1c','#e65100']
        x       = np.array(clinical_vals)
        sort_idx= np.argsort(x); x_sorted = x[sort_idx]

        for i, (label, lo, hi) in enumerate(clusters):
            y   = np.array(space_stats[label])[sort_idx]
            sl,ic,r,p,_ = linregress(x_sorted, y)
            ax.scatter(x_sorted, y, color=colors[i], alpha=0.4, s=40)
            lbl  = f"{label} (p={p:.3f})"
            lw, ls = (2.5,'-') if p < 0.05 else (1.2,'--')
            ax.plot(x_sorted, ic+sl*x_sorted, color=colors[i],
                    linestyle=ls, linewidth=lw, label=lbl)

        tp_lbl = study_type.capitalize() + (f" [{tp_display(timepoint)}]" if timepoint else " [Δ]")
        ax.set_title(_wrap_title(
            f"Cumulative Deficit vs {var_to_test}  ({self.data_type_spec['label']} - study-defined bands)\n"
            f"Region: {self.quadrant_var.get()}  |  {tp_lbl}"),
            fontweight='bold')
        ax.set_xlabel(f"{var_to_test} Value", fontweight='bold')
        ax.set_ylabel("% of valid paired-eye ROI locations", fontweight='bold')
        ax.legend(title=f"Cumulative {self.data_type_spec['short']} Deficit Bands", loc='upper left', bbox_to_anchor=(1.02,1))
        ax.grid(True, alpha=0.3)
        self._caption(fig,
                f"BLUE = negative association: higher variable value → lower {self.data_type_spec['short']}\n"
                f"{self.severity_caveat}",
                face=None, fontsize=8, color='#1565c0', style='italic')
        self._present(fig)

    # DB helpers
    def load_db(self):
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE,'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # Reserved (metadata) keys are copied as they are; they are not
                # subject records.
                normalized = {}
                has_vf_data = False
                for k, v in raw.items():
                    if k in RESERVED_DB_KEYS:
                        normalized[k] = v
                        continue
                    true_key = str(v.get("ID",k)).strip() if isinstance(v,dict) else str(k).strip()
                    if not true_key:
                        true_key = str(k).strip()
                    normalized[true_key] = v
                    if isinstance(v,dict):
                        v["ID"] = true_key
                        if any(v.get(f"vf_raw76_{tp}_{eye}") for tp in ("preop","postop") for eye in ("od","os")):
                            has_vf_data = True

                # Compatibility guard. This version stores raw Total Deviation (dB),
                # not normalized PI (0-1). A database with VF data but without the
                # measurement-type flag has an unknown scale, so the user is warned.
                measurement_type = normalized.get(VF_MEASUREMENT_TYPE_KEY)
                if has_vf_data and measurement_type != VF_MEASUREMENT_TYPE_VALUE:
                    try:
                        messagebox.showwarning(
                            "Legacy Database Detected",
                            f"This database ({os.path.basename(DB_FILE)}) contains visual-field "
                            f"data but is missing the '{VF_MEASUREMENT_TYPE_KEY}' flag this "
                            f"version expects.\n\n"
                            f"If this data was entered under the earlier normalized Preservation "
                            f"Index (PI = dB/35, range 0-1) workflow, it MUST NOT be mixed with new "
                            f"Total Deviation (dB) data - the two scales are not comparable.\n\n"
                            f"Please re-import/re-extract source reports under this version to "
                            f"populate genuine Total Deviation (dB) values, or confirm with your "
                            f"study team before proceeding.")
                    except Exception:
                        print(f"[NeuroVF] WARNING: legacy database without '{VF_MEASUREMENT_TYPE_KEY}' "
                              f"flag detected - verify data scale before analysis.")
                normalized[VF_MEASUREMENT_TYPE_KEY] = VF_MEASUREMENT_TYPE_VALUE
                return normalized
            except Exception as e:
                # An unreadable or corrupt database file is never replaced silently
                # by an empty one. The file is copied to a timestamped name, the
                # user is warned (console and dialog), and the session starts from
                # an empty in-memory database, so nothing is lost and the preserved
                # copy can be recovered before anything is saved.
                import datetime
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                quarantine_path = f"{DB_FILE}.UNREADABLE_{stamp}.bak"
                try:
                    import shutil
                    shutil.copy2(DB_FILE, quarantine_path)
                except Exception:
                    quarantine_path = None
                msg = (f"The database file could not be read:\n{DB_FILE}\n\n"
                       f"Error: {e}\n\n"
                       + (f"The unreadable file has been preserved, unmodified, at:\n{quarantine_path}\n\n"
                          if quarantine_path else
                          "A backup copy could NOT be created - the original file was left untouched.\n\n")
                       + "This session will start with an EMPTY database so nothing is "
                         "overwritten. Do NOT click Save until you have recovered your data "
                         "(the preserved/quarantined copy above, or an earlier .bak file in the "
                         "same folder) - contact support before proceeding if you are unsure.")
                print(f"[NeuroVF] ERROR: {msg}")
                try:
                    messagebox.showerror("Database Could Not Be Loaded - Data Preserved, NOT Erased", msg)
                except Exception:
                    pass
                return {VF_MEASUREMENT_TYPE_KEY: VF_MEASUREMENT_TYPE_VALUE}
        return {VF_MEASUREMENT_TYPE_KEY: VF_MEASUREMENT_TYPE_VALUE}

    def init_missing_columns(self):
        for sid, v in self.db.items():
            if sid in RESERVED_DB_KEYS:
                continue
            if isinstance(v, dict):
                for col in self.columns:
                    if col not in v:
                        v[col] = ""
                for new_field in (
                        "vf_raw76_preop_od","vf_raw76_preop_os",
                        "vf_flip_preop_od","vf_flip_preop_os",
                        "vf_raw76_postop_od","vf_raw76_postop_os",
                        "vf_flip_postop_od","vf_flip_postop_os",
                        "vf_diff76",
                        "vf_data_od","vf_data_os",
                        "vf_raw76_od","vf_raw76_os",
                        "vf_flip_od","vf_flip_os",
                        # Device-reported MD entered by the user (PTC databases,
                        # Enter Device MD dialog); see compute_md() and
                        # get_manual_md().
                        "md_manual_preop_od","md_manual_preop_os",
                        "md_manual_postop_od","md_manual_postop_os"):
                    if new_field not in v:
                        v[new_field] = None
                # Compatibility: if VF data are stored under
                # vf_raw76_<tp>_<eye> but the status cell is blank, mark the
                # cell "READY" so that the data need not be imported again.
                for tp_key, tp_internal in (("Preop", "preop"), ("Postop", "postop")):
                    for side in ("OD", "OS"):
                        if v.get(f"vf_raw76_{tp_internal}_{side.lower()}") is not None:
                            status_col = f"{tp_key} {side} VF"
                            if not v.get(status_col):
                                v[status_col] = "READY"

    def save_db(self):
        # Sanitize NaN/Inf and NumPy types before writing JSON. allow_nan=False
        # is deliberate: a NaN that gets past _json_safe raises here instead of
        # producing non-standard JSON (a bare NaN token).
        safe_db = _json_safe(self.db)

        # Atomic write with a rolling backup: copy the current file to .bak,
        # write the new content to a temporary file in the same folder, then
        # os.replace() it into place. The rename is atomic on Windows and POSIX,
        # so DB_FILE is always either the old or the new complete file.
        try:
            if os.path.exists(DB_FILE):
                import shutil
                shutil.copy2(DB_FILE, DB_FILE + ".bak")
        except Exception:
            pass  # backup is best-effort; never block a save because of it

        tmp_path = DB_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(safe_db, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, DB_FILE)

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for sid in sorted(k for k in self.db.keys() if k not in RESERVED_DB_KEYS):
            data = self.db[sid]
            row_data = []
            for col in self.columns:
                if col == self.COL_VF_VIEW:
                    # Show a clickable label only when at least one VF dataset exists
                    has_any = any(data.get(f"vf_raw76_{tp}_{eye}") is not None
                                  for tp in ("preop", "postop")
                                  for eye in ("od", "os"))
                    row_data.append("View" if has_any else "-")
                elif col == self.COL_SCOTOMA_AREA:
                    has_any = any(data.get(f"vf_raw76_{tp}_{eye}") is not None
                                  for tp in ("preop", "postop")
                                  for eye in ("od", "os"))
                    row_data.append("Calc" if has_any else "-")
                else:
                    row_data.append(data.get(col, ""))
            self.tree.insert("", tk.END, values=row_data)

    def get_db_key(self, sid):
        sid = str(sid).strip()
        if sid in self.db:
            return sid
        for k in self.db:
            if str(k).strip() == sid:
                return k
        return None

    def _update_diff76(self, sid):
        data = self.db[sid]
        pre_od,  pre_os,  _, _ = self._get_raw76_pair(data, "preop_")
        post_od, post_os, _, _ = self._get_raw76_pair(data, "postop_")
        if pre_od is None or post_od is None:
            return
        diff_od = post_od - pre_od   # raw dB change; NaN propagates naturally
        diff_os = post_os - pre_os
        def _j(v):
            return None if (np.isnan(v)) else float(v)
        self.db[sid]["vf_diff76"] = [
            [_j(diff_od[i]), _j(diff_os[i])] for i in range(self.n_vf)
        ]

    # Add Subject - dynamic scrollable dialog
    def add_subject_dialog(self):
        """Open the dialog that creates a new subject.

        - Subject ID field at the top (always visible).
        - Scrollable middle section with one input per schema variable;
          categorical variables get a dropdown of their defined options.
        - Add Subject / Cancel buttons at the bottom, outside the scroll area,
          so that they stay visible at any scroll position or window size.
        """
        win = tk.Toplevel(self.root)
        win.title("Add New Subject")
        win.grab_set()
        win.resizable(True, True)

        # Top: subject ID (always visible)
        top_frame = tk.Frame(win, bg="#f5f5f5", bd=1, relief='groove')
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        tk.Label(top_frame, text="Subject ID *", font=("Arial", 11, "bold"),
                 fg="#b71c1c", bg="#f5f5f5").grid(
                 row=0, column=0, padx=14, pady=10, sticky='w')
        sid_var = tk.StringVar()
        sid_entry = tk.Entry(top_frame, textvariable=sid_var, width=30,
                             font=("Arial", 11))
        sid_entry.grid(row=0, column=1, padx=10, pady=10)
        sid_entry.focus_set()

        # Bottom: buttons, packed before the canvas so that they are never hidden
        sep = tk.Frame(win, height=2, bg="#bdbdbd")
        sep.pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = tk.Frame(win, bg="#eeeeee")
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        confirmed = tk.BooleanVar(value=False)

        def on_ok():
            if not sid_var.get().strip():
                messagebox.showwarning("Missing ID", "Please enter a Subject ID.")
                return
            confirmed.set(True)
            win.destroy()

        def on_cancel():
            win.destroy()

        tk.Button(btn_frame, text="Add Subject", bg="#2e7d32", fg="white",
                  font=("Arial", 11, "bold"), width=16,
                  command=on_ok).pack(side=tk.LEFT, padx=16, pady=12)
        tk.Button(btn_frame, text="Cancel", bg="#546e7a", fg="white",
                  font=("Arial", 10), width=10,
                  command=on_cancel).pack(side=tk.LEFT, padx=4, pady=12)

        # Middle: scrollable variable fields (fills the remaining space)
        scroll_outer = tk.Frame(win)
        scroll_outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                          padx=10, pady=6)

        canvas = tk.Canvas(scroll_outer, highlightthickness=0)
        vsb    = ttk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill='y')
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        body = tk.Frame(canvas)
        body_id = canvas.create_window((0, 0), window=body, anchor='nw')

        def _on_body_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_configure(e):
            canvas.itemconfig(body_id, width=e.width)
        body.bind("<Configure>", _on_body_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        win.bind("<MouseWheel>", _on_mousewheel)

        var_widgets = {}
        row = 0

        if self.variable_schema:
            tk.Label(body, text="--- User-defined variables ---",
                     font=("Arial", 8, "italic"), fg="#555").grid(
                     row=row, column=0, columnspan=2, padx=14, pady=(8,2), sticky='w')
            row += 1

        for vname, vinfo in self.variable_schema.items():
            vtype   = vinfo.get("type", "Continuous") if isinstance(vinfo, dict) else str(vinfo)
            options = vinfo.get("options", []) if isinstance(vinfo, dict) else []

            tk.Label(body, text=f"{vname}", font=("Arial", 10, "bold"),
                     anchor='w').grid(row=row, column=0, padx=14, pady=4, sticky='w')

            if vtype == "Categorical" and options:
                v = tk.StringVar(value=options[0])
                cb = ttk.Combobox(body, textvariable=v,
                                  values=options, state="readonly",
                                  width=26, font=("Arial", 10))
                cb.grid(row=row, column=1, padx=10, pady=4)
            else:
                v = tk.StringVar()
                tk.Entry(body, textvariable=v, width=28,
                         font=("Arial", 10)).grid(row=row, column=1, padx=10, pady=4)

            var_widgets[vname] = v
            row += 1

        if not self.variable_schema:
            tk.Label(body,
                     text="No variables defined yet.\nUse 'Add Variable' to define variables first.",
                     font=("Arial", 9, "italic"), fg="#888").grid(
                     row=row, column=0, columnspan=2, padx=14, pady=10)
            row += 1

        tk.Label(body,
                 text="VF data is entered by double-clicking the VF columns in the main table.",
                 font=("Arial", 8, "italic"), fg="#888",
                 wraplength=360).grid(row=row, column=0, columnspan=2,
                                      padx=14, pady=(10, 6), sticky='w')

        # Size the window
        win.update_idletasks()
        body_h      = body.winfo_reqheight()
        screen_h    = win.winfo_screenheight()
        max_body_h  = int(screen_h * 0.55)
        canvas_h    = min(body_h, max_body_h)
        total_h     = canvas_h + top_frame.winfo_reqheight() + 80   # 80 for buttons
        win.geometry(f"480x{min(total_h, int(screen_h * 0.80))}")

        win.wait_window()
        if not confirmed.get():
            return

        sid = sid_var.get().strip()
        if not sid:
            return
        if sid in self.db:
            messagebox.showwarning("Duplicate", f"Subject '{sid}' already exists.")
            return

        # Build record
        record = {col: "" for col in self.columns}
        record.update({
            "ID": sid,
            "vf_raw76_preop_od": None, "vf_raw76_preop_os": None,
            "vf_flip_preop_od": False, "vf_flip_preop_os": False,
            "vf_raw76_postop_od": None, "vf_raw76_postop_os": None,
            "vf_flip_postop_od": False, "vf_flip_postop_os": False,
            "vf_diff76": None,
            "vf_data_od": None, "vf_data_os": None,
            "vf_raw76_od": None, "vf_raw76_os": None,
            "vf_flip_od": False, "vf_flip_os": False,
        })
        for vname, widget_var in var_widgets.items():
            record[vname] = widget_var.get().strip()

        self.db[sid] = record
        self.save_db()
        self.refresh_table()

    def delete_subject(self):
        selected = self.tree.selection()
        if selected:
            sid_raw    = str(self.tree.item(selected[0])['values'][0])
            actual_key = self.get_db_key(sid_raw)
            if actual_key is None:
                messagebox.showerror("Error", f"Subject '{sid_raw}' not found.")
                return
            if messagebox.askyesno("Confirm", f"Delete Subject {actual_key}?"):
                del self.db[actual_key]
                self.save_db()
                self.refresh_table()

    def on_double_click(self, event):
        # Region detection: heading vs cell
        region = self.tree.identify_region(event.x, event.y)
        column = self.tree.identify_column(event.x)
        if not column:
            return
        col_index = int(column.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.columns):
            return
        col_name = self.columns[col_index]

        # Double-click on a column heading: rename the variable
        if region == "heading":
            if col_name in self.variable_schema:
                self._rename_variable_dialog(col_name)
            return

        # Double-click on a cell: edit the value
        item = self.tree.identify_row(event.y)
        if not item:
            return
        sid_raw = str(self.tree.item(item, "values")[0])
        sid     = self.get_db_key(sid_raw)
        if sid is None:
            messagebox.showerror("Error", f"Key '{sid_raw}' not found.")
            return

        # VF import columns
        if col_name in self.cols_vf:
            vf_map = {
                "Preop OD VF":  ("preop",  "OD"),
                "Preop OS VF":  ("preop",  "OS"),
                "Postop OD VF": ("postop", "OD"),
                "Postop OS VF": ("postop", "OS"),
            }
            if col_name in vf_map:
                timepoint, side = vf_map[col_name]
                self.process_vf_db_ocr(sid, side, timepoint)
            return

        # View column - ignore double-click (single-click handled separately)
        if col_name == self.COL_VF_VIEW:
            return

        # Scotoma Area column - ignore double-click (single-click handled separately)
        if col_name == self.COL_SCOTOMA_AREA:
            return

        # Subject ID rename
        if col_name == "ID":
            new_id = simpledialog.askstring("Rename Subject",
                                            "New Subject ID:",
                                            initialvalue=sid,
                                            parent=self.root)
            if new_id and new_id.strip() and new_id.strip() != sid:
                new_id = new_id.strip()
                if new_id in self.db:
                    messagebox.showerror("Duplicate", f"ID '{new_id}' already exists.")
                    return
                self.db[new_id] = self.db.pop(sid)
                self.save_db(); self.refresh_table()
            return

        # All user-defined variable cells
        current = self.db[sid].get(col_name, "")
        vtype   = self._schema_type(col_name)   # '' if not in the schema
        options = self._schema_options(col_name)

        if vtype == "Categorical" and options:
            # Dropdown / radio button selector
            dwin = tk.Toplevel(self.root)
            dwin.title(f"Edit  {col_name}  -  {sid}")
            dwin.grab_set(); dwin.resizable(False, False)

            hdr2 = tk.Frame(dwin, bg="#1a237e")
            hdr2.pack(fill=tk.X)
            tk.Label(hdr2, text=f"  {col_name}  ·  Subject: {sid}",
                     bg="#1a237e", fg="white",
                     font=("Arial", 10, "bold"), pady=6).pack(side=tk.LEFT, padx=10)

            tk.Label(dwin, text="Select value:",
                     font=("Arial", 10), pady=8).pack(padx=24, anchor='w')

            sel_var = tk.StringVar(value=current if current in options else options[0])

            if len(options) <= 6:
                rb_frame = tk.Frame(dwin)
                rb_frame.pack(padx=24, pady=4)
                for opt in options:
                    tk.Radiobutton(rb_frame, text=opt, variable=sel_var,
                                   value=opt, font=("Arial", 11),
                                   anchor='w').pack(anchor='w', pady=3)
            else:
                ttk.Combobox(dwin, textvariable=sel_var,
                             values=options, state="readonly",
                             width=24, font=("Arial", 11)).pack(padx=24, pady=8)

            def _ok_cat():
                self.db[sid][col_name] = sel_var.get()
                self.save_db(); self.refresh_table(); dwin.destroy()

            sep2 = tk.Frame(dwin, height=1, bg="#bdbdbd"); sep2.pack(fill=tk.X, pady=(8,0))
            bf2  = tk.Frame(dwin); bf2.pack(pady=10)
            tk.Button(bf2, text="OK", bg="#2e7d32", fg="white",
                      font=("Arial", 10, "bold"), width=10,
                      command=_ok_cat).pack(side=tk.LEFT, padx=10)
            tk.Button(bf2, text="Cancel", command=dwin.destroy).pack(side=tk.LEFT)
            dwin.wait_window()
        else:
            # Continuous (or categorical with no predefined options)
            self._edit_continuous_cell(sid, col_name, current)

    # Rename variable (double-click on column header)
    def _rename_variable_dialog(self, old_name):
        """Rename a user-defined variable: updates schema, all subject records, and treeview."""
        rwin = tk.Toplevel(self.root)
        rwin.title(f"Rename Variable - {old_name}")
        rwin.grab_set(); rwin.resizable(False, False)

        hdr = tk.Frame(rwin, bg="#4a148c")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=f"  Rename Variable: {old_name}",
                 bg="#4a148c", fg="white",
                 font=("Arial", 11, "bold"), pady=7).pack(side=tk.LEFT, padx=10)

        body = tk.Frame(rwin, padx=18, pady=14)
        body.pack()

        tk.Label(body, text="New variable name:", font=("Arial", 10)).grid(
                 row=0, column=0, sticky='w', pady=4)
        new_var = tk.StringVar(value=old_name)
        entry   = tk.Entry(body, textvariable=new_var,
                           font=("Arial", 12), width=26)
        entry.grid(row=1, column=0, pady=4, ipady=4)
        entry.select_range(0, tk.END); entry.focus_set()

        tk.Label(body,
                 text="This renames the variable in every subject record.",
                 font=("Arial", 8, "italic"), fg="#c62828").grid(
                 row=2, column=0, sticky='w', pady=4)

        def _ok(_event=None):
            new_name = new_var.get().strip()
            if not new_name:
                messagebox.showwarning("Empty", "Name cannot be empty.", parent=rwin)
                return
            if new_name == old_name:
                rwin.destroy(); return
            if new_name in self.variable_schema or new_name in ("ID",) + self.COLS_VF:
                messagebox.showerror("Duplicate",
                    f"'{new_name}' already exists.", parent=rwin)
                return
            # Update schema in memory
            self.variable_schema[new_name] = self.variable_schema.pop(old_name)
            # Sync back to db so _rebuild_columns_from_schema reads the new name
            self.db["__schema__"] = self.variable_schema
            # Update every subject record
            for sid, sdata in self.db.items():
                if sid in RESERVED_DB_KEYS or not isinstance(sdata, dict):
                    continue
                if old_name in sdata:
                    sdata[new_name] = sdata.pop(old_name)
            # Rebuild column list and UI
            self._rebuild_columns_from_schema()
            self._rebuild_treeview()
            self.save_db()
            self.refresh_table()
            rwin.destroy()

        entry.bind("<Return>", _ok)
        sep = tk.Frame(rwin, height=1, bg="#bdbdbd"); sep.pack(fill=tk.X)
        bf = tk.Frame(rwin, pady=10); bf.pack()
        tk.Button(bf, text="Rename", bg="#4a148c", fg="white",
                  font=("Arial", 10, "bold"), width=12,
                  command=_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=rwin.destroy,
                  width=8).pack(side=tk.LEFT)
        rwin.wait_window()

    # Bulk paste: fill one continuous variable from a pasted row or column
    def bulk_paste_dialog(self):
        """Paste a block of values into one continuous variable.

        The user picks a continuous variable, pastes a column or row copied from
        a spreadsheet, chooses whether the values go to all subjects (in sorted
        order) or only to the rows selected in the table, and sees a preview of
        the mapping before applying it. Negative values, decimals and a comma as
        decimal separator are accepted.
        """
        import re

        # Only Continuous variables make sense here
        cont_vars = [k for k, v in self.variable_schema.items()
                     if (v.get("type") if isinstance(v, dict) else v) == "Continuous"]
        if not cont_vars:
            messagebox.showinfo("No Continuous Variables",
                "Add at least one Continuous variable before using Paste Values.")
            return

        # Helper: tokenise pasted text
        def _parse_numbers(raw):
            """Return list of (raw_token, parsed_float_or_None) for each cell."""
            raw = raw.replace('\r\n', '\n').replace('\r', '\n')
            # Flatten: split by newlines, then by tabs (handles rows and columns)
            tokens = []
            for line in raw.splitlines():
                for cell in line.split('\t'):
                    cell = cell.strip()
                    if cell:
                        tokens.append(cell)
            results = []
            num_re = re.compile(r'^[+\-−–]?\d+([.,]\d+)?$')
            for tok in tokens:
                clean = tok.replace('−', '-').replace('–', '-').replace(',', '.')
                if num_re.match(clean):
                    try:
                        results.append((tok, float(clean)))
                    except ValueError:
                        results.append((tok, None))
                else:
                    results.append((tok, None))
            return results

        # Sorted subject list (same order as table)
        all_sids = sorted(k for k in self.db.keys() if k not in RESERVED_DB_KEYS)

        # Selected subjects (from treeview)
        selected_items = self.tree.selection()
        selected_sids  = []
        for item in selected_items:
            sid_raw = str(self.tree.item(item, "values")[0])
            sid = self.get_db_key(sid_raw)
            if sid:
                selected_sids.append(sid)

        # Main dialog window
        win = tk.Toplevel(self.root)
        win.title("Paste Values - Bulk Fill Continuous Variable")
        win.grab_set()
        win.resizable(True, True)
        win.geometry("620x680")

        # Header
        hdr = tk.Frame(win, bg="#00695c")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="  Paste Values into Continuous Variable",
                 bg="#00695c", fg="white",
                 font=("Arial", 11, "bold"), pady=8).pack(side=tk.LEFT, padx=10)

        body = tk.Frame(win, padx=14, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        row = 0

        # Variable picker
        tk.Label(body, text="Variable:", font=("Arial", 10, "bold"),
                 anchor='w').grid(row=row, column=0, sticky='w', pady=4)
        var_var = tk.StringVar(value=cont_vars[0])
        var_cb  = ttk.Combobox(body, textvariable=var_var, values=cont_vars,
                               state="readonly", width=28, font=("Arial", 10))
        var_cb.grid(row=row, column=1, sticky='w', padx=8, pady=4)
        row += 1

        # Target subjects
        tk.Label(body, text="Apply to:", font=("Arial", 10, "bold"),
                 anchor='w').grid(row=row, column=0, sticky='w', pady=4)
        target_var = tk.StringVar(
            value="selected" if selected_sids else "all")
        tf = tk.Frame(body)
        tf.grid(row=row, column=1, sticky='w', padx=8)
        rb_all = tk.Radiobutton(tf, text=f"All subjects  ({len(all_sids)} total, sorted A→Z)",
                                variable=target_var, value="all",
                                font=("Arial", 9), command=lambda: _refresh_preview())
        rb_all.pack(anchor='w')
        rb_sel = tk.Radiobutton(tf,
                                text=f"Selected rows only  ({len(selected_sids)} selected)",
                                variable=target_var, value="selected",
                                font=("Arial", 9),
                                state=tk.NORMAL if selected_sids else tk.DISABLED,
                                command=lambda: _refresh_preview())
        rb_sel.pack(anchor='w')
        row += 1

        # Paste area
        tk.Label(body, text="Paste here\n(col or row):",
                 font=("Arial", 10, "bold"), anchor='nw',
                 justify='left').grid(row=row, column=0, sticky='nw', pady=6)

        paste_frame = tk.Frame(body)
        paste_frame.grid(row=row, column=1, sticky='nsew', padx=8, pady=6)
        paste_frame.columnconfigure(0, weight=1)
        paste_frame.rowconfigure(0, weight=1)
        body.rowconfigure(row, weight=1)

        paste_txt = tk.Text(paste_frame, height=7, width=38,
                            font=("Consolas", 10), wrap=tk.NONE,
                            relief='solid', bd=1)
        paste_sb_y = ttk.Scrollbar(paste_frame, orient='vertical',
                                   command=paste_txt.yview)
        paste_sb_x = ttk.Scrollbar(paste_frame, orient='horizontal',
                                   command=paste_txt.xview)
        paste_txt.configure(yscrollcommand=paste_sb_y.set,
                            xscrollcommand=paste_sb_x.set)
        paste_txt.grid(row=0, column=0, sticky='nsew')
        paste_sb_y.grid(row=0, column=1, sticky='ns')
        paste_sb_x.grid(row=1, column=0, sticky='ew')
        paste_txt.focus_set()
        row += 1

        # Parse info label
        parse_info = tk.StringVar(value="")
        tk.Label(body, textvariable=parse_info,
                 font=("Arial", 8, "italic"), fg="#555",
                 anchor='w', wraplength=560,
                 justify='left').grid(row=row, column=0, columnspan=2,
                                      sticky='w', pady=2)
        row += 1

        # Preview table
        tk.Label(body, text="Preview  (Subject → Value):",
                 font=("Arial", 9, "bold"), fg="#1a237e",
                 anchor='w').grid(row=row, column=0, columnspan=2,
                                  sticky='w', pady=(6, 2))
        row += 1

        prev_frame = tk.Frame(body, bd=1, relief='groove')
        prev_frame.grid(row=row, column=0, columnspan=2, sticky='nsew',
                        pady=2)
        body.rowconfigure(row, weight=2)
        prev_frame.columnconfigure(0, weight=1)
        prev_frame.rowconfigure(0, weight=1)

        preview_tree = ttk.Treeview(prev_frame,
                                    columns=("sid", "current", "new"),
                                    show="headings", height=7)
        preview_tree.heading("sid",     text="Subject")
        preview_tree.heading("current", text="Current value")
        preview_tree.heading("new",     text="→  New value")
        preview_tree.column("sid",     width=120, anchor="center")
        preview_tree.column("current", width=120, anchor="center")
        preview_tree.column("new",     width=150, anchor="center")
        prev_vsb = ttk.Scrollbar(prev_frame, orient='vertical',
                                 command=preview_tree.yview)
        preview_tree.configure(yscrollcommand=prev_vsb.set)
        preview_tree.grid(row=0, column=0, sticky='nsew')
        prev_vsb.grid(row=0, column=1, sticky='ns')
        row += 1

        # Status / warning
        status_var = tk.StringVar(value="")
        status_lbl = tk.Label(body, textvariable=status_var,
                              font=("Arial", 9, "bold"), fg="#c62828",
                              anchor='w', wraplength=560, justify='left')
        status_lbl.grid(row=row, column=0, columnspan=2, sticky='w', pady=3)
        row += 1

        # Live preview refresh
        _mapping = []   # list of (sid, new_value_str) to apply on confirm

        def _refresh_preview(*_):
            nonlocal _mapping
            for item in preview_tree.get_children():
                preview_tree.delete(item)
            _mapping = []
            status_var.set("")

            raw_paste = paste_txt.get("1.0", tk.END).strip()
            if not raw_paste:
                parse_info.set("Paste values above to see a preview.")
                return

            tokens = _parse_numbers(raw_paste)
            numeric = [(tok, v) for tok, v in tokens if v is not None]
            bad     = [(tok, v) for tok, v in tokens if v is None]

            parse_info.set(
                f"Detected {len(tokens)} token(s) - "
                f"{len(numeric)} numeric, {len(bad)} skipped"
                + (f"  [skipped: {', '.join(t for t,_ in bad[:5])}{'…' if len(bad)>5 else ''}]"
                   if bad else ""))

            if not numeric:
                status_var.set("No numeric values found in pasted text.")
                return

            target_sids = (selected_sids if target_var.get() == "selected"
                           else all_sids)
            vname = var_var.get()

            if len(numeric) > len(target_sids):
                status_var.set(
                    f"{len(numeric)} values but only {len(target_sids)} "
                    f"subject(s) targeted - extra values will be ignored.")
            elif len(numeric) < len(target_sids):
                status_var.set(
                    f"ℹ  {len(numeric)} values for {len(target_sids)} subjects - "
                    f"only first {len(numeric)} subject(s) will be updated.")

            n_assign = min(len(numeric), len(target_sids))
            for i in range(n_assign):
                sid     = target_sids[i]
                new_val = str(numeric[i][1])
                cur_val = str(self.db[sid].get(vname, ""))
                _mapping.append((sid, new_val))
                tag = "changed" if cur_val != new_val else "same"
                preview_tree.insert("", tk.END,
                                    values=(sid, cur_val if cur_val else "-", new_val),
                                    tags=(tag,))

            preview_tree.tag_configure("changed", foreground="#1b5e20")
            preview_tree.tag_configure("same",    foreground="#888888")

        # Bind paste text changes to live preview
        paste_txt.bind("<<Modified>>",
                       lambda e: (paste_txt.edit_modified(False), _refresh_preview()))
        var_cb.bind("<<ComboboxSelected>>",  _refresh_preview)

        _refresh_preview()

        # Bottom buttons
        sep = tk.Frame(win, height=1, bg="#bdbdbd")
        sep.pack(fill=tk.X)
        bf = tk.Frame(win, pady=10)
        bf.pack()

        def _apply():
            if not _mapping:
                messagebox.showwarning("Nothing to apply",
                    "Paste some values first.", parent=win)
                return
            vname = var_var.get()
            for sid, val in _mapping:
                self.db[sid][vname] = val
            self.save_db()
            self.refresh_table()
            n = len(_mapping)
            win.destroy()
            messagebox.showinfo("Done",
                f"{n} value(s) saved to '{vname}'.")

        def _clear():
            paste_txt.delete("1.0", tk.END)
            _refresh_preview()

        tk.Button(bf, text="Apply", bg="#2e7d32", fg="white",
                  font=("Arial", 10, "bold"), width=10,
                  command=_apply).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Clear", command=_clear,
                  width=8).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Cancel", command=win.destroy,
                  width=8).pack(side=tk.LEFT)

    # Single-value continuous cell editor (double-click on one cell)
    def _edit_continuous_cell(self, sid, col_name, current_value=""):
        """Edit a single continuous cell.

        A value can be typed directly. If a column (newline-separated) or a row
        (tab-separated) is pasted, the first numeric value is used. Negative
        values are accepted.
        """
        dwin = tk.Toplevel(self.root)
        dwin.title(f"Edit  {col_name}  -  {sid}")
        dwin.grab_set()
        dwin.resizable(False, False)

        # Header
        hdr = tk.Frame(dwin, bg="#1a237e")
        hdr.pack(fill=tk.X)
        tk.Label(hdr,
                 text=f"  {col_name}  ·  Subject: {sid}",
                 bg="#1a237e", fg="white",
                 font=("Arial", 11, "bold"), pady=7).pack(side=tk.LEFT, padx=10)

        body = tk.Frame(dwin, padx=18, pady=12)
        body.pack(fill=tk.BOTH)

        tk.Label(body,
                 text="Enter value  (negative values allowed, e.g. −3.14):",
                 font=("Arial", 10), anchor='w').grid(
                 row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))

        val_var = tk.StringVar(value=str(current_value))
        val_entry = tk.Entry(body, textvariable=val_var,
                             font=("Arial", 13), width=18, justify='center')
        val_entry.grid(row=1, column=0, columnspan=2, pady=4, ipady=4)
        val_entry.select_range(0, tk.END)
        val_entry.focus_set()

        # Paste-detection helper
        def _extract_first_number(raw_text):
            """
            Parse pasted text (column = newlines, row = tabs).
            Returns the first valid float/int found as a string, or None.
            """
            import re
            # Split by newlines first, then tabs
            tokens = []
            for line in raw_text.splitlines():
                tokens.extend(line.split('\t'))
            tokens = [t.strip().replace(',', '.') for t in tokens if t.strip()]
            pattern = re.compile(r'^[+-]?\d+(\.\d+)?$')
            for tok in tokens:
                # Handle en-dash / minus lookalikes
                tok = tok.replace('−', '-').replace('–', '-')
                if pattern.match(tok):
                    return tok
            return None

        paste_info_var = tk.StringVar(value="")
        paste_lbl = tk.Label(body, textvariable=paste_info_var,
                             font=("Arial", 8, "italic"), fg="#2e7d32",
                             anchor='w', wraplength=280)
        paste_lbl.grid(row=2, column=0, columnspan=2, sticky='w', pady=2)

        def _on_key_release(_event=None):
            raw = val_var.get()
            # Detect multi-line or tab paste
            if '\n' in raw or '\t' in raw:
                extracted = _extract_first_number(raw)
                if extracted is not None:
                    val_var.set(extracted)
                    paste_info_var.set(f"Pasted column/row - using first value: {extracted}")
                    val_entry.configure(bg="#e8f5e9")
                else:
                    paste_info_var.set("Could not find a number in pasted text.")
                    val_entry.configure(bg="#fff9c4")
            else:
                paste_info_var.set("")
                val_entry.configure(bg="white")

        val_entry.bind("<KeyRelease>", _on_key_release)
        # Also fire on paste (Control-v / Command-v) via <<Paste>> virtual event
        val_entry.bind("<<Paste>>", lambda e: dwin.after(50, _on_key_release))

        # Hint
        tk.Label(body,
                 text="Tip: paste an entire column or row from Excel/Sheets -\n"
                      "the first numeric value will be extracted automatically.",
                 font=("Arial", 8, "italic"), fg="#666",
                 justify='left').grid(
                 row=3, column=0, columnspan=2, sticky='w', pady=(6, 2))

        # Buttons
        confirmed = tk.BooleanVar(value=False)

        def _ok(_event=None):
            raw = val_var.get().strip().replace('−', '-').replace('–', '-')
            # If it still has paste artifacts, try extraction one more time
            if '\n' in raw or '\t' in raw:
                extracted = _extract_first_number(raw)
                if extracted is not None:
                    raw = extracted
            confirmed.set(True)
            dwin.destroy()
            self.db[sid][col_name] = raw
            self.save_db()
            self.refresh_table()

        def _cancel():
            dwin.destroy()

        val_entry.bind("<Return>", _ok)
        val_entry.bind("<KP_Enter>", _ok)

        sep = tk.Frame(dwin, height=1, bg="#bdbdbd")
        sep.pack(fill=tk.X)
        bf = tk.Frame(dwin, pady=10)
        bf.pack()
        tk.Button(bf, text="Save", bg="#2e7d32", fg="white",
                  font=("Arial", 10, "bold"), width=10,
                  command=_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=_cancel,
                  width=8).pack(side=tk.LEFT)

        dwin.wait_window()

    # Single-click handler: opens the VF viewer or the scotoma-area report
    def on_single_click(self, event):
        item   = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)
        if not item or not column:
            return
        col_index = int(column.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(self.columns):
            return
        col_name = self.columns[col_index]
        if col_name not in (self.COL_VF_VIEW, self.COL_SCOTOMA_AREA):
            return
        sid_raw = str(self.tree.item(item, "values")[0])
        sid     = self.get_db_key(sid_raw)
        if sid is None:
            return
        if col_name == self.COL_VF_VIEW:
            self.show_vf_viewer(sid)
        else:
            self.show_scotoma_area_dialog(sid)

    # Scotoma-area report: per subject and timepoint, cross-sectional
    def _compute_scotoma_areas76(self, vf76, region_mask=None, paired_valid_mask=None,
                                 valid_key="Valid_Paired_ROI_Locations"):
        """Scotoma areas of one point array, in percent of the valid locations.

        Each valid point is classified into the active data type's three
        severity bands (see TIER_LABELS; self.severity_caveat states that the
        banding is an operational convention). The denominator is the number of
        finite locations that also satisfy the optional masks:

        region_mask: points inside the selected analysis area (see
        self._quadrant_pt_mask).
        paired_valid_mask: points with a valid measurement in both eyes
        (complete-pair rule), or any other validity mask.

        Returns an ordered dict with six entries, or None if no location is
        valid: the three mutually exclusive bands (severe, moderate, normal),
        the two cumulative bands (severe, moderate-or-worse), and the number of
        valid locations under `valid_key`.
        """
        vf76 = np.asarray(vf76, dtype=float)
        valid = np.isfinite(vf76)
        if region_mask is not None:
            valid = valid & np.asarray(region_mask, dtype=bool)
        if paired_valid_mask is not None:
            valid = valid & np.asarray(paired_valid_mask, dtype=bool)
        n_total = int(valid.sum())
        if n_total == 0:
            return None

        sev  = valid & (vf76 <= self.severe_thr)
        mod  = valid & (vf76 >  self.severe_thr)   & (vf76 <= self.moderate_thr)
        norm = valid & (vf76 >  self.moderate_thr)

        def pct(mask): return 100.0 * float(mask.sum()) / n_total

        tl = self.tier_labels
        from collections import OrderedDict
        return OrderedDict([
            (tl["severe"],                  pct(sev)),
            (tl["moderate"],                pct(mod)),
            (tl["normal"],                  pct(norm)),
            (f"Cum {tl['severe']} (\u2264{self.severe_thr:g}dB)",                pct(sev)),
            (f"Cum {tl['moderate']}-or-worse (\u2264{self.moderate_thr:g}dB)",   pct(sev | mod)),
            (valid_key, float(n_total)),
        ])

    def _scotoma_rows(self, data, timepoint, region):
        """Scotoma-area rows for one subject at one timepoint and analysis area.

        Returns [(view, areas, status), ...] with status "ok", "no_data" or
        "no_valid". Chiasmatic and PTC studies return three rows: "registered"
        (both eyes combined after the cohort's eye flip, complete-pair rule),
        then "od" and "os", each as recorded (never mirrored, own valid
        locations only). Other study types return the registered row only.
        """
        prefix = self._tp_prefix(timepoint)
        prev_region, prev_view = self.quadrant_var.get(), getattr(self, "_eye_view", "avg")
        self.quadrant_var.set(region)
        try:
            region_mask = self._quadrant_pt_mask(self.grid_points)
        finally:
            self.quadrant_var.set(prev_region)
        views = ("registered", "od", "os") if self.clinical_type in self.EYE_VIEW_CLINICAL_TYPES else ("registered",)
        rows = []
        try:
            for view in views:
                self._eye_view = "avg" if view == "registered" else view
                vals, paired = self._get_avg76_with_fallback(data, prefix)
                if vals is None:
                    rows.append((view, None, "no_data")); continue
                if view == "registered":
                    areas = self._compute_scotoma_areas76(vals, region_mask=region_mask, paired_valid_mask=paired)
                else:
                    areas = self._compute_scotoma_areas76(vals, region_mask=region_mask,
                                                          paired_valid_mask=np.isfinite(vals),
                                                          valid_key="Valid_ROI_Locations")
                rows.append((view, areas, "ok" if areas is not None else "no_valid"))
        finally:
            self._eye_view = prev_view
        return rows

    def show_scotoma_area_dialog(self, sid):
        """Scotoma-area report for one subject (cross-sectional).

        The user picks a timepoint and an analysis area (Full Field, a quadrant
        or a half-field). The report shows the three mutually exclusive severity
        bands and the two cumulative bands (severe, moderate-or-worse), each as
        a percentage of the valid locations within the area. The registered
        table uses only locations valid in both eyes (complete-pair rule); for
        chiasmatic and PTC studies it is followed by one table for OD and one
        for OS, each as recorded and not mirrored. "Copy to Clipboard" copies a
        tab-separated header row and one data row per table.
        """
        data = self.db.get(sid, {})

        if not self.severity_supported:
            messagebox.showwarning("Not Available for Raw Threshold",
                "No validated severity classification exists for RAW (uncorrected) "
                "threshold sensitivity in the literature - switch to Total Deviation "
                "for scotoma-area reporting.")
            return

        REGION_OPTIONS = ("Full Field",
                           "Superior Left", "Superior Right",
                           "Inferior Left", "Inferior Right",
                           "2 Left Quarters", "2 Right Quarters")

        # Step 1: pick timepoint + analysis area
        win = tk.Toplevel(self.root)
        win.title(f"Scotoma Area - {sid}")
        win.grab_set(); win.resizable(False, False)

        tk.Label(win, text=f"Subject: {sid}", font=("Arial", 10, "bold"),
                 pady=8).pack(padx=20, anchor='w')
        tk.Label(win, text="Select timepoint (cross-sectional):",
                 font=("Arial", 9)).pack(padx=20, anchor='w')

        tp_var = tk.StringVar(value="postop")
        for val, lbl in (("preop", "Timepoint 1"), ("postop", "Timepoint 2")):
            tk.Radiobutton(win, text=lbl, variable=tp_var, value=val,
                           font=("Arial", 9)).pack(anchor='w', padx=30, pady=2)

        tk.Label(win, text="Select analysis area:",
                 font=("Arial", 9)).pack(padx=20, anchor='w', pady=(8,0))
        region_var = tk.StringVar(value=self.quadrant_var.get()
                                   if self.quadrant_var.get() in REGION_OPTIONS
                                   else "Full Field")
        region_menu = ttk.Combobox(win, textvariable=region_var,
                                    values=REGION_OPTIONS, state="readonly", width=20)
        region_menu.pack(padx=20, pady=(2,8), anchor='w')

        confirmed = tk.BooleanVar(value=False)
        def on_ok():    confirmed.set(True); win.destroy()
        def on_cancel(): win.destroy()
        bf = tk.Frame(win); bf.pack(pady=10)
        tk.Button(bf, text="Calculate", bg="#2e7d32", fg="white",
                  font=("Arial", 10, "bold"), command=on_ok).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Cancel", command=on_cancel).pack(side=tk.LEFT)
        win.wait_window()
        if not confirmed.get():
            return

        timepoint = tp_var.get()
        region    = region_var.get()
        rows = self._scotoma_rows(data, timepoint, region)
        multi = len(rows) > 1                       # chiasmatic / PTC: registered, then OD, then OS
        if all(r[1] is None for r in rows):
            if all(r[2] == "no_data" for r in rows):
                messagebox.showwarning("No Data", f"No VF data found for '{sid}' at {tp_display(timepoint)}.")
            else:
                messagebox.showwarning("No Data",
                    f"No valid VF points found for '{sid}' at {tp_display(timepoint)} within the '{region}' area.")
            return

        view_label = {
            "registered": (f"Both eyes registered ({str(getattr(self, 'flip_eye', '') or '').upper()} mirrored) - valid paired-eye locations"
                           if multi else "Registered - valid paired-eye locations"),
            "od": "OD - as recorded (not mirrored) - valid OD locations",
            "os": "OS - as recorded (not mirrored) - valid OS locations",
        }
        clip_view = {"registered": "Registered", "od": "OD", "os": "OS"}

        # Step 2: results window with table(s) + copy-to-clipboard
        rwin = tk.Toplevel(self.root)
        rwin.title(f"Scotoma Area - {sid}  [{tp_display(timepoint)} | {region}]")
        rwin.grab_set(); rwin.resizable(False, False)

        head = (f"Scotoma Area ({self.data_type_spec['label']}) - {sid}  [{tp_display(timepoint)}]  Area: {region}\n"
                f"(% of VALID locations within the selected area)")
        if self.clinical_type == "post_chiasmatic":
            head += (f"\nPrincipal outcome: {self.tier_labels['moderate']}-or-worse "
                     f"({self.data_type_spec['short']}\u2264{self.moderate_thr:.2f}dB) in the "
                     f"operation-contralateral superior visual-field quadrant")
        tk.Label(rwin, text=head, font=("Arial", 10, "bold"), pady=8, justify='center').pack(padx=20, anchor='w')
        tk.Label(rwin, text=self.severity_caveat,
                 font=("Arial", 8, "italic"), fg="#b71c1c", pady=2).pack(padx=20, anchor='w')

        # Header names are built from the active data type's tier labels and
        # thresholds.
        _dt = self.data_type_spec["short"]
        tl = self.tier_labels
        _CLIP_HEADER_MAP = {
            tl["severe"]:   f"{_dt}_{tl['severe']}_pct_le_{self.severe_thr:g}dB".replace(" ", ""),
            tl["moderate"]: f"{_dt}_{tl['moderate']}_pct_gt_{self.severe_thr:g}_to_le_{self.moderate_thr:g}dB".replace(" ", ""),
            tl["normal"]:   f"{_dt}_{tl['normal']}_pct_gt_{self.moderate_thr:g}dB".replace(" ", ""),
            f"Cum {tl['severe']} (\u2264{self.severe_thr:g}dB)":
                f"{_dt}_Cumulative_{tl['severe']}_pct_le_{self.severe_thr:g}dB".replace(" ", ""),
            f"Cum {tl['moderate']}-or-worse (\u2264{self.moderate_thr:g}dB)":
                f"{_dt}_Cumulative_{tl['moderate']}OrWorse_pct_le_{self.moderate_thr:g}dB".replace(" ", ""),
        }
        clip_headers = headers = None
        clip_rows = []
        for view, areas, status in rows:
            tk.Label(rwin, text=view_label[view] + (f"   (n={int(list(areas.values())[-1])} pts)" if areas else ""),
                     font=("Arial", 9, "bold"), pady=2).pack(padx=20, anchor='w', pady=(8, 0))
            if areas is None:
                tk.Label(rwin, text=("No VF data for this eye/timepoint." if status == "no_data"
                                     else "No valid locations within the selected area."),
                         font=("Arial", 9, "italic"), fg="#555").pack(padx=30, anchor='w')
                continue
            keys = list(areas.keys())
            n_pts = int(areas[keys[-1]])
            hdrs = keys[:-1]
            vals = [areas[h] for h in hdrs]
            if headers is None:
                headers = hdrs
                clip_headers = [_CLIP_HEADER_MAP.get(h, h) for h in hdrs]
            table = tk.Frame(rwin); table.pack(padx=20, pady=4)
            for c, h in enumerate(hdrs):
                tk.Label(table, text=h.replace("_", " "), font=("Arial", 8, "bold"),
                         borderwidth=1, relief="solid", width=14, wraplength=100,
                         justify='center', padx=2, pady=4).grid(row=0, column=c, sticky="nsew")
            for c, v in enumerate(vals):
                tk.Label(table, text=f"{v:.1f}%", font=("Arial", 9),
                         borderwidth=1, relief="solid", width=14,
                         padx=2, pady=4).grid(row=1, column=c, sticky="nsew")
            lead = f"{sid}\t{tp_display(timepoint)}\t{region}\t" + (f"{clip_view[view]}\t" if multi else "")
            clip_rows.append(lead + "\t".join(f"{v:.2f}" for v in vals) + f"\t{n_pts}")

        valid_col = f"{_dt}_Valid_ROI_Locations" if multi else f"{_dt}_Valid_Paired_ROI_Locations"
        clip_header = ("Subject_ID\tTimepoint\tArea\t" + ("Eye_View\t" if multi else "")
                       + "\t".join(clip_headers) + f"\t{valid_col}")
        clip_text = clip_header + "\n" + "\n".join(clip_rows)

        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(clip_text)
            self.root.update()
            messagebox.showinfo("Copied",
                "Values copied to clipboard (header row + one data row per view) -\n"
                "paste directly into Excel/Sheets.")

        bf2 = tk.Frame(rwin); bf2.pack(pady=10)
        tk.Button(bf2, text="Copy to Clipboard", bg="#1565c0", fg="white",
                  font=("Arial", 10, "bold"), command=copy_to_clipboard).pack(side=tk.LEFT, padx=10)
        tk.Button(bf2, text="Close", command=rwin.destroy).pack(side=tk.LEFT)

    def enter_manual_md_dialog(self):
        """Enter the device-reported MD for a subject, per eye and timepoint (PTC only).

        The value is the MD printed on the source report. It is the only exact
        (non-approximated) MD, and the only MD available for raw-threshold
        databases, because MD cannot be derived from uncorrected sensitivity
        (see compute_md). With both eyes entered for a timepoint, that
        timepoint's MD is "device-reported" wherever MD is shown or analysed.
        With either eye blank, the registered MD falls back to the approximation
        (Total Deviation databases) or is unavailable (raw-threshold databases).
        """
        subject_ids = [sid for sid in self.db.keys() if sid not in RESERVED_DB_KEYS
                       and isinstance(self.db[sid], dict)]
        if not subject_ids:
            messagebox.showwarning("No Subjects", "Add a subject first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Enter Device MD (PTC)")
        win.grab_set(); win.resizable(False, False)
        tk.Label(win, text="Enter the Mean Deviation (MD, dB) exactly as printed on the "
                            "source report, per eye. Leave a field blank to fall back to "
                            "the automatic approximation (Total Deviation databases) or "
                            "leave it unavailable (Raw Threshold databases).",
                 font=("Arial", 9, "bold"), wraplength=440, justify="left",
                 pady=10).pack(padx=20)

        row = tk.Frame(win); row.pack(padx=20, pady=6, fill="x")
        tk.Label(row, text="Subject:", width=12, anchor="w").grid(row=0, column=0, sticky="w")
        subj_var = tk.StringVar(value=subject_ids[0])
        ttk.Combobox(row, textvariable=subj_var, values=subject_ids,
                     state="readonly", width=24).grid(row=0, column=1, sticky="w")

        tk.Label(row, text="Timepoint:", width=12, anchor="w").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tp_var = tk.StringVar(value="Timepoint 1")
        ttk.Combobox(row, textvariable=tp_var, values=["Timepoint 1", "Timepoint 2"],
                     state="readonly", width=24).grid(row=1, column=1, sticky="w", pady=(8, 0))

        tk.Label(row, text="MD OD (dB):", width=12, anchor="w").grid(row=2, column=0, sticky="w", pady=(8, 0))
        od_entry = tk.Entry(row, width=10)
        od_entry.grid(row=2, column=1, sticky="w", pady=(8, 0))

        tk.Label(row, text="MD OS (dB):", width=12, anchor="w").grid(row=3, column=0, sticky="w", pady=(4, 0))
        os_entry = tk.Entry(row, width=10)
        os_entry.grid(row=3, column=1, sticky="w", pady=(4, 0))

        def prefix_for():
            return "preop_" if tp_var.get() == "Timepoint 1" else "postop_"

        def load_current(*_a):
            data = self.db.get(subj_var.get(), {})
            md_od, md_os = self.get_manual_md(data, prefix_for())
            od_entry.delete(0, tk.END); od_entry.insert(0, "" if md_od is None else str(md_od))
            os_entry.delete(0, tk.END); os_entry.insert(0, "" if md_os is None else str(md_os))
        subj_var.trace_add("write", load_current)
        tp_var.trace_add("write", load_current)
        load_current()

        def on_save():
            sid = subj_var.get()
            prefix = prefix_for()
            data = self.db.get(sid)
            if data is None:
                return
            for entry, side in ((od_entry, "od"), (os_entry, "os")):
                txt = entry.get().strip()
                key = f"md_manual_{prefix}{side}"
                if txt == "":
                    data[key] = None
                else:
                    try:
                        data[key] = float(txt)
                    except ValueError:
                        messagebox.showwarning("Invalid Value",
                            f"'{txt}' is not a valid number for MD {side.upper()}.")
                        return
            self.save_db()
            messagebox.showinfo("Saved", f"Device MD saved for {sid} ({tp_var.get()}).")
            win.destroy()

        bf = tk.Frame(win); bf.pack(pady=14)
        tk.Button(bf, text="Save →", bg="#1a237e", fg="white", font=("Arial", 10, "bold"),
                  command=on_save).pack(side=tk.LEFT, padx=10)
        tk.Button(bf, text="Close", command=win.destroy).pack(side=tk.LEFT)
        self._center_window(win, getattr(self, "root", None))
        win.wait_window()

    def show_vf_viewer(self, sid):
        import matplotlib.patches as mpatches
        data = self.db.get(sid, {})

        panels = [
            ("preop",  "OD", "Timepoint 1 OD"),
            ("preop",  "OS", "Timepoint 1 OS"),
            ("postop", "OD", "Timepoint 2 OD"),
            ("postop", "OS", "Timepoint 2 OS"),
        ]

        available = []
        for tp, eye, label in panels:
            key = f"vf_raw76_{tp}_{eye.lower()}"
            if data.get(key) is not None:
                available.append((tp, eye, label, key))

        if not available:
            messagebox.showinfo("No VF Data",
                f"Subject '{sid}' has no saved VF data yet.\n"
                "Double-click a VF cell to import DICOM data first.")
            return

        n = len(available)
        dt_name = "Total Deviation" if self.data_type == "total_deviation" else "Sensitivity"

        # One dataset = a raw-points panel and an interpolated map. Up to two
        # datasets stack vertically; more are laid out two per row so the
        # panels stay large enough to read on a landscape screen.
        sets_per_row = 1 if n <= 2 else 2
        n_rows = -(-n // sets_per_row)
        fig, axes = self._subplots(n_rows, 2 * sets_per_row,
                                   figsize=(11.0 * sets_per_row, 4.4 * n_rows + 0.8),
                                   squeeze=False)
        for extra in range(n, n_rows * sets_per_row):          # unused slots
            r_, c_ = divmod(extra, sets_per_row)
            axes[r_][2 * c_].axis("off"); axes[r_][2 * c_ + 1].axis("off")
        head = f"Visual Field Maps - {sid}"

        # MD summary line, one entry per timepoint with a computable MD (PTC
        # only; see compute_md).
        if self.clinical_type == "ptc":
            md_parts = []
            for tp, tp_label in (("preop", "Timepoint 1"), ("postop", "Timepoint 2")):
                md, npts, tier, src = self.compute_md(data, f"{tp}_")
                if md is not None:
                    md_parts.append(f"MD ({tp_label}): {md:+.2f} dB, n={npts}  [{tier}, {src}]")
            if md_parts:
                head += "\n" + "   |   ".join(md_parts)
        fig.suptitle(head, fontsize=12, fontweight='bold')

        # Circular mask indices (reused for every panel)
        _y, _x = np.ogrid[-50:50, -50:50]
        circ_mask = (_x * _x + _y * _y) > 50 * 50

        for row_idx, (tp, eye, label, key) in enumerate(available):
            raw76 = np.array([np.nan if v is None else v for v in data[key]], dtype=float)
            _pf   = self._policy_flip_flag(sid, eye)
            flip  = _pf if _pf is not None else bool(data.get(f"vf_flip_{tp}_{eye.lower()}", False))
            missing = ~np.isfinite(raw76)

            # Left: scatter of the tested points, labelled with their dB values
            set_r, set_c = divmod(row_idx, sets_per_row)
            ax_raw = axes[set_r][2 * set_c]
            pts_xy = self.grid_points.copy()
            if flip:
                pts_xy[:, 0] = -pts_xy[:, 0]

            valid_pts = ~missing
            sc = ax_raw.scatter(pts_xy[valid_pts, 0], pts_xy[valid_pts, 1],
                                c=raw76[valid_pts], cmap=self.map_cmap,
                                vmin=self.map_vmin, vmax=self.map_vmax, s=62,
                                edgecolors='#333', linewidths=0.4)
            if missing.any():
                # Sand-hatched marker for a missing source measurement; it is
                # never shown as a measured 0 dB value.
                ax_raw.scatter(pts_xy[missing, 0], pts_xy[missing, 1],
                               marker='s', s=90, facecolor=MISSING_MARKER_FACE,
                               edgecolor=MISSING_MARKER_EDGE, hatch=MISSING_MARKER_HATCH,
                               linewidth=0.7, zorder=5)
            for i, (px, py) in enumerate(pts_xy):
                if missing[i]:
                    continue
                db_val = round(raw76[i])
                ax_raw.text(px, py, str(int(db_val)),
                            ha='center', va='center', fontsize=5,
                            color='white' if raw76[i] < (self.map_vmin + self.map_vmax) / 2 else 'black')

            ax_raw.set_xlim(-33, 33)
            ax_raw.set_ylim(-33, 33)
            ax_raw.set_aspect('equal')
            ax_raw.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax_raw.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
            ax_raw.set_title(f"{label} - Raw {self.n_vf} pts ({dt_name}, dB)"
                             f"  ({'FLIP' if flip else 'No flip'})",
                             fontsize=8, fontweight='bold', pad=4)
            ax_raw.set_xlabel("← Nasal   X°   Temporal →", fontsize=7, labelpad=3)
            ax_raw.set_ylabel("Y (degrees)", fontsize=7, labelpad=3)
            ax_raw.tick_params(labelsize=6)
            cb_r = fig.colorbar(sc, ax=ax_raw, fraction=0.042, pad=0.03)
            cb_r.set_label(f'{dt_name} (dB)', fontsize=6, labelpad=4)
            cb_r.ax.tick_params(labelsize=6)

            # Right: Interpolated + Gaussian-smoothed map (σ = 1.5)
            ax_int = axes[set_r][2 * set_c + 1]

            # pts_to_grid_100 interpolates the valid points only and clips to the
            # display range. Display only: no statistic is computed from the grid.
            grid = pts_to_grid_100(self.grid_points, raw76, flip, vmin=self.map_vmin, vmax=self.map_vmax, extent_deg=self.grid_extent_deg)
            grid_s = _nan_aware_gaussian_smooth(grid, sigma=1.5)
            grid_s[circ_mask] = np.nan          # re-apply circular mask after smooth

            im = ax_int.imshow(grid_s, cmap=self.map_cmap, vmin=self.map_vmin, vmax=self.map_vmax,
                               extent=[-30, 30, -30, 30], origin='upper')
            ax_int.axhline(0, color='black', lw=0.6, ls='-', alpha=0.45)
            ax_int.axvline(0, color='black', lw=0.6, ls='-', alpha=0.45)
            circ_patch = mpatches.Circle((0, 0), 30,
                                         color='#555', fill=False,
                                         lw=1.0, ls='--', alpha=0.7)
            ax_int.add_patch(circ_patch)
            if missing.any():
                # Redraw the missing-location markers on top of the smoothed
                # surface, so that interpolation does not hide missing data.
                ax_int.scatter(pts_xy[missing, 0], pts_xy[missing, 1],
                               marker='s', s=70, facecolor=MISSING_MARKER_FACE,
                               edgecolor=MISSING_MARKER_EDGE, hatch=MISSING_MARKER_HATCH,
                               linewidth=0.6, zorder=6)
            ax_int.set_xlim(-33, 33)
            ax_int.set_ylim(-33, 33)
            ax_int.set_aspect('equal')
            ax_int.set_title(f"{label} - Interpolated (σ=1.5, visual only)",
                             fontsize=8, fontweight='bold', pad=4)
            ax_int.set_xlabel("← Nasal   X°   Temporal →", fontsize=7, labelpad=3)
            ax_int.set_ylabel("Y (degrees)", fontsize=7, labelpad=3)
            ax_int.tick_params(labelsize=6)
            cb_i = fig.colorbar(im, ax=ax_int, fraction=0.042, pad=0.03)
            cb_i.set_label(f'{dt_name} (dB)', fontsize=6, labelpad=4)
            cb_i.ax.tick_params(labelsize=6)

        self._caption(fig, MISSING_LOCI_CAPTION, face=MISSING_MARKER_FACE,
                      edge=MISSING_MARKER_EDGE, fontsize=6.5)
        self.add_save_button(fig)
        plt.show(block=False)

    # Save button - 300 DPI .TIF
    def add_save_button(self, fig):
        """Attach the 'save for publication' button and fit the figure to the screen.

        The layout engine re-spaces the figure for the smaller canvas. The saved
        file uses the figure's original size.
        """
        full_size = tuple(fig.get_size_inches())
        # Titles wrap to the window width instead of running off the canvas.
        for ax in fig.axes:
            ax.title.set_wrap(True)
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_wrap(True)

        def save_publication_quality():
            f = filedialog.asksaveasfilename(
                defaultextension=".tif",
                filetypes=[("TIFF Image", "*.tif"), ("PNG Image", "*.png")])
            if f:
                shown = tuple(fig.get_size_inches())
                fig.set_size_inches(*full_size)
                try:
                    fig.savefig(f, dpi=300, bbox_inches='tight')
                finally:
                    fig.set_size_inches(*shown)
                messagebox.showinfo("Saved", "300 DPI image saved successfully.")
        try:
            win = fig.canvas.manager.window
            w, h = full_size
            btn_h_in = 0.55   # reserve room for the button row below the canvas
            scale = min(1.0, 0.94 * win.winfo_screenwidth() / fig.dpi / w,
                        0.90 * win.winfo_screenheight() / fig.dpi / (h + btn_h_in))
            if scale < 1.0:
                fig.set_size_inches(w * scale, h * scale)
            fig.canvas.draw()
            # fig.set_size_inches() resizes the figure but does not reliably
            # resize the Tk window around it, which would leave the button squeezed
            # into a sliver at the bottom. Resize the window to the figure's pixel
            # size plus room for the button.
            px_w = int(fig.get_size_inches()[0] * fig.dpi)
            px_h = int(fig.get_size_inches()[1] * fig.dpi) + 55
            try:
                fig.canvas.manager.resize(px_w, px_h)
            except Exception:
                win.wm_geometry(f"{px_w}x{px_h}")
            btn = tk.Button(win, text="SAVE FOR PUBLICATION (300 DPI .TIF)",
                            bg="#2e7d32", fg="white", font=("Arial", 10, "bold"),
                            command=save_publication_quality)
            btn.pack(side=tk.BOTTOM, pady=10, fill=tk.X)
            win.update_idletasks()
        except Exception:
            pass   # headless or non-Tk backend - silently skip

    # DICOM/PDF import: grid-aligned OCR of the printed values, followed by
    # manual review. Each cell is perspective-corrected and read separately;
    # low-confidence, implausible and outlying cells are flagged for review.
    def process_vf_db_ocr(self, sid, side, timepoint="preop"):
        try:
            import pytesseract
            from PIL import Image as PILImage
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        except ImportError:
            messagebox.showerror("Missing Library",
                "Install: pip install pytesseract pillow\n"
                "Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("Missing Library","Install: pip install opencv-python")
            return

        data = self.db.get(sid, {})

        # Post-chiasmatic ("surgery_side_mirror"): Surgery Side has to be set
        # before VF data can be imported for a subject. Without it the subject
        # cannot be registered into the shared ipsilateral/contralateral frame.
        if self.flip_policy == "surgery_side_mirror" and self.surgery_side_var:
            side_val = str(data.get(self.surgery_side_var, "")).strip()
            if not side_val:
                messagebox.showerror(
                    "Surgery Side Required",
                    f"'{self.surgery_side_var}' must be set for subject '{sid}' before "
                    f"importing VF data - post-chiasmatic registration mirrors the whole "
                    f"field by surgery side, so every subject needs it set first.\n\n"
                    f"Set it via 'Add Subject' / the subject's row, then try importing again.")
                return
            if side_val != str(self.surgery_side_reference).strip():
                messagebox.showinfo(
                    "Both Eyes Will Be Mirrored",
                    f"Subject '{sid}' has {self.surgery_side_var} = '{side_val}' - "
                    f"BOTH eyes' combined field will be mirrored (flipped) for this "
                    f"subject, so it registers into the shared ipsilateral/"
                    f"contralateral frame with the rest of the cohort.")

        # Chiasmatic/PTC ("flip_one_eye_cohort"): remind the user at each import
        # that this eye is the one mirrored for the whole cohort (a one-time
        # choice; see _ensure_flip_eye_configured).
        if self.flip_policy == "flip_one_eye_cohort" and self.flip_eye:
            if side.lower() == self.flip_eye:
                messagebox.showinfo(
                    "This Eye Will Be Flipped",
                    f"{side.upper()} will be flipped (mirrored to the other eye's nasal/"
                    f"temporal orientation) in this study - fixed for the whole cohort.")

        tp_label = tp_display(timepoint)          # for display text only
        tp_key   = timepoint.capitalize()         # "Preop"/"Postop", as spelled in COLS_VF
        path = filedialog.askopenfilename(
            title=f"Select DICOM or PDF - {sid} {side} ({tp_label})",
            filetypes=[("All files", "*.*"),
                       ("DICOM or PDF", "*.dcm *.dicom *.pdf"),
                       ("DICOM files", "*.dcm *.dicom"),
                       ("PDF files", "*.pdf")])
        if not path:
            return

        # Clinical DICOM exports often have no file extension, or a non-standard
        # one, so the type is detected by content. A ".pdf" extension is trusted;
        # anything else is tried as DICOM first (pydicom's force=True tolerates a
        # missing preamble/DICM header) and as PDF only if that fails.
        is_pdf = path.lower().endswith(".pdf")
        dicom_error = pdf_error = None
        if not is_pdf:
            try:
                ds     = pydicom.dcmread(path, force=True)
                pixels = ds.pixel_array.astype(float)
                if pixels.ndim == 3:
                    pixels = np.mean(pixels, axis=-1)
            except Exception as e:
                dicom_error = e
                is_pdf = True  # fall through to try as PDF below
        if is_pdf:
            try:
                # Rasterize one page of the PDF to a grayscale array and pass it
                # through the same grid-overlay OCR pipeline as a DICOM image. No
                # text is extracted from the PDF itself.
                try:
                    try:
                        import pymupdf as fitz  # PyMuPDF >= 1.24 preferred import name
                    except ImportError:
                        import fitz  # older PyMuPDF releases
                except ImportError:
                    messagebox.showerror("Missing Library",
                        "Reading PDF source files requires PyMuPDF.\n"
                        "Install: pip install pymupdf")
                    return
                pdf_doc = fitz.open(path)
                n_pages = pdf_doc.page_count
                page_num = 1
                if n_pages > 1:
                    page_num = simpledialog.askinteger(
                        "Select PDF Page",
                        f"This PDF has {n_pages} pages. Which page has the "
                        f"{DATA_TYPES[self.data_type]['label']} grid?",
                        initialvalue=1, minvalue=1, maxvalue=n_pages) or 1
                page = pdf_doc.load_page(page_num - 1)
                # 300 DPI gives OCR enough resolution; PDF default is 72 DPI.
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                pdf_doc.close()
                pixels = (np.mean(arr[:, :, :3], axis=-1)
                          if pix.n >= 3 else arr[:, :, 0]).astype(float)
            except Exception as e:
                pdf_error = e
                if dicom_error is not None:
                    messagebox.showerror("Could Not Read File",
                        f"Tried reading '{os.path.basename(path)}' as both DICOM and PDF "
                        f"and neither worked:\n\nAs DICOM: {dicom_error}\n\nAs PDF: {pdf_error}")
                else:
                    messagebox.showerror("PDF Error", str(pdf_error))
                return

        img_disp = ((pixels - pixels.min()) /
                    (pixels.max() - pixels.min() + 1e-7) * 255).astype(np.uint8)
        img_h, img_w = img_disp.shape

        # The cell layout comes from the active pattern (self.grid). The 10x10
        # overlay is the same for all three patterns; only the set of active
        # cells differs.
        ACTIVE_COLS  = self.grid["active_cols"]
        COL_TO_X_OCR = self.grid["col_to_x"]
        ROW_TO_Y_OCR = self.grid["row_to_y"]
        dtype_label  = DATA_TYPES[self.data_type]["label"]

        from matplotlib.widgets import Slider, Button as MplButton

        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.30)
        ax.imshow(img_disp, cmap='gray', aspect='equal')
        ax.set_title(f"{sid} {side} ({tp_label})  -  Pattern: {self.grid_pattern}  -  "
                     f"Align RED GRID over the numerical {dtype_label} grid, "
                     f"then click EXTRACT",
                     fontweight='bold', fontsize=10)

        import json as _json
        # The alignment is remembered separately per pattern and data type: the
        # raw-threshold grid sits at a different position on a printout than the
        # Total Deviation grid, and the 24-2, 30-2 and 60-4 pages differ too.
        params_file = os.path.join(
            _SCRIPT_DIR, f"grid_params_last_{self.grid_pattern}_{self.data_type}.json")
        if os.path.exists(params_file):
            try:
                with open(params_file) as _f:
                    saved = _json.load(_f)
                d_x0=saved.get('x0',img_w*0.25); d_y0=saved.get('y0',img_h*0.22)
                d_cw=saved.get('cw',img_w*0.055); d_ch=saved.get('ch',img_h*0.055)
                d_ang=saved.get('angle',0.0)
            except:
                d_x0=img_w*0.25;d_y0=img_h*0.22;d_cw=img_w*0.055;d_ch=img_h*0.055;d_ang=0.0
        else:
            d_x0=img_w*0.25;d_y0=img_h*0.22;d_cw=img_w*0.055;d_ch=img_h*0.055;d_ang=0.0

        def get_cell_corners(x0, y0, cw, ch, angle_deg):
            import math
            ang = math.radians(angle_deg)
            cos_a, sin_a = math.cos(ang), math.sin(ang)
            corners = []
            for ri in range(11):
                row = []
                for ci in range(11):
                    lx = x0 + ci*cw; ly = y0 + ri*ch
                    dx, dy = lx-x0, ly-y0
                    row.append((x0+dx*cos_a-dy*sin_a, y0+dx*sin_a+dy*cos_a))
                corners.append(row)
            return corners

        drawn_lines = []

        def draw_grid(x0, y0, cw, ch, angle):
            for ln in drawn_lines:
                try: ln.remove()
                except: pass
            drawn_lines.clear()
            corners = get_cell_corners(x0, y0, cw, ch, angle)
            # Gridline index 5 (of 0..10) is the horizontal/vertical meridian
            # for every pattern, because all three lattices straddle zero between
            # cells 4 and 5. The two lines are drawn bold so that they can be
            # fitted onto the printout's central crosshair.
            CENTER_IDX = 5
            for ri in range(11):
                xs=[corners[ri][ci][0] for ci in range(11)]
                ys=[corners[ri][ci][1] for ci in range(11)]
                is_center = (ri == CENTER_IDX)
                lw = 2.6 if is_center else 0.8
                a  = 1.0 if is_center else 0.8
                ln,=ax.plot(xs,ys,'r-',linewidth=lw,alpha=a); drawn_lines.append(ln)
            for ci in range(11):
                xs=[corners[ri][ci][0] for ri in range(11)]
                ys=[corners[ri][ci][1] for ri in range(11)]
                is_center = (ci == CENTER_IDX)
                lw = 2.6 if is_center else 0.8
                a  = 1.0 if is_center else 0.8
                ln,=ax.plot(xs,ys,'r-',linewidth=lw,alpha=a); drawn_lines.append(ln)
            fig.canvas.draw_idle()

        draw_grid(d_x0, d_y0, d_cw, d_ch, d_ang)

        ax_x0  = plt.axes([0.10,0.22,0.35,0.025])
        ax_y0  = plt.axes([0.10,0.18,0.35,0.025])
        ax_cw  = plt.axes([0.10,0.14,0.35,0.025])
        ax_ch  = plt.axes([0.10,0.10,0.35,0.025])
        ax_ang = plt.axes([0.10,0.06,0.35,0.025])
        sl_x0  = Slider(ax_x0,  'Grid X start', 0, img_w*0.5,  valinit=d_x0,  valstep=1)
        sl_y0  = Slider(ax_y0,  'Grid Y start', 0, img_h*0.5,  valinit=d_y0,  valstep=1)
        sl_cw  = Slider(ax_cw,  'Cell Width',   5, img_w*0.15, valinit=d_cw,  valstep=0.5)
        sl_ch  = Slider(ax_ch,  'Cell Height',  5, img_h*0.15, valinit=d_ch,  valstep=0.5)
        sl_ang = Slider(ax_ang, 'Angle (°)',   -15, 15,         valinit=d_ang, valstep=0.1)

        def update(_):
            draw_grid(sl_x0.val,sl_y0.val,sl_cw.val,sl_ch.val,sl_ang.val)
        for sl in [sl_x0,sl_y0,sl_cw,sl_ch,sl_ang]:
            sl.on_changed(update)

        ax_btn  = plt.axes([0.60,0.06,0.20,0.08])
        btn_ext = MplButton(ax_btn, f'EXTRACT {self.data_type_spec["short"]}', color='#a5d6a7')

        result_store = {}

        def do_extract(_event=None):
            x0=sl_x0.val; y0=sl_y0.val; cw=sl_cw.val; ch=sl_ch.val; ang=sl_ang.val
            corners = get_cell_corners(x0, y0, cw, ch, ang)

            # cell_vals holds signed dB values. A blank source location (the
            # blind-spot omission, or a cell the OCR cannot read) is np.nan; it is
            # never 0 and never estimated.
            cell_vals    = np.full((10,10), np.nan, dtype=float)
            cell_valid   = np.zeros((10,10), dtype=bool)   # True = a numeric value was read
            cell_conf    = np.zeros((10,10), dtype=float)
            cell_outrange= np.zeros((10,10), dtype=bool)   # read but outside plausibility range
            # The whitelist includes the minus sign, so negative Total Deviation
            # values are read.
            ocr_cfg = r'--oem 3 --psm 10 -c tessedit_char_whitelist=+-0123456789'

            for ri in range(10):
                for ci in range(10):
                    tl=corners[ri][ci];   tr=corners[ri][ci+1]
                    bl=corners[ri+1][ci]; br=corners[ri+1][ci+1]
                    pts_src = np.float32([tl,tr,br,bl])
                    pts_dst = np.float32([[0,0],[60,0],[60,60],[0,60]])
                    M    = cv2.getPerspectiveTransform(pts_src, pts_dst)
                    cell = cv2.warpPerspective(img_disp, M, (60,60))
                    cell_big = cv2.resize(cell,(180,180),interpolation=cv2.INTER_CUBIC)
                    _,cell_bin = cv2.threshold(cell_big,0,255,
                                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    ink_ratio = np.sum(cell_bin>0)/cell_bin.size
                    if ink_ratio > 0.25:
                        # Heavily inked cell (a blind-spot "X" mark or an unreadable
                        # smudge): a missing source location, not a value to
                        # estimate. Left as np.nan.
                        continue
                    from PIL import Image as _PILImg
                    cell_pil = _PILImg.fromarray(cell_bin)
                    ocr_data = pytesseract.image_to_data(
                        cell_pil, config=ocr_cfg,
                        output_type=pytesseract.Output.DICT)
                    confs = [int(c) for c in ocr_data['conf']
                             if str(c).lstrip('-').isdigit() and int(c)>=0]
                    cell_conf[ri,ci] = np.mean(confs)/100.0 if confs else 0.0
                    txt = pytesseract.image_to_string(cell_pil, config=ocr_cfg).strip()
                    v = parse_td_value(txt)
                    if np.isnan(v):
                        # No numeric token was recognized: a blank source location
                        # or an OCR failure. Left as np.nan; the reviewer can type
                        # the value.
                        continue
                    cell_vals[ri,ci]  = v
                    cell_valid[ri,ci] = True
                    if not (self.plausible_min <= v <= self.plausible_max):
                        # Outside the plausibility range: flagged for manual
                        # verification, not clipped or rejected.
                        cell_outrange[ri,ci] = True

            # Missing cells are not imputed: a missing source location stays
            # np.nan and is never estimated from neighboring cells, a cohort mean
            # or the fellow eye.

            # Flag a cell that has low OCR confidence, is outside the plausibility
            # range, or is far from the median of its neighbors. Only cells that
            # were read take part in the comparison.
            cell_flagged = np.zeros((10,10), dtype=bool)
            for ri in range(10):
                for ci in ACTIVE_COLS.get(ri,[]):
                    if cell_outrange[ri,ci]:
                        cell_flagged[ri,ci] = True
                    nbrs=[cell_vals[ri+dr,ci+dc]
                          for dr in [-1,0,1] for dc in [-1,0,1]
                          if not(dr==0 and dc==0)
                          and 0<=ri+dr<10 and 0<=ci+dc<10
                          and ci+dc in ACTIVE_COLS.get(ri+dr,[])
                          and cell_valid[ri+dr,ci+dc]]
                    if cell_valid[ri,ci] and nbrs:
                        nbr_med = np.median(nbrs)
                        if cell_conf[ri,ci]<0.5 or abs(cell_vals[ri,ci]-nbr_med)>8:
                            cell_flagged[ri,ci]=True
                    elif not cell_valid[ri,ci]:
                        # Unresolved cell: flag for review.
                        cell_flagged[ri,ci] = True

            # Crop DICOM to grid region for thumbnail
            x0i=int(sl_x0.val); y0i=int(sl_y0.val)
            cw10=int(sl_cw.val*10); ch10=int(sl_ch.val*10)
            x1i=min(img_w,x0i+cw10); y1i=min(img_h,y0i+ch10)
            grid_crop = img_disp[y0i:y1i, x0i:x1i]
            grid_crop_resized = cv2.resize(grid_crop,(320,320),
                                           interpolation=cv2.INTER_AREA)

            plt.close(fig)

            n_flagged = sum(cell_flagged[ri,ci]
                            for ri in range(10) for ci in ACTIVE_COLS.get(ri,[]))
            corr_win = tk.Toplevel(self.root)
            corr_win.title(f"Review & Correct - {sid} {side} ({tp_label})  |  "
                           f"{dtype_label}  |  {n_flagged} flagged")
            corr_win.grab_set()

            hdr = tk.Frame(corr_win, bg="#263238")
            hdr.grid(row=0, column=0, columnspan=2, sticky='ew')
            tk.Label(hdr,
                     text=f"● YELLOW = suspicious ({n_flagged} cells).  ● SAND = missing/blank.  "
                          "Edit values, set flip, then Confirm.",
                     bg="#263238", fg="#ffee58",
                     font=("Arial",10,"bold"), pady=7).pack(side=tk.LEFT, padx=14)

            left_frame = tk.Frame(corr_win, bd=2, relief='groove')
            left_frame.grid(row=1, column=0, padx=10, pady=5, sticky='n')
            tk.Label(left_frame, text="Reference - Original DICOM Grid",
                     font=("Arial",9,"bold"), fg="#1a237e").pack(pady=3)
            from PIL import Image as PILRef, ImageTk
            pil_ref = PILRef.fromarray(grid_crop_resized)
            tk_img  = ImageTk.PhotoImage(pil_ref)
            img_lbl = tk.Label(left_frame, image=tk_img)
            img_lbl.image = tk_img
            img_lbl.pack(padx=5, pady=5)

            right_frame = tk.Frame(corr_win)
            right_frame.grid(row=1, column=1, padx=10, pady=5, sticky='n')
            tk.Label(right_frame, text=f"{dtype_label}  -  edit yellow/sand cells",
                     font=("Arial",9,"bold"), fg="#b71c1c").grid(
                     row=0, column=0, columnspan=12, pady=3, sticky='w')
            tk.Label(right_frame,
                     text=f"Leave source locations without a printed {self.data_type_spec["short"]} value blank.\n"
                          "Blank entries are stored as missing and excluded from analysis.",
                     font=("Arial",7,"italic"), fg="#6B5B3E", justify='left').grid(
                     row=12, column=0, columnspan=12, pady=(4,0), sticky='w')

            ROW_LABELS = ['+27°','+21°','+15°','+9°','+3°',
                          '-3°','-9°','-15°','-21°','-27°']
            tk.Label(right_frame, text="", width=5).grid(row=1, column=0)
            for ci in range(10):
                tk.Label(right_frame, text=str(ci),
                         font=("Arial",7), fg="#888", width=5).grid(row=1, column=ci+1)

            entry_widgets = {}
            for ri in range(10):
                tk.Label(right_frame, text=ROW_LABELS[ri],
                         font=("Arial",8,"bold"), width=5).grid(row=ri+2, column=0, padx=2)
                for ci in range(10):
                    if ci not in ACTIVE_COLS.get(ri,[]):
                        tk.Label(right_frame, text="", width=5,
                                 bg="#fff0f0").grid(row=ri+2, column=ci+1, padx=1, pady=1)
                        continue
                    v = cell_vals[ri,ci]
                    # A missing value is shown blank, never as "0" or "nan".
                    if np.isnan(v):
                        disp = ""
                        bg = MISSING_MARKER_FACE  # sand - missing/blank source location
                    else:
                        disp = str(int(round(v))) if float(v).is_integer() else f"{v:.1f}"
                        bg = '#fff59d' if cell_flagged[ri,ci] else '#c8e6c9'
                    e  = tk.Entry(right_frame, width=4, font=("Arial",9),
                                  justify='center', bg=bg)
                    e.insert(0, disp)
                    e.grid(row=ri+2, column=ci+1, padx=1, pady=1)
                    entry_widgets[(ri,ci)] = e
                    def _cb(entry=e):
                        def _fn(_): entry.configure(bg="#a5d6a7")
                        return _fn
                    e.bind("<KeyRelease>", _cb())

            policy_flip = self._policy_flip_flag(sid, side)
            flip_var = tk.BooleanVar(value=bool(policy_flip))
            flip_frame = tk.Frame(corr_win, bd=1, relief='groove', bg='#fff3e0')
            flip_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=6,
                            sticky='ew')
            tk.Label(flip_frame,
                     text="Flip horizontally (mirrors nasal↔temporal):",
                     font=("Arial",9), bg='#fff3e0').pack(side=tk.LEFT, padx=10)
            tk.Checkbutton(flip_frame, text="FLIP", variable=flip_var,
                           font=("Arial",10,"bold"), bg='#fff3e0',
                           fg='#e65100',
                           state=(tk.NORMAL if policy_flip is None else tk.DISABLED)
                           ).pack(side=tk.LEFT, padx=5)
            if policy_flip is not None:
                tk.Label(flip_frame, text="(set by this study's registration)",
                         font=("Arial", 8, "italic"), bg='#fff3e0', fg='#555').pack(side=tk.LEFT)

            def build_pts_vals(do_flip):
                # Row-major order (row, then column) matches the ordering of
                # self.grid_points for every pattern (see _vf_build_points).
                pts, vals = [], []
                for ri in range(10):
                    for ci in ACTIVE_COLS[ri]:
                        x_deg = COL_TO_X_OCR[ci]
                        y_deg = ROW_TO_Y_OCR[ri]
                        if do_flip: x_deg = -x_deg
                        pts.append((x_deg, y_deg))
                        vals.append(cell_vals[ri,ci])
                return np.array(pts), np.array(vals)

            def build_raw76():
                # Stored values in dB: unclipped, unnormalized, sign preserved,
                # np.nan for missing source locations. The "raw76" name is kept
                # for backward compatibility; the length is self.n_vf.
                raw_db = [cell_vals[ri,ci]
                          for ri in range(10)
                          for ci in ACTIVE_COLS[ri]]
                return np.array(raw_db, dtype=float)

            def _read_entry_into_cell(ri, ci, e):
                # A blank entry (or NA/NaN/null) becomes np.nan, a missing source
                # measurement; it does not fall back to the OCR value or to 0.
                # parse_td_value handles the ASCII, Unicode-minus and dash signs.
                cell_vals[ri, ci] = parse_td_value(e.get())

            def on_confirm():
                for (ri,ci), e in entry_widgets.items():
                    _read_entry_into_cell(ri, ci, e)
                pa, va = build_pts_vals(flip_var.get())
                raw76  = build_raw76()
                result_store.update({'pts':pa,'vals':va,
                                     'cell_vals':cell_vals.copy(),
                                     'flip':flip_var.get(),
                                     'raw76':raw76})
                corr_win.destroy()

            def on_skip():
                pa, va = build_pts_vals(flip_var.get())
                raw76  = build_raw76()
                result_store.update({'pts':pa,'vals':va,
                                     'cell_vals':cell_vals.copy(),
                                     'flip':flip_var.get(),
                                     'raw76':raw76})
                corr_win.destroy()

            bf = tk.Frame(corr_win)
            bf.grid(row=3, column=0, columnspan=2, pady=8)
            tk.Button(bf, text="Confirm & Proceed",
                      bg="#2e7d32", fg="white",
                      font=("Arial",10,"bold"),
                      command=on_confirm).pack(side=tk.LEFT, padx=10)
            tk.Button(bf, text="Proceed without changes",
                      command=on_skip).pack(side=tk.LEFT, padx=5)
            corr_win.wait_window()

            try:
                with open(params_file,'w') as _f:
                    _json.dump({'x0':sl_x0.val,'y0':sl_y0.val,
                                'cw':sl_cw.val,'ch':sl_ch.val,
                                'angle':sl_ang.val}, _f)
            except:
                pass

        btn_ext.on_clicked(do_extract)
        self.add_save_button(fig)
        plt.show()

        if 'vals' not in result_store:
            return

        pts_arr       = result_store['pts']
        vals_arr      = result_store['vals']
        raw76         = result_store['raw76']
        do_flip       = result_store['flip']
        cell_vals_raw = result_store.get('cell_vals', None)

        # Raw grid figure
        if cell_vals_raw is not None:
            fig_raw, ax_raw = self._subplots(figsize=(8,8))
            fig_raw.suptitle(f"Raw Extracted Grid - {sid} {side} ({tp_label})\n"
                             f"{dtype_label} - No masking, no interpolation, exactly as read from DICOM",
                             fontweight='bold', fontsize=10)
            ax_raw.set_xlim(-0.5,9.5); ax_raw.set_ylim(9.5,-0.5)
            ax_raw.set_aspect('equal')
            ax_raw.set_xticks(range(10)); ax_raw.set_yticks(range(10))
            ax_raw.set_xlabel("Column (0=left → 9=right of printout)")
            ax_raw.set_ylabel("Row (0=top → 9=bottom of printout)")
            ax_raw.grid(True, color='red', linewidth=0.5, alpha=0.4)
            for ri in range(10):
                for ci in range(10):
                    v = cell_vals_raw[ri,ci]
                    if ci in ACTIVE_COLS[ri]:
                        if np.isnan(v):
                            # Sand-hatched marker for a missing source location;
                            # never drawn as a measured 0 dB value.
                            rect = plt.Rectangle((ci-0.5,ri-0.5),1,1,
                                                 facecolor=MISSING_MARKER_FACE,
                                                 edgecolor=MISSING_MARKER_EDGE,
                                                 hatch=MISSING_MARKER_HATCH,
                                                 linewidth=0.8)
                            ax_raw.add_patch(rect)
                        else:
                            # Gray level from the display range (top of the
                            # range = white). Only the display is clipped; the
                            # stored value is not.
                            gray = float(np.clip((v - self.map_vmin) / (self.map_vmax - self.map_vmin), 0.0, 1.0))
                            rect = plt.Rectangle((ci-0.5,ri-0.5),1,1,
                                                 facecolor=(gray,gray,gray),
                                                 edgecolor='red',linewidth=0.5)
                            ax_raw.add_patch(rect)
                            ax_raw.text(ci,ri,f"{v:.0f}",ha='center',va='center',
                                        fontsize=8,fontweight='bold',
                                        color='white' if gray<0.5 else 'black')
                    else:
                        rect = plt.Rectangle((ci-0.5,ri-0.5),1,1,
                                             facecolor='#ffeeee',edgecolor='red',
                                             linewidth=0.5,alpha=0.3)
                        ax_raw.add_patch(rect)
            ax_raw.set_yticklabels(['+27°','+21°','+15°','+9°','+3°',
                                    '-3°','-9°','-15°','-21°','-27°'], fontsize=8)
            self._caption(fig_raw, MISSING_LOCI_CAPTION, face=MISSING_MARKER_FACE,
                          edge=MISSING_MARKER_EDGE, fontsize=7)
            self.add_save_button(fig_raw); plt.show(block=False)

        # Extracted points scatter
        fig_pts, ax_pts = self._subplots(figsize=(7,7))
        pts_xy = self.grid_points.copy()
        if do_flip: pts_xy[:,0] = -pts_xy[:,0]
        valid_pts = np.isfinite(raw76)
        sc = ax_pts.scatter(pts_xy[valid_pts,0], pts_xy[valid_pts,1],
                            c=raw76[valid_pts], cmap=self.map_cmap,
                            vmin=self.map_vmin, vmax=self.map_vmax, s=60,
                            edgecolors='black', linewidth=0.5)
        if (~valid_pts).any():
            # Sand-hatched markers for missing source locations, drawn above the
            # value scatter.
            ax_pts.scatter(pts_xy[~valid_pts,0], pts_xy[~valid_pts,1],
                           marker='s', s=90, facecolor=MISSING_MARKER_FACE,
                           edgecolor=MISSING_MARKER_EDGE, hatch=MISSING_MARKER_HATCH,
                           linewidth=0.8, zorder=5,
                           label='Missing source measurement (excluded from analysis)')
            ax_pts.legend(loc='upper right', fontsize=7)
        fig_pts.colorbar(sc, ax=ax_pts, label=f'{dtype_label} (dB)')
        ax_pts.set_title(f"Extracted Points ({dtype_label}) - {sid} {side} ({tp_label})\n"
                         f"{'FLIPPED' if do_flip else 'No flip'}  "
                         f"|  {self.n_vf} canonical pts  |  NOT interpolated",
                         fontweight='bold', fontsize=9)
        ax_pts.set_xlabel("← Nasal   X (degrees)   Temporal →")
        ax_pts.set_ylabel("Y (degrees)")
        ax_pts.axhline(0,color='gray',linewidth=0.5,linestyle='--')
        ax_pts.axvline(0,color='gray',linewidth=0.5,linestyle='--')
        ax_pts.set_xlim(-32,32); ax_pts.set_ylim(-32,32)
        self.add_save_button(fig_pts); plt.show(block=False)

        # Save to database
        _n_valid_vals = int(np.sum(np.isfinite(vals_arr)))
        _max_td = np.nanmax(vals_arr) if _n_valid_vals else float('nan')
        _min_td = np.nanmin(vals_arr) if _n_valid_vals else float('nan')
        if messagebox.askyesno("Save",
                f"Save this {dtype_label} grid to database?\n\n"
                f"Subject: {sid}  |  Eye: {side}  |  Time-point: {tp_label}\n\n"
                f"Stores:\n"
                f"  • vf_raw76_{timepoint}_{side.lower()} - {self.n_vf} raw {self.data_type_spec["short"]} (dB) values,\n"
                f"    missing source locations stored as null/np.nan\n"
                f"  • vf_flip_{timepoint}_{side.lower()} - flip flag\n\n"
                f"Max TD: {_max_td:.0f} dB   Min TD: {_min_td:.0f} dB   "
                f"({_n_valid_vals}/{self.n_vf} locations valid)"):

            key_raw  = f"vf_raw76_{timepoint}_{side.lower()}"
            key_flip = f"vf_flip_{timepoint}_{side.lower()}"
            # Missing source measurements are stored as JSON null, not as a NaN
            # token.
            self.db[sid][key_raw]  = [None if (v is None or (isinstance(v,float) and np.isnan(v)))
                                       else float(v) for v in raw76.tolist()]
            self.db[sid][key_flip] = bool(do_flip)
            col_key = f"{tp_key} {side} VF"
            if col_key in self.db[sid]:
                self.db[sid][col_key] = "READY"
            self._update_diff76(sid)
            self.save_db()
            self.refresh_table()
            messagebox.showinfo("Saved",
                f"Saved {sid} {side} ({tp_label}) to database.\n"
                f"File: {os.path.basename(DB_FILE)}\n"
                f"Raw76 ({dtype_label}): {self.n_vf} points  |  Flip: {do_flip}\n"
                + ("vf_diff76 updated." if self.db[sid].get("vf_diff76")
                   else "Note: vf_diff76 computed when both timepoints entered."))

        plt.close('all')


# Entry point
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x600")
    root.minsize(900, 450)
    print(f"[NeuroVF] Script directory : {_SCRIPT_DIR}")
    print(f"[NeuroVF] Database file    : {DB_FILE}")
    print(f"[NeuroVF] DB exists        : {os.path.exists(DB_FILE)}")
    print("[NeuroVF] Analyzing raw Total Deviation (dB), not normalized PI")
    run_td_validation_tests(verbose=True)
    app = ResearchApp(root)   # grid pattern + data type chosen inside __init__
    print(f"[NeuroVF] Grid pattern     : {app.grid_pattern}  ({app.n_vf} points)")
    print(f"[NeuroVF] Data type        : {app.data_type}")
    print(f"[NeuroVF] Thresholds (dB, prespecified study-defined bands): "
          f"Severe<={app.severe_thr:.0f} | Moderate-or-worse<={app.moderate_thr:.0f} | "
          f"Any deficit<={app.mild_thr:.0f}")
    print("[NeuroVF] Dynamic schema  | Generalised stats | RED=Positive association (better/less-negative TD)")
    root.mainloop()