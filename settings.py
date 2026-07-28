import os
from pathlib import Path
from typing import Union

try:
    import lasair_configs as _lasair_configs
except ImportError:
    _lasair_configs = None


def _lasair_setting(name, env_var):
    value = os.environ.get(env_var)
    if value:
        return value
    if _lasair_configs is not None:
        return getattr(_lasair_configs, name, None)
    return None


def _lasair_bool(name, env_var, default=False):
    value = os.environ.get(env_var)
    if value is not None:
        return value.lower() in ('1', 'true', 'yes')
    if _lasair_configs is not None:
        return bool(getattr(_lasair_configs, name, default))
    return default


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(PROJECT_ROOT)
PACKAGE_ROOT = REPO_ROOT / "source"
DATA_DIR = PACKAGE_ROOT / "data"
INFO_DIR = DATA_DIR / "info"


def _resolve_external_data_path() -> Path:
    env_path = os.environ.get("NEEDLE_DATA_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (REPO_ROOT.parent / "needle-data").resolve()


def upsampled_dir(version: str) -> Path:
    """Return path to a bundled or user-generated upsampled dataset folder."""
    return DATA_DIR / version


def pjoin(base: Union[str, Path], *parts: Union[str, Path]) -> Path:
    """Join path segments safely (works with str or Path)."""
    return Path(base).joinpath(*parts)


# ---------------------------------------------------------------------------
# Lasair streaming
# ---------------------------------------------------------------------------

API_TOKEN = _lasair_setting('API_TOKEN', 'LASAIR_API_TOKEN')
LASAIR_ENDPOINT = _lasair_setting(
    'LASAIR_ENDPOINT',
    'LASAIR_ENDPOINT',
) or 'https://lasair-ztf.lsst.ac.uk/api'
TOPIC_IN = _lasair_setting('TOPIC_IN', 'LASAIR_TOPIC_IN')
GROUP_ID = _lasair_setting('GROUP_ID', 'LASAIR_GROUP_ID')
TOPIC_OUT = _lasair_setting('TOPIC_OUT', 'LASAIR_TOPIC_OUT')
TEST = _lasair_bool('TEST', 'LASAIR_TEST', default=False)

MAPSDIR = _lasair_setting('MAPSDIR', 'MAPSDIR') or os.path.join(PROJECT_ROOT, 'source', 'maps')

# ---------------------------------------------------------------------------
# NEEDLE inference / annotator
# ---------------------------------------------------------------------------

LABEL_LIST = ['SN', 'SLSN-I', 'TDE']
BCLASSIFIER_PATH = 'source/quality_classification'
MODEL_PATH_TH = 'models/hosted_model/'
MODEL_PATH_T = 'models/hostless_model/'

NEEDLE_OBJ_PATH = 'needle_alerts'
RECORD_PATH = 'records'

CALIBRATED_THRESHOLDS = {
    'hosted': {
        'SLSN-I': 0.81,
        'TDE': 0.80,
        'SN': 0.50
    },
    'hostless': {
        'SLSN-I': 0.53,
        'SN': 0.50
    },
}

# Host-transient angular offset (arcsec). TDE candidates are expected near the nucleus.
OFFSET_TDE_MAX_ARCSEC = 2.0
OFFSET_NUCLEAR_MAX_ARCSEC = 2.0
MISSING_OFFSET_SENTINEL_ARCSEC = 99.0
# Set True when retraining with log_offset / is_offset features (meta dim 33 -> 35).
USE_EXTENDED_OFFSET_FEATURES = True

CALIBRATED_EPOCHS = {
    'hosted': 35,
    'hostless': 80
}

SCALING_PATH = {
    'hosted': 'models/hosted_model/scaling_data.json',
    'hostless': 'models/hostless_model/scaling_data.json'
}

# ---------------------------------------------------------------------------
# Training / preprocessing data paths
# ---------------------------------------------------------------------------

SEED = 667

DATASET_PATH = DATA_DIR / "upsampled_200_20260611_no_split"

EXCEPTION_IMG_PATH = PACKAGE_ROOT / "image" / "failed_objs.txt"
EXCEPTION_LC_PATH = PACKAGE_ROOT / "light_curve" / "failed_objs.txt"

SCALING_DATA_PATH = DATASET_PATH / 'hosted_set' / "scaling_data.json"
SCALING_DATA_HOSTLESS_PATH = DATASET_PATH / 'hostless_set' / "scaling_data.json"

OBJ_INFO_PATH = INFO_DIR / "ztf_train_valid_set.csv"
CSV_2024_PATH = INFO_DIR / "20220301_20240225.csv"
CSV_2025_PATH = INFO_DIR / "20240225_20250603.csv"

DEFAULT_DATA_PATH = _resolve_external_data_path()
MODELS_DIR = Path(
    os.environ.get("NEEDLE_MODELS_PATH", REPO_ROOT / "needle" / "models")
).expanduser().resolve()

IMG_DATA_PATH = DEFAULT_DATA_PATH / 'image_sets_v3'
MAG_DATA_PATH = DEFAULT_DATA_PATH / 'mag_sets_v4'
HOST_DATA_PATH = DEFAULT_DATA_PATH / 'host_ext_20260609'
MAP_PATH = DEFAULT_DATA_PATH / 'maps'
UNTOUCHED_2025_PATH = DEFAULT_DATA_PATH / 'untouched_2025/'
UNTOUCHED_2025_INFO_PATH = UNTOUCHED_2025_PATH / '20240225_20250603.csv'
UNTOUCHED_2025_INPUT_IMG_PATH = UNTOUCHED_2025_PATH / 'images/'
UNTOUCHED_2025_IMG_OUTPUT_PATH = UNTOUCHED_2025_PATH / 'image_preprocessing_output/'
UNTOUCHED_2025_UNMASKED_IMG_OUTPUT_PATH = UNTOUCHED_2025_PATH / 'image_unmasked_output/'
UNTOUCHED_2025_MAG_OUTPUT_PATH = UNTOUCHED_2025_PATH / 'mags/'
UNTOUCHED_2025_HOST_PATH = UNTOUCHED_2025_PATH / 'host_ext_20260609/'
UNTOUCHED_2025_LC_OUTPUT_PATH = UNTOUCHED_2025_PATH / 'light_curve_upsampling_output/'

PHOTO_OUTPUT_PATH = DEFAULT_DATA_PATH / 'photo_processing_output'
IMG_OUTPUT_PATH = DEFAULT_DATA_PATH / 'image_preprocessing_output'
UNMASKED_IMG_OUTPUT_PATH = DEFAULT_DATA_PATH / 'image_unmasked_output'
NEEDLE_SET_PATH = DEFAULT_DATA_PATH / 'needle_inputs'

QUALITY_CLASSIFICATION_DIR = Path(
    os.environ.get(
        "NEEDLE_QUALITY_CLASSIFICATION_PATH",
        REPO_ROOT / "needle/image/quality_classification_tf",
    )
).expanduser().resolve()


def resolve_data_path(path: Union[str, Path]) -> Path:
    """Resolve a relative path against known data roots."""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in (DATA_DIR, NEEDLE_SET_PATH, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return DATA_DIR / path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

RAW_LABEL_DICT = {
    "3-class": {
        'SLSN-II': 0,
        'SN Ib/c': 0,
        'SN Ib-pec': 0,
        'SN IIP': 0,
        'SN Ia': 0,
        'SN Ic-pec': 0,
        'SN II': 0,
        'SN II-pec': 0,
        'SN Ia-CSM': 0,
        'SN Ibn': 0,
        'SN Ic-BL': 0,
        'SN IIb': 0,
        'SN Iax': 0,
        'SN Ic': 0,
        'SN Ia-91T': 0,
        'SN Ia-91bg': 0,
        'SN Ib': 0,
        'SN IIn': 0,
        'SN Ia-SC': 0,
        'SN Ia-pec': 0,
        'SN Icn': 0,
        'SLSN-I': 1,
        'TDE': 2,
        'TDE-He': 2,
        'SN Ca-rich-Ca': 3,
        'LRN': 3,
        'LBV': 3,
        'nova': 3,
        'Gap': 3
    },
    "2-class": {
        'SLSN-II': 0,
        'SN Ib/c': 0,
        'SN Ib-pec': 0,
        'SN IIP': 0,
        'SN Ia': 0,
        'SN Ic-pec': 0,
        'SN II': 0,
        'SN II-pec': 0,
        'SN Ia-CSM': 0,
        'SN Ibn': 0,
        'SN Ic-BL': 0,
        'SN IIb': 0,
        'SN Iax': 0,
        'SN Ic': 0,
        'SN Ia-91T': 0,
        'SN Ia-91bg': 0,
        'SN Ib': 0,
        'SN IIn': 0,
        'SN Ia-SC': 0,
        'SN Ia-pec': 0,
        'SN Icn': 0,
        'SLSN-I': 1,
        'TDE': 0,
        'TDE-He': 0,
        'SN Ca-rich-Ca': 0,
        'LRN': 0,
        'LBV': 0,
        'nova': 0,
        'Gap': 0
    },
    "classify": {
        "Ia": 0,
        "II": 0,
        "Stripped Envelope": 0,
        "Ic": 0,
        "Interacting SN": 0,
        "SLSN-I": 1,
        "TDE": 2,
        "Non-SN": 3,
        "Other": 4
    },
    "reverse_label": {
        "0": "SN",
        "1": "SLSN-I",
        "2": "TDE",
        "3": "Non-SN",
        "4": "Other"
    },
    "label-hosted": {
        "SN": 0,
        "SLSN-I": 1,
        "TDE": 2
    },
    "label-hostless": {
        "SN": 0,
        "SLSN-I": 1
    },
    "test_num": {
        "SN": 15,
        "SLSN-I": 15,
        "TDE": 15
    }
}

LABEL_DICT = {'SN': 0, 'SLSN-I': 1, 'TDE': 2}
LABEL_DICT_HOSTED = {'SN': 0, 'SLSN-I': 1, 'TDE': 2}
LABEL_DICT_HOSTLESS = {'Non-SLSN': 0, 'SLSN-I': 1}
LABEL_DICT_SLSN = {'SN': 0, 'SLSN-I': 1, 'TDE': 0}
LABEL_DICT_TDE = {'SN': 0, 'SLSN-I': 0, 'TDE': 1}

FEATURE_LIMIT_DICT = {
    'host_u': [0, 23.3],
    'host_g': [0, 23.2],
    'host_r': [0, 23.1],
    'host_i': [0, 22.3],
    'host_z': [0, 21.4],
    't_g_minus_r': [-50, 50],
    'ratio_recent': [0, 100],
    'offset': [0, 20],
}
