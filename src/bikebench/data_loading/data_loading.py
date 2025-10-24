# data_loading.py
import os
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Iterable

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from bikebench.resource_utils import datasets_path
from bikebench.transformation.one_hot_encoding import ONE_HOT_ENCODED_BIKEBENCH_COLUMNS, BOOLEAN_COLUMNS

# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
DV_API      = os.environ.get("DATAVERSE_API_URL",  "https://dataverse.harvard.edu/api")
ACCESS_API  = os.environ.get("DATAVERSE_DATA_URL", "https://dataverse.harvard.edu/api/access/datafile")
DATAVERSE_DOI = os.environ.get("DATAVERSE_DOI", "10.7910/DVN/BSJSM6")  # override via env if needed

DATASETS_ROOT = Path(datasets_path("."))

# In-memory cache of remote file index: {doi: { "dir/label": {"id": int, "size": int} }}
_REMOTE_INDEX: Dict[str, Dict[str, Dict]] = {}

# ------------------------------------------------------------------------------------
# Remote index and download-by-name
# ------------------------------------------------------------------------------------
def _headers() -> Optional[Dict[str, str]]:
    tok = os.environ.get("DATAVERSE_API_TOKEN")
    return {"X-Dataverse-key": tok} if tok else None

def _with_pid(doi: str) -> str:
    return doi if doi.startswith(("doi:", "hdl:")) else f"doi:{doi}"

def _list_remote_files(doi: str = DATAVERSE_DOI, *, refresh: bool = False) -> Dict[str, Dict]:
    """
    Return mapping 'dir/label' -> {'id': int, 'size': int} for latestVersion.
    Cached per-DOI; set refresh=True to refetch.
    """
    if (not refresh) and (doi in _REMOTE_INDEX):
        return _REMOTE_INDEX[doi]

    url = f"{DV_API}/datasets/:persistentId"
    # Try with token (can see draft), fall back to anonymous (published)
    r = requests.get(url, params={"persistentId": _with_pid(doi)}, headers=_headers(), timeout=60)
    if r.status_code == 401:
        r = requests.get(url, params={"persistentId": _with_pid(doi)}, timeout=60)
    r.raise_for_status()

    out: Dict[str, Dict] = {}
    for f in r.json()["data"]["latestVersion"].get("files", []):
        df   = f["dataFile"]
        dlab = f.get("directoryLabel") or ""
        lab  = f.get("label") or ""
        key  = f"{dlab}/{lab}" if dlab else lab
        out[key] = {"id": df["id"], "size": int(df.get("filesize", 0))}
    _REMOTE_INDEX[doi] = out
    return out

def _resolve_key_to_id(key: str, doi: str = DATAVERSE_DOI) -> int:
    idx = _list_remote_files(doi)
    if key in idx:
        return idx[key]["id"]
    # one refresh try (handles recent changes)
    idx = _list_remote_files(doi, refresh=True)
    if key in idx:
        return idx[key]["id"]
    # helpful error with suggestions by basename
    base = Path(key).name
    close = sorted(k for k in idx if Path(k).name == base)
    hint = f" Known with same filename: {close[:6]}" if close else ""
    raise FileNotFoundError(f"Dataverse key not found: {key}.{hint}")

def _download_by_key(key: str, dest_path: Path, doi: str = DATAVERSE_DOI) -> None:
    fid = _resolve_key_to_id(key, doi)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{ACCESS_API}/{fid}", stream=True, timeout=180) as r:
        r.raise_for_status()
        tmp = dest_path.with_suffix(dest_path.suffix + ".part")
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as out, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {dest_path.name}"
        ) as pbar:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    out.write(chunk)
                    pbar.update(len(chunk))
        tmp.rename(dest_path)

# ------------------------------------------------------------------------------------
# Local fetch: non-destructive, download only if missing (or repair=True)
# ------------------------------------------------------------------------------------
def download_if_missing(remote_path: str, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    """
    If the local file exists -> return it.
    If missing (or repair=True) -> fetch from Dataverse by *path name*.
    """
    local_path = Path(datasets_path(remote_path))
    if local_path.exists() and not repair:
        return str(local_path)
    _download_by_key(remote_path, local_path, doi=doi)
    return str(local_path)

# ------------------------------------------------------------------------------------
# Archive extraction (no checksum; extract once & reuse)
# ------------------------------------------------------------------------------------
def _key_without_tar_gz(key: str) -> str:
    return key[:-7] if key.endswith(".tar.gz") else key

def _dir_has_files(d: Path) -> bool:
    try:
        next(p for p in d.rglob("*") if p.is_file())
        return True
    except StopIteration:
        return False

def _extract_tar_if_needed(archive_key: str, *, repair: bool = False, keep_archive: bool = True, doi: str = DATAVERSE_DOI) -> Path:
    """
    - If extracted dir exists (non-empty) and repair=False -> reuse it.
    - Else ensure archive exists locally (download by name if needed), then (re)extract.
    """
    if not archive_key.endswith(".tar.gz"):
        raise ValueError(f"Expected .tar.gz key, got: {archive_key}")

    target_rel = _key_without_tar_gz(archive_key)
    target_dir = Path(datasets_path(target_rel))

    if target_dir.exists() and _dir_has_files(target_dir) and not repair:
        return target_dir

    archive_path = Path(download_if_missing(archive_key, repair=repair, doi=doi))

    tmp_dir = target_dir.parent / f".__extract_tmp_{target_dir.name}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        # basic path traversal guard
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            return os.path.commonprefix([abs_directory, abs_target]) == abs_directory
        for m in tar.getmembers():
            dest = os.path.join(tmp_dir, m.name)
            if not is_within_directory(tmp_dir, dest):
                raise Exception("Attempted Path Traversal in Tar File")
        tar.extractall(path=tmp_dir)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_dir.rename(target_dir)

    if not keep_archive:
        try: archive_path.unlink()
        except Exception: pass

    return target_dir

# ------------------------------------------------------------------------------------
# Generic file loaders
# ------------------------------------------------------------------------------------
def load_any_file(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".tab":
        return pd.read_csv(filepath, index_col=0, sep="\t")
    elif ext == ".csv":
        return pd.read_csv(filepath, index_col=0)
    elif ext == ".npy":
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

# ------------------------------------------------------------------------------------
# Dataset loaders (non-destructive; pass repair=True to refetch)
# ------------------------------------------------------------------------------------
def _pair_paths(name_prefix: str, folder: str, y_ext: Optional[str], y_train_is_npy: bool):
    x_train = f"{folder}/{name_prefix}_X_train.csv"
    y_train = f"{folder}/{name_prefix}_Y_train.npy" if y_train_is_npy else f"{folder}/{name_prefix}_Y_train.{y_ext or 'csv'}"
    x_test  = f"{folder}/{name_prefix}_X_test.csv"
    y_test  = f"{folder}/{name_prefix}_Y_test.{y_ext or 'csv'}"
    return x_train, y_train, x_test, y_test

def load_dataset_pair(name_prefix: str, folder: str, y_ext: str = None, y_train_is_npy: bool = False, *, repair: bool = False, doi: str = DATAVERSE_DOI):
    x_train, y_train, _, _ = _pair_paths(name_prefix, folder, y_ext, y_train_is_npy)
    X_train = load_any_file(download_if_missing(x_train, repair=repair, doi=doi))
    Y_train = load_any_file(download_if_missing(y_train, repair=repair, doi=doi))
    return X_train, Y_train

def load_dataset_pair_test(name_prefix: str, folder: str, y_ext: str = None, *, repair: bool = False, doi: str = DATAVERSE_DOI):
    _, _, x_test, y_test = _pair_paths(name_prefix, folder, y_ext, False)
    X_test  = load_any_file(download_if_missing(x_test,  repair=repair, doi=doi))
    Y_test  = load_any_file(download_if_missing(y_test,  repair=repair, doi=doi))
    return X_test, Y_test

# ---- Predictive modeling dataset functions ----
def load_aero_train(*, repair: bool = False, doi: str = DATAVERSE_DOI):      return load_dataset_pair("aero", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_aero_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):       return load_dataset_pair_test("aero", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_structure_train(*, repair: bool = False, doi: str = DATAVERSE_DOI): return load_dataset_pair("structure", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_structure_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):  return load_dataset_pair_test("structure", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_validity_train(*, repair: bool = False, doi: str = DATAVERSE_DOI):  return load_dataset_pair("validity", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_validity_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):   return load_dataset_pair_test("validity", "Predictive_Modeling_Datasets", repair=repair, doi=doi)
def load_usability_cont_train(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    return load_dataset_pair("usability_cont", "Predictive_Modeling_Datasets", y_ext="tab", repair=repair, doi=doi)
def load_usability_cont_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    return load_dataset_pair_test("usability_cont", "Predictive_Modeling_Datasets", y_ext="tab", repair=repair, doi=doi)
def load_aesthetics_train(*, repair: bool = False, doi: str = DATAVERSE_DOI, y_train_is_npy=True):
    return load_dataset_pair("aesthetics", "Predictive_Modeling_Datasets", y_train_is_npy=y_train_is_npy, repair=repair, doi=doi)
def load_aesthetics_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    return load_dataset_pair_test("aesthetics", "Predictive_Modeling_Datasets", repair=repair, doi=doi)

def one_hot_encode_material(data: pd.DataFrame):
    data = data.copy()
    data["Material"] = pd.Categorical(data["Material"], categories=["Steel", "Aluminum", "Titanium"])
    mats_oh = pd.get_dummies(data["Material"], prefix="Material=", prefix_sep="")
    data.drop(["Material"], axis=1, inplace=True)
    return pd.concat([mats_oh, data], axis=1)

def load_structure_train_oh(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    X, Y = load_structure_train(repair=repair, doi=doi);  return one_hot_encode_material(X), Y
def load_structure_test_oh(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    X, Y = load_structure_test(repair=repair, doi=doi);   return one_hot_encode_material(X), Y
def load_validity_train_oh(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    X, Y = load_validity_train(repair=repair, doi=doi);   return one_hot_encode_material(X), Y
def load_validity_test_oh(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    X, Y = load_validity_test(repair=repair, doi=doi);    return one_hot_encode_material(X), Y

# ---- Generative modeling dataset functions ----
def load_bike_bench_train(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench.csv", repair=repair, doi=doi)
    return pd.read_csv(path, index_col=0)

def load_bike_bench_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_test.csv", repair=repair, doi=doi)
    return pd.read_csv(path, index_col=0)

def load_bike_bench_mixed_modality_train(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_mixed_modality.csv", repair=repair, doi=doi)
    df =  pd.read_csv(path, index_col=0)
    categorical_cols = ONE_HOT_ENCODED_BIKEBENCH_COLUMNS
    boolean_cols = BOOLEAN_COLUMNS
    continuous_cols = df.columns.difference(categorical_cols + boolean_cols).tolist()
    df[continuous_cols] = df[continuous_cols].astype(np.float32)
    return df

def load_bike_bench_mixed_modality_test(*, repair: bool = False, doi: str = DATAVERSE_DOI):
    path = download_if_missing("Generative_Modeling_Datasets/bike_bench_mixed_modality_test.csv", repair=repair, doi=doi)
    df = pd.read_csv(path, index_col=0)
    categorical_cols = ONE_HOT_ENCODED_BIKEBENCH_COLUMNS
    boolean_cols = BOOLEAN_COLUMNS
    continuous_cols = df.columns.difference(categorical_cols + boolean_cols).tolist()
    df[continuous_cols] = df[continuous_cols].astype(np.float32)
    return df

# ---- Original_BIKED_Data (numbered helpers) ----
def _find_numbered_file(base_dir: Path, n: int, exts: set[str], paren_style: bool = False) -> str:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int")
    names = [f"({n}){ext}" if paren_style else f"{n}{ext}" for ext in exts]
    # direct child
    for name in names:
        p = base_dir / name
        if p.is_file():
            return str(p)
    # recursive or stem fallback
    for name in names:
        hits = sorted(base_dir.rglob(name))
        if hits: return str(hits[0])
    for p in sorted(base_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            stem = p.stem[1:-1] if (paren_style and p.stem.startswith("(") and p.stem.endswith(")")) else p.stem
            if stem == str(n): return str(p)
    raise FileNotFoundError(f"No file numbered {n} under {base_dir}")

def load_biked_original_png(n: int, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    base = _extract_tar_if_needed("Original_BIKED_Data/Images.tar.gz", repair=repair, doi=doi)
    return _find_numbered_file(base, n, {".png"}, paren_style=True)

def load_biked_original_bcad(n: int, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    base = _extract_tar_if_needed("Original_BIKED_Data/BCAD.tar.gz", repair=repair, doi=doi)
    return _find_numbered_file(base, n, {".bcad"}, paren_style=True)

# ---- Real_Extended_Data (numbered helpers) ----
def load_png(n: int, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    base = _extract_tar_if_needed("Real_Extended_Data/pngs.tar.gz", repair=repair, doi=doi)
    return _find_numbered_file(base, n, {".png"}, paren_style=False)

def load_svg(n: int, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    base = _extract_tar_if_needed("Real_Extended_Data/svgs.tar.gz", repair=repair, doi=doi)
    return _find_numbered_file(base, n, {".svg"}, paren_style=False)

def load_xml(n: int, *, repair: bool = False, doi: str = DATAVERSE_DOI) -> str:
    base = _extract_tar_if_needed("Real_Extended_Data/xmls.tar.gz", repair=repair, doi=doi)
    return _find_numbered_file(base, n, {".xml"}, paren_style=False)
