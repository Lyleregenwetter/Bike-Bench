# -*- coding: utf-8 -*-

import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from bikebench.data_loading import data_loading

# =========================
# XML parsing (lean & safe)
# =========================
_NUM_INT_RE   = re.compile(r'^[+-]?\d+$')
_NUM_FLOAT_RE = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')

def _cast(value):
    """Cast XML entry text to bool/int/float when possible; else keep string."""
    if value is None:
        return np.nan
    s = str(value).strip()
    lo = s.lower()
    if lo == "true":  return True
    if lo == "false": return False
    if _NUM_INT_RE.match(s):
        try: return int(s)
        except Exception: pass
    if _NUM_FLOAT_RE.match(s):
        try: return float(s)
        except Exception: pass
    return s

def parse_bcad_file(path: Path) -> dict:
    tree = ET.parse(data_loading.load_biked_original_bcad(path))
    root = tree.getroot()
    out = {}
    for entry in root.findall("entry"):
        key = entry.get("key")
        if key is None:
            continue
        out[key] = _cast(entry.text)
    return out

# ==================================
# Legacy prefill (compact & robust)
# ==================================
def prefill_old_bcad_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing tube sub/main diameters row-wise to support older BCAD files."""
    df = df.copy()

    TT_SUB  = ["Top tube rear diameter","Top tube rear dia2","Top tube front diameter","Top tube front dia2"]
    TT_MAIN = ["Top tube diameter"]
    DT_SUB  = ["Down tube rear diameter","Down tube rear dia2","Down tube front diameter","Down tube front dia2"]
    DT_MAIN = ["Down tube diameter","Down tube aero diameter"]
    ST_SUB  = ["Seat tube rear diameter","Seat tube rear dia2","Seat tube front diameter","Seat tube front dia2"]
    ST_MAIN = ["Seat tube diameter","Seat tube aero diameter"]
    HT_MAIN = ["Head tube diameter","Head tube aero diameter"]
    CHAINSTAY_MAIN = ["Chain stay back diameter","Chain stay vertical diameter","Chain stay horizontal diameter","CHAINSTAYAUXrearDIAMETER"]
    SEATSTAY_MAIN  = ["Seat stay bottom diameter","SEATSTAY_HR","SEATSTAY_VF","SEATSTAY_HF"]

    for col in TT_SUB + TT_MAIN + DT_SUB + DT_MAIN + ST_SUB + ST_MAIN + HT_MAIN + CHAINSTAY_MAIN + SEATSTAY_MAIN:
        if col not in df.columns:
            df[col] = np.nan

    def _isnum(v):
        return isinstance(v, (int,float,np.integer,np.floating)) and not pd.isna(v)

    def _fill_group(r, subs, mains):
        sub_vals = [r[c] for c in subs  if _isnum(r.get(c))]
        sub_mean = float(np.mean(sub_vals)) if sub_vals else np.nan
        main_vals= [r[c] for c in mains if _isnum(r.get(c))]
        main_fb  = float(main_vals[0]) if main_vals else np.nan

        for c in subs:
            if pd.isna(r[c]):
                v = sub_mean if not np.isnan(sub_mean) else main_fb
                if not np.isnan(v): r[c] = v
        sub_vals = [r[c] for c in subs if _isnum(r.get(c))]
        sub_mean = float(np.mean(sub_vals)) if sub_vals else sub_mean
        for c in mains:
            if pd.isna(r[c]):
                v = sub_mean if not np.isnan(sub_mean) else main_fb
                if not np.isnan(v): r[c] = v
        return r

    def _fill_head_tube(r):
        d, a = r.get("Head tube diameter"), r.get("Head tube aero diameter")
        if pd.isna(d) and _isnum(a): r["Head tube diameter"] = float(a)
        if pd.isna(a) and _isnum(d): r["Head tube aero diameter"] = float(d)
        return r

    df = df.apply(lambda r: _fill_group(r, TT_SUB, TT_MAIN), axis=1)
    df = df.apply(lambda r: _fill_group(r, DT_SUB, DT_MAIN), axis=1)
    df = df.apply(lambda r: _fill_group(r, ST_SUB, ST_MAIN), axis=1)
    df = df.apply(_fill_head_tube, axis=1)
    return df

def drop_designs_with_xml_features(
    df: pd.DataFrame,
    substrings=("TNDM", "EXTRATUBE"),
    *,
    drop_feature_columns: bool = True,
    export_dropped_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove designs whose *original XML* contains any columns with names that include
    the given substrings and have non-null values.

    - substrings: tuple of name fragments to match (case-sensitive, like your code).
    - drop_feature_columns: if True, remove those matched columns from the kept df.
    - export_dropped_path: optional CSV path to save the dropped designs (with feature
      columns removed, matching your old behavior).
    """
    if df.empty:
        return df

    # Find columns whose names contain any of the substrings
    match_cols = [c for c in df.columns if any(sub in c for sub in substrings)]
    if not match_cols:
        if verbose:
            print(f"No feature columns matched {substrings}; nothing to drop.")
        return df

    # Rows to drop: any non-null in any matched column
    has_feature = df[match_cols].notna().any(axis=1)
    n_drop = int(has_feature.sum())

    if n_drop == 0:
        # Optionally still drop the feature columns to keep the frame clean
        out = df.drop(columns=match_cols, errors="ignore") if drop_feature_columns else df.copy()
        if verbose:
            print(f"Matched {len(match_cols)} feature column(s), but no rows had values to drop.")
        return out

    # Build kept and dropped sets
    kept = df.loc[~has_feature].copy()
    dropped = df.loc[has_feature].copy()

    # Remove the feature columns if requested
    if drop_feature_columns:
        kept.drop(columns=match_cols, inplace=True, errors="ignore")
        dropped.drop(columns=match_cols, inplace=True, errors="ignore")

    # Optional export of the dropped designs (like your old function)
    if export_dropped_path is not None:
        dropped.to_csv(export_dropped_path, index=True)

    if verbose:
        print(f"Dropped {n_drop} design(s) due to features in {substrings}. Kept {len(kept)}.")

    return kept

# =========================
# Conversions & derivation
# =========================
def _safe_mean(vals):
    xs = [float(v) for v in vals if isinstance(v, (int,float,np.integer,np.floating)) and not pd.isna(v)]
    return float(np.mean(xs)) if xs else np.nan

def _getf(row, key):
    v = row.get(key, np.nan)
    return float(v) if isinstance(v, (int,float,np.integer,np.floating)) and not pd.isna(v) else np.nan

def convert_bike_bench(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Wheel deltas (ensure sources exist)
    for c in ["Wheel diameter rear","Wheel diameter front","ERD rear","ERD front","BSD rear","BSD front"]:
        if c not in df.columns: df[c] = np.nan
    df["RDERD"] = df["Wheel diameter rear"]  - df["ERD rear"]
    df["FDERD"] = df["Wheel diameter front"] - df["ERD front"]
    df["RDBSD"] = df["Wheel diameter rear"]  - df["BSD rear"]
    df["FDBSD"] = df["Wheel diameter front"] - df["BSD front"]
    df.drop(columns=["ERD rear","ERD front","BSD rear","BSD front"], inplace=True, errors="ignore")

    # Row-wise geometry & tube diameters
    def _row_calc(row):
        out = {}

        # DT Length & Stack
        BBD  = _getf(row,"BB textfield")
        FCD  = _getf(row,"FCD textfield")
        WDR  = _getf(row,"Wheel diameter rear")
        WDF  = _getf(row,"Wheel diameter front")
        x    = _getf(row,"FORK0R")
        fkl  = _getf(row,"FORK0L")
        htlx = _getf(row,"Head tube lower extension2")
        lsth = _getf(row,"Lower stack height")
        ha   = _getf(row,"Head angle")

        if not any(pd.isna([BBD,FCD,WDR,WDF,ha])):
            FTY = BBD - WDR/2.0 + WDF/2.0
            FTX_sq = FCD**2 - FTY**2
            FTX = np.sqrt(max(FTX_sq, 0.0))
            y = (fkl or 0.0) + (htlx or 0.0) + (lsth or 0.0)
            ha_rad = np.deg2rad(ha)
            dtx = FTX - y*np.cos(ha_rad) - (x or 0.0)*np.sin(ha_rad)
            dty = FTY + y*np.sin(ha_rad) + (x or 0.0)*np.cos(ha_rad)
            out["DT Length"] = float(np.sqrt(dtx**2 + dty**2))
            htl = _getf(row,"Head tube length textfield")
            stack_y = (fkl or 0.0) + (lsth or 0.0) + (htl or 0.0)
            out["Stack"] = float(FTY + stack_y*np.sin(ha_rad) + (x or 0.0)*np.cos(ha_rad))

        # Average diameters
        out["csd"] = _safe_mean([row.get("Chain stay back diameter"), row.get("Chain stay vertical diameter"), row.get("Chain stay horizontal diameter"), row.get("CHAINSTAYAUXrearDIAMETER")])
        out["ssd"] = _safe_mean([row.get("Seat stay bottom diameter"), row.get("SEATSTAY_HR"), row.get("SEATSTAY_VF"), row.get("SEATSTAY_HF")])

        # Top tube diameter
        tt_type = int(row.get("Top tube type", 1)) if pd.notna(row.get("Top tube type", np.nan)) else 1
        if tt_type == 1:
            out["ttd"] = _safe_mean([
                row.get("Top tube rear diameter"), row.get("Top tube rear dia2"),
                row.get("Top tube front diameter"), row.get("Top tube front dia2"),
            ])
        else:
            out["ttd"] = row.get("Top tube diameter", np.nan)

        # Down tube diameter
        dt_type = int(row.get("Down tube type", 1)) if pd.notna(row.get("Down tube type", np.nan)) else 1
        if dt_type == 2:
            out["dtd"] = _safe_mean([
                row.get("Down tube rear diameter"), row.get("Down tube rear dia2"),
                row.get("Down tube front diameter"), row.get("Down tube front dia2"),
            ])
        elif dt_type == 0:
            out["dtd"] = row.get("Down tube aero diameter", np.nan)
        else:
            out["dtd"] = row.get("Down tube diameter", np.nan)

        # Seat tube diameter
        st_type = int(row.get("Seat tube type", 1)) if pd.notna(row.get("Seat tube type", np.nan)) else 1
        if st_type == 2:
            out["std"] = _safe_mean([
                row.get("Seat tube rear diameter"), row.get("Seat tube rear dia2"),
                row.get("Seat tube front diameter"), row.get("Seat tube front dia2"),
            ])
        elif st_type == 0:
            out["std"] = row.get("Seat tube aero diameter", np.nan)
        else:
            out["std"] = row.get("Seat tube diameter", np.nan)

        # Head tube diameter (aero vs round)
        ht_type = int(row.get("Head tube type", 1)) if pd.notna(row.get("Head tube type", np.nan)) else 1
        out["htd"] = row.get("Head tube aero diameter", np.nan) if ht_type == 0 else row.get("Head tube diameter", np.nan)

        # Fixed wall thickness values
        out["Wall thickness Bottom Bracket"] = 2.0
        out["Wall thickness Head tube"]      = 1.1

        return pd.Series(out)

    derived = df.apply(_row_calc, axis=1)
    for c in derived.columns:
        df[c] = derived[c]

    if "Seat stay mount location" in df.columns:
        m = pd.to_numeric(df["Seat stay mount location"], errors="coerce")
        df = df.loc[m.isna() | m.isin([0, 1, 5])]  # drop 2,3,...; keep NaN if present
        m = pd.to_numeric(df["Seat stay mount location"], errors="coerce")  # re-eval after filter
        df.loc[m.isin([1, 5]), "Seat stay junction0"] = (
            pd.to_numeric(df.loc[m == 1, "Seat stay junction0"], errors="coerce")
            + pd.to_numeric(df.loc[m == 1, "Seat tube extension2"], errors="coerce")
            - pd.to_numeric(df.loc[m == 1, "ttd"], errors="coerce") / 2.0
        )

    # Expand any "*sRGB" into R/G/B
    for col in list(df.columns):
        if isinstance(col, str) and col.endswith("sRGB"):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64) + 2**24
            base = col[:-4]
            df[f"{base}R_RGB"] = (vals % 2**24) // 2**16
            df[f"{base}G_RGB"] = (vals % 2**16) // 2**8
            df[f"{base}B_RGB"] =  vals % 2**8
            df.drop(columns=[col], inplace=True)

    return df



TYPE_SPEC = {
    "Seatpost LENGTH": "float",
    "CS textfield": "float",
    "BB textfield": "float",
    "Stack": "float",
    "Head angle": "float",
    "Head tube length textfield": "float",
    "Seat stay junction0": "float",
    "Seat tube length": "float",
    "Seat angle": "float",
    "DT Length": "float",
    "FORK0R": "float",
    "BB diameter": "float",
    "ttd": "float",
    "dtd": "float",
    "csd": "float",
    "std": "float",
    "htd": "float",
    "ssd": "float",
    "Chain stay position on BB": "float",
    "SSTopZOFFSET": "float",
    "MATERIAL": "cat",
    "Head tube upper extension2": "float",
    "Seat tube extension2": "float",
    "Head tube lower extension2": "float",
    "SEATSTAYbrdgshift": "float",
    "CHAINSTAYbrdgshift": "float",
    "SEATSTAYbrdgdia1": "float",
    "CHAINSTAYbrdgdia1": "float",
    "SEATSTAYbrdgCheck": "bool",
    "CHAINSTAYbrdgCheck": "bool",
    "Dropout spacing": "float",
    "Wall thickness Bottom Bracket": "float",
    "Wall thickness Top tube": "float",
    "Wall thickness Head tube": "float",
    "Wall thickness Down tube": "float",
    "Wall thickness Chain stay": "float",
    "Wall thickness Seat stay": "float",
    "Wall thickness Seat tube": "float",
    "Wheel diameter front": "float",
    "RDBSD": "float",
    "Wheel diameter rear": "float",
    "FDBSD": "float",
    "Fork type": "int",
    "Stem kind": "int",
    "Handlebar style": "cat",
    "BB length": "float",
    "Wheel cut": "float",
    "BELTorCHAIN": "bool",
    "Number of cogs": "int",
    "Number of chainrings": "int",
    "FIRST color R_RGB": "int",
    "FIRST color G_RGB": "int",
    "FIRST color B_RGB": "int",
    "RIM_STYLE front": "cat",
    "RIM_STYLE rear": "cat",
    "SPOKES composite front": "int",
    "SPOKES composite rear": "int",
    "SBLADEW front": "float",
    "SBLADEW rear": "float",
    "Saddle length": "float",
    "Saddle height": "float",
    "Seat tube type": "int",
    "Head tube type": "int",
    "Down tube type": "int",
}

_NAN_LIKE = {"nan","none","null",""}

def _coerce_nan_like(x):
    if x is None: return np.nan
    if isinstance(x, float) and np.isnan(x): return np.nan
    if isinstance(x, str) and x.strip().lower() in _NAN_LIKE: return np.nan
    return x

def _coerce_column(s: pd.Series, kind: str) -> pd.Series:
    if kind == "bool":
        if s.dtype == bool: return s
        mapping = {"true":True,"t":True,"1":True,"yes":True,"y":True,
                   "false":False,"f":False,"0":False,"no":False,"n":False}
        def conv(v):
            if isinstance(v,(bool,np.bool_)): return bool(v)
            if isinstance(v,(int,np.integer)) and v in (0,1): return bool(v)
            if isinstance(v,str):
                key = v.strip().lower()
                if key in mapping: return mapping[key]
            return np.nan
        out = s.map(conv)
        mode = out.mode(dropna=True)
        fill = bool(mode.iloc[0]) if len(mode) else False
        return out.fillna(fill).astype(bool)

    if kind == "int":
        coerced = pd.to_numeric(s, errors="coerce")
        mode = coerced.mode(dropna=True)
        fill = int(mode.iloc[0]) if len(mode) else 0
        coerced = coerced.fillna(fill)
        if s.name in {"FIRST color R_RGB","FIRST color G_RGB","FIRST color B_RGB"}:
            coerced = coerced.clip(0, 255)
        return coerced.round().astype(int)

    if kind == "float":
        coerced = pd.to_numeric(s, errors="coerce")
        med = coerced.median(skipna=True)
        fill = float(med) if not np.isnan(med) else 0.0
        return coerced.fillna(fill).astype(float)

    if kind == "cat":
        s = s.map(_coerce_nan_like)
        if s.notna().any():
            m = s.mode(dropna=True)
            fill = m.iloc[0] if len(m) else "Unknown"
        else:
            fill = "Unknown"
        return s.fillna(fill).astype("category")

    return s  # unknown kind

def apply_type_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col, kind in TYPE_SPEC.items():
        if col in df.columns:
            df[col] = _coerce_column(df[col], kind)
    return df

def filter_by_seat_stay_mount_location(
    df: pd.DataFrame,
    col: str = "Seat stay mount location",
    keep_values=(0, 1, 5),
    drop_missing: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only rows where `col` ∈ keep_values (defaults to {0,1}).
    If drop_missing=False, rows with NaN in `col` are kept.
    """
    if df.empty or col not in df.columns:
        if verbose:
            print(f"No filtering: column '{col}' not present.")
        return df

    s = pd.to_numeric(df[col], errors="coerce")
    keep_mask = s.isin(keep_values) | (s.isna() & (not drop_missing))
    dropped = int((~keep_mask).sum())
    if verbose and dropped > 0:
        print(f"Dropping {dropped} designs where '{col}' ∉ {keep_values}.")
    return df.loc[keep_mask].copy()

def jitter_thickness(df: pd.DataFrame) -> pd.DataFrame:
    """Add Gaussian noise to wall thickness columns to avoid exact duplicates."""
    mean = np.zeros(7)
    cov = 0.1*np.eye(7) + 0.4 * np.ones((7, 7))

    scaler_exp = np.random.multivariate_normal(mean, cov, size=len(df))
    scaler = np.exp(scaler_exp)
    df = df.copy()
    df_thickness = df[THICKNESS_COLS].values * scaler
    df[THICKNESS_COLS] = df_thickness
    return df
def report_and_drop_exact_duplicates(
    df: pd.DataFrame,
    keep: str = "first",        # same semantics as DataFrame.duplicated
    verbose: bool = True,
    max_groups_to_print: int = 200,
    export_csv: str | None = None,
) -> pd.DataFrame:
    """
    Find exact duplicate rows across ALL columns in `df`, print which IDs are dropped,
    then return a de-duplicated frame.
    - Keeps the first occurrence in the current row order (your index is the file id).
    - Prints each duplicate set as: kept <id>, dropped [ids...].
    """
    if df.empty:
        return df.copy()

    # Hash each row across all columns (NaNs in the same places hash the same)
    row_hash = pd.util.hash_pandas_object(df, index=False)
    dup_mask = row_hash.duplicated(keep=False)

    if not dup_mask.any():
        if verbose:
            print("No exact duplicate designs found.")
        return df.copy()

    # Build groups in original order
    groups = {}
    for idx_label, hval in row_hash[dup_mask].items():
        groups.setdefault(hval, []).append(idx_label)

    # Print & collect drops
    dropped_ids = []
    if verbose:
        print(f"Found {len(groups)} exact duplicate set(s):")
    for i, (hval, id_list) in enumerate(groups.items(), start=1):
        # preserve original order of appearance
        # (id_list is already in DataFrame order because we iterated row_hash in order)
        kept_id = id_list[0] if keep == "first" else id_list[-1]
        drop_list = id_list[1:] if keep == "first" else id_list[:-1]
        dropped_ids.extend(drop_list)
        if verbose and i <= max_groups_to_print:
            print(f"  set {i:>3}: kept {kept_id}, dropped {drop_list}")
    if verbose and len(groups) > max_groups_to_print:
        print(f"  ... {len(groups) - max_groups_to_print} more duplicate set(s) not shown")

    if export_csv:
        pd.Series(dropped_ids, name="dropped_duplicate_ids").to_csv(export_csv, index=False)

    # Drop them
    return df.loc[~df.index.isin(dropped_ids)].copy()


MANUAL_ELIM = [199, 240, 751, 754, 1062, 1065, 1104, 1151, 1154, 1209, 1232, 1233, 1287, 1321, 1344, 1346, 1355, 1356, 1382, 1416, 1453, 
                      1457, 1464, 1546, 1787, 1863, 1873, 2019, 2163, 2405, 2641, 2770, 2772, 2853, 2880, 2884, 2890, 3032, 3125, 3126, 3127, 3202, 
                      3142, 3144, 3161, 3203, 3207, 3209, 3214, 3504, 3505, 3509, 3513, 3515, 3554, 3555, 3651, 3779, 3988, 3978, 3981, 4093,
                      4200, 4231, 4232, 4297, 4319]


# ===========================
# Main builder (end-to-end)
# ===========================
def build_bikebench_dataframe(
    n_files: int = 4800,
    outlier_sd: float = 10.0,
    jitter: bool = True,
    show_progress: bool = True,
    
) -> pd.DataFrame:
    """
    Build the Bike-Bench DataFrame from BCAD XML files (simplified pipeline).
      1) Parse (1).bcad...(n).bcad with tqdm.
      2) Prefill legacy tube fields.
      3) Convert & derive.
      4) Select columns directly from TYPE_SPEC order and coerce/impute by type.
      5) Drop numeric 10-SD outliers.
    Index is the numeric file id; index name remains None.
    """

    # Parse
    records, ids = [], []
    it = tqdm(range(1, n_files + 1), desc="Loading BCAD", unit="file") if show_progress else range(1, n_files + 1)
    for idx in range(1, n_files + 1):
        try:
            rec = parse_bcad_file(idx)
            records.append(rec)
            ids.append(idx)
        except Exception as e:
            print(f"Warning: failed to parse {idx}: {e}")

    if not records:
        df_empty = pd.DataFrame()
        df_empty.index = pd.Index([], name=None)
        return df_empty

    raw_df = pd.DataFrame.from_records(records, index=ids).sort_index()
    raw_df.index.name = None

    raw_df = drop_designs_with_xml_features(
    raw_df,
    substrings=("TNDM", "EXTRATUBE"),
    drop_feature_columns=True,    
    export_dropped_path=None,  
    verbose=True,
)

    # Prefill legacy fields, convert/derive
    raw_df = prefill_old_bcad_fields(raw_df)

    # NEW: drop designs where Seat stay mount location is not 0
    raw_df = filter_by_seat_stay_mount_location(raw_df, keep_values=(0,1,5), drop_missing=False, verbose=True)

    # Convert/derive
    converted = convert_bike_bench(raw_df)  

    # Select columns per TYPE_SPEC order (only those present)
    ordered_cols = [c for c in TYPE_SPEC.keys() if c in converted.columns]
    result = converted[ordered_cols].copy()
    result.index.name = None

    result["MATERIAL"] = result["MATERIAL"].map(_material_to_3class)

    # Enforce schema & impute
    result = apply_type_schema(result)

    # Enforce schema & impute
    result = apply_type_schema(result)

    result = report_and_drop_exact_duplicates(
        result,
        keep="first",
        verbose=True,
        max_groups_to_print=200,          # tweak if too chatty / too quiet
        export_csv=None                   # e.g., "duplicates_dropped.csv" to archive
    )

    # Drop 10-SD outliers on numeric only
    num_cols = result.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        means = result[num_cols].mean()
        stds  = result[num_cols].std(ddof=0)
        valid = stds[stds > 0].index
        if len(valid) > 0:
            diffs = (result[valid] - means[valid]).abs()
            mask  = (diffs > (outlier_sd * stds[valid])).any(axis=1)
            result = result.loc[~mask]

    result.index.name = None

    # Manual eliminations (based on visual inspection of outliers)
    before = len(result)
    result = result.loc[~result.index.isin(MANUAL_ELIM)]
    if before != len(result):
        print(f"Dropped {before - len(result)} manually removed design(s).")

    if jitter:
        result = jitter_thickness(result)

    return result

def _material_to_3class(val: object) -> str:
    """Map raw MATERIAL values to three classes: ALUMINIUM, STEEL, BAMBOO."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "ALUMINIUM"  # treat missing like OTHER -> ALUMINIUM

    s = str(val).strip().upper()
    # normalize a bit
    s_norm = re.sub(r"[^A-Z]", "", s)  # remove spaces, dashes, etc.

    # direct keep
    if s_norm == "TITANIUM":
        return "TITANIUM"
    if s_norm == "STEEL":
        return "STEEL"

    # map CARBON (and a couple common aliases) to STEEL
    if s_norm == "CARBON":
        return "STEEL"

    # everything else (OTHER, ALUMINIUM, ALLOY, unknowns...) → ALUMINIUM
    return "ALUMINIUM"


THICKNESS_COLS = [
    "Wall thickness Bottom Bracket",
    "Wall thickness Top tube",
    "Wall thickness Head tube",
    "Wall thickness Down tube",
    "Wall thickness Chain stay",
    "Wall thickness Seat stay",
    "Wall thickness Seat tube",
]

