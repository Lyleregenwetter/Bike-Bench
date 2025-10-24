from __future__ import annotations
import os, time, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import httpx
from openai import OpenAI
from httpx import ReadError, RemoteProtocolError, ConnectError, HTTPStatusError, TimeoutException

import re
import json
import glob


# ============================= Stage 1: PRECOMPUTE =============================

def precompute_train_condition_pairs(
    bench,
    device: str = "cpu",
    n_pairs: int = 5,
    batch_size: int = 64,
    max_batches: int = 50,
) -> Dict[str, Any]:
    """Collect n_pairs TRAIN conditions; for each, pick 1 random valid + 1 random invalid bike."""
    data_tens = bench.get_train_data(categorical=False)       # torch.Tensor [N, D]
    data_df   = bench.get_train_data(categorical=True).copy() # pd.DataFrame [N, ...]
    req_names = list(bench.requirement_names)

    is_constraint = ~np.array(bench.is_objective, dtype=bool)
    pairs: List[Dict[str, Any]] = []

    def _pick_one(mask_np: np.ndarray) -> Optional[int]:
        idxs = np.flatnonzero(mask_np)
        if idxs.size == 0:
            return None
        return int(np.random.choice(idxs, size=1)[0])

    batches_done = 0
    while len(pairs) < n_pairs and batches_done < max_batches:
        conds = bench.get_train_conditions(batch_size, mode="embedding")  # dict of tensors/arrays

        # infer batch length
        B = None
        for v in conds.values():
            B = v.shape[0] if isinstance(v, torch.Tensor) else len(v)
            break
        if B is None:
            break

        for i in range(B):
            cond_i = {k: (v[i] if isinstance(v, torch.Tensor) else v[i]) for k, v in conds.items()}

            scores = bench.evaluate(data_tens, cond_i)  # [N, R]
            valid  = torch.all(scores[:, is_constraint] <= 0, dim=1)

            valid_mask_np   = valid.detach().cpu().numpy()
            invalid_mask_np = ~valid_mask_np

            vi = _pick_one(valid_mask_np)
            ii = _pick_one(invalid_mask_np)
            if vi is None or ii is None:
                continue

            scores_np = scores.detach().cpu().numpy()
            scores_df = pd.DataFrame(scores_np, columns=req_names, index=data_df.index)

            pairs.append({
                "cond": cond_i,
                "valid_row":      data_df.iloc[[vi]].reset_index(drop=True),
                "valid_scores":   scores_df.iloc[[vi]].reset_index(drop=True),
                "invalid_row":    data_df.iloc[[ii]].reset_index(drop=True),
                "invalid_scores": scores_df.iloc[[ii]].reset_index(drop=True),
            })
            if len(pairs) >= n_pairs:
                break

        batches_done += 1

    return {"pairs": pairs[:n_pairs], "n_pairs": len(pairs[:n_pairs]), "requirement_names": req_names}


def precompute_all_conditions(
    bench,
    device: str = "cpu",
    n_valid: int = 5,
    n_invalid: int = 5,
    max_conditions: int = 100,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """
    WARNING: Benchmarker.get_single_test_condition() sets has_received_test_conditions=True.
    If you call evaluate() after this, Benchmarker will mark has_used_test_conditions=True.
    """
    data_tens = bench.data
    data_df   = bench.data_categorical
    all_cache: Dict[int, Dict[str, Any]] = {}

    is_constraint = ~np.array(bench.is_objective, dtype=bool)

    for cidx in range(max_conditions):
        cond   = bench.get_single_test_condition(cidx, device=device)  # flips has_received_test_conditions
        scores = bench.evaluate(data_tens, cond)                        # may flip has_used_test_conditions
        valid  = torch.all(scores[:, is_constraint] <= 0, dim=1)

        scores_df = pd.DataFrame(scores.detach().cpu().numpy(), columns=bench.requirement_names, index=data_df.index)
        valid_mask   = valid.cpu().numpy()
        invalid_mask = ~valid_mask

        valid_data,   valid_scores   = data_df[valid_mask],   scores_df[valid_mask]
        invalid_data, invalid_scores = data_df[invalid_mask], scores_df[invalid_mask]

        take_v = min(n_valid,   len(valid_data))
        take_i = min(n_invalid, len(invalid_data))

        def _sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
            if k == 0 or df.empty: return df.iloc[[]].reset_index(drop=True)
            idx = np.random.choice(df.index, size=k, replace=False)
            return df.loc[idx].reset_index(drop=True)

        all_cache[cidx] = {
            "valid_data_subset":     _sample(valid_data, take_v),
            "valid_scores_subset":   _sample(valid_scores, take_v),
            "invalid_data_subset":   _sample(invalid_data, take_i),
            "invalid_scores_subset": _sample(invalid_scores, take_i),
            "valid_n": int(take_v), "invalid_n": int(take_i),
        }

    return all_cache


# ============================== Small UTIL helpers =============================

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def _to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, header=True)

def rough_token_count(text: str) -> int:
    return len(text.replace(",", " ,").split())

def rough_token_count_messages(messages: List[Dict[str, str]]) -> int:
    return sum(rough_token_count(m.get("content", "")) for m in messages)


# ================== Stage 2: BUILD MESSAGES (numbered spec) ===================

def build_prompt_messages_numbered_single(
    bench,
    cache: Dict[int, Dict[str, pd.DataFrame]],
    condition_idx: int,
    static_dir: str,
    n_out: int,
) -> List[Dict[str, str]]:
    entry = cache[condition_idx]
    vd, vs = entry["valid_data_subset"], entry["valid_scores_subset"]
    idd, is_ = entry["invalid_data_subset"], entry["invalid_scores_subset"]
    valid_n, invalid_n = entry["valid_n"], entry["invalid_n"]

    # single human-readable condition text (test) — used at #5 and #14
    condition_sentence = bench.get_test_text_condition(condition_idx)

    # statics as before, but we no longer read 14_text_prompt.txt
    S0  = _read_txt(os.path.join(static_dir, "0_system_prompt.txt"))
    S1  = _read_txt(os.path.join(static_dir, "1_intro.txt"))
    S2  = _read_txt(os.path.join(static_dir, "2_parameter_descriptions.txt"))
    S3  = _read_txt(os.path.join(static_dir, "3_criterion_descriptions.txt"))
    S4  = _read_txt(os.path.join(static_dir, "4_condition_intro.txt"))
    S6  = _read_txt(os.path.join(static_dir, "6_valid_data_intro.txt"))
    S8  = _read_txt(os.path.join(static_dir, "8_valid_scores_intro.txt"))
    S10 = _read_txt(os.path.join(static_dir, "10_invalid_data_intro.txt"))
    S12 = _read_txt(os.path.join(static_dir, "12_invalid_scores_intro.txt"))
    S13 = _read_txt(os.path.join(static_dir, "13_condition.txt"))
    S15 = _read_txt(os.path.join(static_dir, "15_task.txt"))

    def render(t: str) -> str:
        return (t.replace("{{valid_n}}", str(valid_n))
                 .replace("{{invalid_n}}", str(invalid_n))
                 .replace("{{out_n}}", str(n_out)))

    parts: List[Tuple[str, str]] = [
        ("system", render(S0)),                 # 0
        ("user",   render(S1)),                 # 1
        ("user",   render(S2)),                 # 2
        ("user",   render(S3)),                 # 3
        ("user",   render(S4)),                 # 4
        ("user",   condition_sentence),         # 5
        ("user",   render(S6)),                 # 6
        ("user",   _to_csv(vd)),                # 7
        ("user",   render(S8)),                 # 8
        ("user",   _to_csv(vs)),                # 9
        ("user",   render(S10)),                # 10
        ("user",   _to_csv(idd)),               # 11
        ("user",   render(S12)),                # 12
        ("user",   _to_csv(is_)),               # 13 (content)
        ("user",   render(S13)),                # 13_condition.txt
        ("user",   condition_sentence),         # 14  ← now the condition again
        ("user",   render(S15)),                # 15_task.txt
    ]
    return [{"role": r, "content": c} for r, c in parts]



def build_prompt_messages_numbered_paired(
    bench,
    train_cache: Dict[str, Any],
    condition_idx: Optional[int],
    static_dir: str,
    n_out: int,
) -> List[Dict[str, str]]:
    pairs = train_cache["pairs"]
    n_pairs = train_cache["n_pairs"]
    condition_sentences = [bench.conditions_to_sentence(p["cond"]) for p in pairs]

    target_condition_sentence = bench.get_test_text_condition(condition_idx)

    valid_rows, valid_scores, invalid_rows, invalid_scores = [], [], [], []
    for idx, p in enumerate(pairs, start=1):
        vr = p["valid_row"].copy();       vr.insert(0, "ConditionID", idx)
        vs = p["valid_scores"].copy();    vs.insert(0, "ConditionID", idx)
        ir = p["invalid_row"].copy();     ir.insert(0, "ConditionID", idx)
        is_ = p["invalid_scores"].copy(); is_.insert(0, "ConditionID", idx)
        valid_rows.append(vr); valid_scores.append(vs)
        invalid_rows.append(ir); invalid_scores.append(is_)

    valid_rows_df   = pd.concat(valid_rows,   axis=0, ignore_index=True) if valid_rows else pd.DataFrame()
    valid_scores_df = pd.concat(valid_scores, axis=0, ignore_index=True) if valid_scores else pd.DataFrame()
    invalid_rows_df   = pd.concat(invalid_rows,   axis=0, ignore_index=True) if invalid_rows else pd.DataFrame()
    invalid_scores_df = pd.concat(invalid_scores, axis=0, ignore_index=True) if invalid_scores else pd.DataFrame()

    S0  = _read_txt(os.path.join(static_dir, "0_system_prompt.txt"))
    S1  = _read_txt(os.path.join(static_dir, "1_intro.txt"))
    S2  = _read_txt(os.path.join(static_dir, "2_parameter_descriptions.txt"))
    S3  = _read_txt(os.path.join(static_dir, "3_criterion_descriptions.txt"))
    S4  = _read_txt(os.path.join(static_dir, "4_condition_intro_2.txt"))
    S6  = _read_txt(os.path.join(static_dir, "6_valid_data_intro.txt"))
    S8  = _read_txt(os.path.join(static_dir, "8_valid_scores_intro.txt"))
    S10 = _read_txt(os.path.join(static_dir, "10_invalid_data_intro.txt"))
    S12 = _read_txt(os.path.join(static_dir, "12_invalid_scores_intro.txt"))
    S13 = _read_txt(os.path.join(static_dir, "13_condition_2.txt"))
    S15 = _read_txt(os.path.join(static_dir, "15_task.txt"))

    def render(t: str) -> str:
        return (t.replace("{{valid_n}}", str(n_pairs))
                 .replace("{{invalid_n}}", str(n_pairs))
                 .replace("{{out_n}}", str(n_out)))

    parts: List[Tuple[str, str]] = [
        ("system", render(S0)),                                  # 0
        ("user",   render(S1)),                                  # 1
        ("user",   render(S2)),                                  # 2
        ("user",   render(S3)),                                  # 3
        ("user",   render(S4)),                                  # 4
        ("user",   "\n".join(f"[{i+1}] {t}" for i, t in enumerate(condition_sentences))),  # 5
        ("user",   render(S6)),                                  # 6
        ("user",   _to_csv(valid_rows_df)),
        ("user",   render(S8)),                                  # 8
        ("user",   _to_csv(valid_scores_df)),
        ("user",   render(S10)),                                 # 10
        ("user",   _to_csv(invalid_rows_df)),
        ("user",   render(S12)),                                 # 12
        ("user",   _to_csv(invalid_scores_df)),
        ("user",   render(S13)),                                 # 13_condition_2.txt
        ("user",   target_condition_sentence),                   # 14  ← now the text condition
        ("user",   render(S15)),                                 # 15_task.txt
    ]
    return [{"role": r, "content": c} for r, c in parts]


def _messages_to_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    return [{"role": m["role"], "content": [{"type": "input_text", "text": m["content"]}]} for m in messages]


# ===================== Hardened LLM call (retries + fallback) ==================

def call_llm_with_retries(
    messages: List[Dict[str, str]],
    model: str = "gpt-5",
    reasoning_effort: str = "high",
    max_output_tokens: int = 200_000,
    timeout_s: float = 1200.0,
    max_retries: int = 3,
    initial_backoff_s: float = 1.0,
) -> str:
    input_items = _messages_to_input(messages)

    http_client = httpx.Client(
        timeout=httpx.Timeout(timeout_s, read=timeout_s, write=timeout_s, connect=10.0)
    )
    client = OpenAI(http_client=http_client)

    RETRYABLE = (RemoteProtocolError, ReadError, ConnectError, TimeoutException)

    def _stream_once() -> str:
        chunks: List[str] = []
        with client.responses.stream(
            model=model,
            input=input_items,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
        ) as stream:
            for event in stream:
                t = getattr(event, "type", None)
                if t == "response.output_text.delta":
                    chunks.append(event.delta)
                elif t == "response.error":
                    raise RuntimeError(f"OpenAI streaming error: {getattr(event, 'error', None)}")
            _ = stream.get_final_response()
        return "".join(chunks)

    def _nonstream_once() -> str:
        resp = client.responses.create(
            model=model,
            input=input_items,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
        )
        return "".join(resp.output_text)

    backoff = initial_backoff_s
    for attempt in range(1, max_retries + 1):
        try:
            return _stream_once() if attempt < max_retries else _nonstream_once()
        except RETRYABLE:
            time.sleep(backoff * (1.0 + random.random() * 0.25)); backoff *= 2.0
        except HTTPStatusError as e:
            if 500 <= e.response.status_code < 600 and attempt < max_retries:
                time.sleep(backoff); backoff *= 2.0; continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(backoff); backoff *= 2.0; continue
            raise
    raise RuntimeError("Exhausted retries without completing the request.")


# ============================ Stage 3: TOP-LEVEL RUN ===========================

def run_single_condition(
    bench,
    *,
    mode: str,  # "single" or "paired"
    cache: Optional[Dict[int, Dict[str, pd.DataFrame]]] = None,
    condition_idx: Optional[int] = None,
    train_cache: Optional[Dict[str, Any]] = None,
    static_dir: str = "LLM_prompts",
    model: str = "gpt-5",
    reasoning_effort: str = "high",
    save_txt_path: Optional[str] = None,
    n_out: int = 5,
    timeout_s: float = 1200.0,
) -> str:
    """
    mode="single": uses test cache + condition_idx.
    mode="paired": uses train_cache with n_pairs conditions listed at #5.
    """
    if mode == "single":
        if cache is None or condition_idx is None:
            raise ValueError("For mode='single', provide cache and condition_idx.")
        messages = build_prompt_messages_numbered_single(
            bench=bench, cache=cache, condition_idx=condition_idx, static_dir=static_dir, n_out=n_out
        )
        token_hint = f"unconditional/{condition_idx}"
    elif mode == "paired":
        if train_cache is None:
            raise ValueError("For mode='paired', provide train_cache (from precompute_train_condition_pairs).")
        messages = build_prompt_messages_numbered_paired(
            bench=bench, train_cache=train_cache, condition_idx=condition_idx, static_dir=static_dir, n_out=n_out
        )
        token_hint = f"conditional/{condition_idx}"
    else:
        raise ValueError("mode must be 'single' or 'paired'.")

    print(f"{token_hint} ≈ prompt tokens: {rough_token_count_messages(messages)}")

    raw = call_llm_with_retries(
        messages=messages,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_s=timeout_s,
        max_output_tokens=200_000,
        max_retries=3,
    )

    if save_txt_path:
        os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
        with open(save_txt_path, "w", encoding="utf-8") as f:
            f.write(raw)

    return raw



def _extract_first_csv_block(text: str) -> str:
    m = re.search(r"```(?:csv)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    return (m.group(1) if m else text).strip()

def _split_rows(text: str) -> List[str]:
    return [ln for ln in text.splitlines() if ln.strip() != ""]

def _tokenize(line: str, sep: str) -> List[str]:
    if sep == r"\s+":
        return [tok for tok in re.split(r"\s+", line.strip()) if tok != ""]
    else:
        return [tok.strip() for tok in line.split(sep)]

def _choose_best_sep(lines: List[str], expected_cols_total: int) -> str:
    """Pick the separator that yields the most lines with the expected total columns."""
    candidates = [",", ";", "\t", r"\s+"]
    best_sep, best_good = ",", -1
    for sep in candidates:
        good = 0
        for ln in lines:
            toks = _tokenize(ln, sep)
            # Accept either index+features OR features-only for robustness
            if len(toks) == expected_cols_total or len(toks) == expected_cols_total - 1:
                good += 1
        if good > best_good or (good == best_good and sep == ","):
            best_sep, best_good = sep, good
    return best_sep

def _to_features_drop_firstcol(
    lines: List[str], sep: str, feature_cols: int, expected_cols_total: int
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """
    Convert raw token lines to a features-only DataFrame by:
      - Accepting rows as either [index + features] (65) or [features only] (64),
      - Dropping the first token if length == 65,
      - Keeping a parallel index Series (numeric or NaN) for ordering/reporting,
      - Counting how many rows had wrong length.
    Returns: (features_df[str], idx_series[float], bad_len_count)
    """
    good_rows: List[List[str]] = []
    idx_vals: List[Optional[str]] = []
    bad_len = 0

    for ln in lines:
        toks = _tokenize(ln, sep)
        if len(toks) == expected_cols_total:
            # index + 64 features
            idx_vals.append(toks[0])
            good_rows.append(toks[1:])
        elif len(toks) == feature_cols:
            # already 64 features, no index token present
            idx_vals.append(None)
            good_rows.append(toks)
        else:
            bad_len += 1

    df = pd.DataFrame(good_rows, dtype=str)
    idx_series = pd.to_numeric(pd.Series(idx_vals), errors="coerce")
    return df, idx_series, bad_len

def _filter_sort_report(
    features_raw: pd.DataFrame, idx_series: pd.Series, out_n: int
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Use the optional integer index (1..out_n) to:
      - filter invalid indices,
      - drop duplicate indices (keep first),
      - sort ascending by index,
    but always return features-only data (no index written).
    """
    # valid integer indices in [1, out_n]
    mask_valid_idx = idx_series.notna() & (idx_series % 1 == 0) & (idx_series >= 1) & (idx_series <= out_n)
    dropped_bad_index = int((~mask_valid_idx).sum())

    df2 = features_raw.loc[mask_valid_idx].copy()
    df2["__idx__"] = idx_series.loc[mask_valid_idx].astype(int).to_numpy()

    # drop duplicate indices (keep first)
    before = df2.shape[0]
    df2 = df2.drop_duplicates(subset="__idx__", keep="first")
    kept_unique = df2.shape[0]
    dup_dropped = before - kept_unique

    # sort by index
    df2 = df2.sort_values("__idx__")

    # presence/missing report
    present = set(df2["__idx__"].tolist())
    missing = [i for i in range(1, out_n + 1) if i not in present]

    # remove helper before returning
    features_df = df2.drop(columns="__idx__", errors="ignore").reset_index(drop=True)

    report = {
        "dropped_bad_index": int(dropped_bad_index),
        "kept_unique_rows": int(kept_unique),
        "missing_indices": missing,
        "duplicate_indices_dropped": int(dup_dropped),
    }
    return features_df, report

# --- main (drop-first-row, drop-first-token) ---

def process_raw_file_lenient(
    raw_txt_path: str,
    out_csv_path: str,
    report_json_path: Optional[str],
    out_n: int,
    feature_cols: int = 64,
) -> Dict[str, Any]:
    expected_cols_total = feature_cols + 1  # index + features
    report: Dict[str, Any] = {
        "source": raw_txt_path, "parsed": False, "valid": False, "reason": None,
        "rows_in": None, "rows_kept": None, "cols_expected_total": expected_cols_total,
        "bad_len_rows": None, "dropped_bad_index": None, "duplicate_indices_dropped": None,
        "missing_indices": None, "output_path": out_csv_path,
        # kept for compatibility with your previous reporting
        "header_present": True, "header_validated": False, "header_mismatch_positions": [],
        "detected_header": None,
    }

    with open(raw_txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    text = _extract_first_csv_block(raw)
    lines_all = _split_rows(text)
    report["rows_in"] = len(lines_all)

    if not lines_all:
        report["reason"] = "empty_input"
        if report_json_path:
            os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
            json.dump(report, open(report_json_path, "w"), indent=2)
        return report

    # Choose sep using both 65 and 64 token possibilities
    sep = _choose_best_sep(lines_all, expected_cols_total)

    # === Bulletproof rule: drop the first row unconditionally ===
    lines = lines_all[1:]

    # Build features-only DF while tracking (optional) indices for ordering/report
    features_raw, idx_series, bad_len = _to_features_drop_firstcol(
        lines, sep, feature_cols, expected_cols_total
    )
    report["bad_len_rows"] = int(bad_len)
    report["parsed"] = True

    if features_raw.empty:
        report["reason"] = "no_rows_with_expected_column_count"
        if report_json_path:
            os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
            json.dump(report, open(report_json_path, "w"), indent=2)
        return report

    # Order/dedupe using index (if present), but output is strictly features
    features_df, idx_report = _filter_sort_report(features_raw, idx_series, out_n)
    report.update(idx_report)
    report["rows_kept"] = int(features_df.shape[0])

    if features_df.empty:
        report["reason"] = "no_rows_with_valid_index"
        if report_json_path:
            os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
            json.dump(report, open(report_json_path, "w"), indent=2)
        return report

    # Final safety check: exactly feature_cols columns
    if features_df.shape[1] != feature_cols:
        report["reason"] = f"wrong_feature_column_count: expected {feature_cols}, got {features_df.shape[1]}"
        if report_json_path:
            os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
            json.dump(report, open(report_json_path, "w"), indent=2)
        return report

    # Save WITHOUT headers (bench expects raw features only)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    features_df.to_csv(out_csv_path, index=False, header=False)
    report["valid"] = True

    if report_json_path:
        os.makedirs(os.path.dirname(report_json_path), exist_ok=True)
        json.dump(report, open(report_json_path, "w"), indent=2)

    return report

# --- batch over a folder ---

def process_raw_folder_lenient(
    raw_dir: str,
    processed_dir: str,
    report_dir: Optional[str],
    out_n: int,
    feature_cols: int = 64,
    pattern: str = "*.txt",
) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, pattern)))
    os.makedirs(processed_dir, exist_ok=True)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    reports: List[Dict[str, Any]] = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        out_csv = os.path.join(processed_dir, f"{name}.csv")
        rep_json = os.path.join(report_dir, f"{name}.json") if report_dir else None
        rep = process_raw_file_lenient(
            raw_txt_path=p,
            out_csv_path=out_csv,
            report_json_path=rep_json,
            out_n=out_n,
            feature_cols=feature_cols,
        )
        reports.append(rep)
    return pd.DataFrame(reports).set_index("source").sort_index()
