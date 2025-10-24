import torch
import json
import os
import torch
import numpy as np
import pandas as pd

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgb
import re
from typing import List, Optional, Dict
from itertools import cycle
from bikebench.transformation import ordered_columns
from textwrap import fill

from bikebench.transformation import ordered_columns
from bikebench.transformation.one_hot_encoding import encode_to_continuous
from bikebench.design_evaluation.design_evaluation import get_standard_evaluations, construct_tensor_evaluator, get_standard_evaluations_with_constraint_threshold
from bikebench.data_loading import data_loading
from bikebench.benchmarking import benchmarking_utils
from bikebench.benchmarking.scoring import get_ref_point

class Benchmarker:
    def __init__(self, device, masked_constraints=False, gradient_free=False):
        self.device = device
        self.data_columns = ordered_columns.bike_bench_columns
        if masked_constraints:
            evaluations_list = get_standard_evaluations_with_constraint_threshold(device, aesthetics_mode="Embedding")
        else:
            evaluations_list = get_standard_evaluations(device, aesthetics_mode="Embedding")
        self.evaluator, self.requirement_names, self.is_objective, self.is_conditional = construct_tensor_evaluator(evaluations_list, self.data_columns, device=device)
        self.eval_count = 0
        self.data = torch.tensor(data_loading.load_bike_bench_train().values, device=device, dtype=torch.float32)
        self.data_categorical = data_loading.load_bike_bench_mixed_modality_train()
        self.masked_constraints = masked_constraints
        self.gradient_free = gradient_free
        self.has_received_test_conditions = False
        self.has_used_test_conditions = False
        self.ref_point = get_ref_point(self.evaluator, self.requirement_names, self.is_objective)

    def get_train_data(self, categorical=False):
        if categorical:
            return self.data_categorical
        return self.data

    def evaluate(self, result, conditions, from_categorical=False):
        if self.has_received_test_conditions and not self.has_used_test_conditions:
            self.has_used_test_conditions = True
            print("Evaluation function called after receiving test conditions; logging as unconditional evaluation!")

        #simple wrapper for self.evaluator above with counting (based on size of result_tens)
        if from_categorical:
            result = encode_to_continuous(result)
            result = torch.tensor(result.values, device=self.device)
        self.eval_count += result.shape[0]
        if self.gradient_free:
            with torch.no_grad():
                return self.evaluator(result, conditions)
        else:
            return self.evaluator(result, conditions)

    def score(self, result_tens):
        main_scores, detailed_scores, all_evaluation_scores = benchmarking_utils.evaluate_designs(result_tens, evaluate_as_aggregate=False)
        #append eval_count, has_used_test_conditions, masked_constraints to main_scores
        main_scores["Evaluation Count"] = self.eval_count
        main_scores["Conditional?"] = not self.has_used_test_conditions
        main_scores["Masked Constraints?"] = self.masked_constraints
        main_scores["Gradient Free?"] = self.gradient_free

        self.result_tens = result_tens
        self.main_scores = main_scores
        self.detailed_scores = detailed_scores
        self.all_evaluation_scores = all_evaluation_scores
        return main_scores, detailed_scores, all_evaluation_scores
    
    def save_results(self, path_prefix):
        #save result_tens, main_scores, detailed_scores, all_evaluation_scores to files.
        # main_scores and detailed_scores are pd series, so we can save them as json.
        os.makedirs(path_prefix, exist_ok=True)
        torch.save(self.result_tens, f"{path_prefix}/result_tens.pt")
        with open(f"{path_prefix}/main_scores.json", "w") as f:
            json.dump(self.main_scores.to_dict(), f, indent=4)
        with open(f"{path_prefix}/detailed_scores.json", "w") as f:
            json.dump(self.detailed_scores.to_dict(), f, indent=4)
        torch.save(self.all_evaluation_scores, f"{path_prefix}/all_evaluation_scores.pt")

    def convert_df_to_continuous(self, df):
        return encode_to_continuous(df)

    def get_single_test_condition(self, idx=0, device="cpu", mode = "embedding"):
        self.has_received_test_conditions = True
        return benchmarking_utils.get_single_test_condition(idx, device=device, mode=mode)

    def get_test_conditions(self):
        self.test_conditions = benchmarking_utils.get_test_conditions()
        self.has_received_test_conditions = True
        return self.test_conditions
    
    def get_train_conditions(self, n, mode = "embedding"):
        return benchmarking_utils.get_train_conditions(n, mode=mode)

    def get_test_text_condition(self, idx):
        condition = self.get_single_test_condition(idx, device=self.device, mode="text")
        self.has_received_test_conditions = True
        text = self.conditions_to_sentence(condition)
        return text

    def get_train_text_conditions(self, n):
        conditions = self.get_train_conditions(n, mode="text")
        texts = [self.conditions_to_sentence({k: conditions[k][i:i+1] for k in conditions}) for i in range(n)]
        return texts

    def conditions_to_sentence(self, cond):
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x)

        rider_vals = to_list(cond['Rider'])
        assert len(rider_vals) == 6, "Expected 6 rider body measurements."

        use_case_vec = to_list(cond['Use Case'])
        use_cases = ["Road Biking", "Mountain Biking", "Commuting"]
        # Use argmax to be robust even if it's not perfectly one-hot
        use_case = use_cases[use_case_vec.index(max(use_case_vec))]

        text = cond.get('Text', '').strip()

        # Name mapping in the fixed order you specified
        parts = [
            "Upper leg length",
            "Lower leg length",
            "Arm length",
            "Torso length",
            "Neck and head length",
            "Torso width",
        ]

        # Build the body dimensions phrase with the required punctuation
        body_dims = ", ".join(
            f"{name} - {val:.10f}" for name, val in zip(parts, rider_vals)
        )

        return (
            f"Rider Body Dimensions: {body_dims}. "
            f"Use Case: {use_case}. "
            f"Bike Description: {text}"
        )


def get_unconditionally_valid_sample(bench, data_tens):
    #We sample random conditions, because we will only use scores for unconditional constraints
    conditions = bench.get_train_conditions(data_tens.shape[0])
    eval_scores = bench.evaluate(data_tens, conditions=conditions) #scores
    constraint_indices = ~np.array(bench.is_objective, dtype=bool) #get indices of constraints
    unconditional_indices = ~np.array(bench.is_conditional, dtype=bool) #get indices of unconditional requirements
    unconditional_constraints = np.logical_and(constraint_indices, unconditional_indices) #get indices of unconditional constraints
    validity_tens = torch.all(eval_scores[:,unconditional_constraints] <= 0, axis=1)

    valid_data = data_tens[validity_tens]

    #Duplicate as many as neede to get to 100 samples 
    valid_datapoints = valid_data.shape[0] 
    repeats = 100//valid_datapoints
    remainder = 100 % valid_datapoints
    repeated_samples = valid_data.repeat(repeats, 1)
    remainder_indices = torch.randperm(valid_data.shape[0])[:remainder]
    remainder_samples = valid_data[remainder_indices]
    all_samples = torch.concat([repeated_samples, remainder_samples], axis=0)

    #repeat for each of the 100 conditions
    all_samples = all_samples.repeat(100,1)

    return all_samples

def get_single_conditionally_valid_sample(bench, data_tens, condition_idx):
    condition = bench.get_single_test_condition(condition_idx, device=bench.device)
    eval_scores = bench.evaluate(data_tens, conditions=condition)
    constraint_indices = ~np.array(bench.is_objective, dtype=bool) #get indices of constraints
    validity_tens = torch.all(eval_scores[:,constraint_indices] <= 0, axis=1)
    valid_data = data_tens[validity_tens]

    #if 0 valid, return random 100 samples
    if valid_data.shape[0] == 0:
        rand_indices = torch.randperm(data_tens.shape[0])[:100]
        condition_samples = data_tens[rand_indices]
        return condition_samples

    #Duplicate as many as neede to get to 100 samples
    valid_datapoints = valid_data.shape[0]
    repeats = 100//valid_datapoints
    remainder = 100 % valid_datapoints
    repeated_samples = valid_data.repeat(repeats, 1)
    remainder_indices = torch.randperm(valid_data.shape[0])[:remainder]
    remainder_samples = valid_data[remainder_indices]
    condition_samples = torch.concat([repeated_samples, remainder_samples], axis=0)
    return condition_samples

def get_conditionally_valid_sample(bench, data_tens):

    all_samples = []
    for i in range(100):
        condition_samples = get_single_conditionally_valid_sample(bench, data_tens, i)
        all_samples.append(condition_samples)
    all_samples = torch.concat(all_samples, dim=0) 
    return all_samples

def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1:'st',2:'nd',3:'rd'}.get(n%10,'th')
    return f"{n}{suffix}"

def _format_num(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-2 or abs(x) >= 1e3:
        s = f"{x:.2e}"
        return re.sub(r"e([+-])0*(\d+)", r"e\1\2", s)
    s = f"{x:.3g}"
    if "e" in s:
        return s
    digits = len(s.replace(".", ""))
    if "." in s:
        zeros_needed = 3 - digits
        if zeros_needed > 0:
            s = s + "0" * zeros_needed
    else:
        zeros_needed = 3 - digits
        s = s + "." + "0" * zeros_needed
    return s

# -------------- NEW: renderer-only dashboard ----------------

class ScoreReportDashboard:
    """
    Renderer-only scorecard:
      - No internal evaluation.
      - Consumes:
          * overall_scores: dict[model_name] -> pd.Series with
              ["Hypervolume", "Constraint Satisfaction Rate", "Maximum Mean Discrepancy"]
          * requirement_scores: dict[model_name] -> torch.Tensor shaped (N, R) or (A, B, R)
             where R = #requirements (e.g., 50). If 3D, flattens first two dims.
      - Obtains requirement names/types from either:
          * requirement_names & is_objective (direct), or
          * construct_tensor_evaluator(evaluations, data_columns, device), using either passed
            'evaluations' or default get_standard_evaluations(device).
    """
    def __init__(
        self,
        requirement_scores: Optional[Dict[str, torch.Tensor]] = None,
        overall_scores: Optional[Dict[str, pd.Series]] = None,
        *,
        filepaths: Optional[Dict[str, str]] = None,
        model_colors: Optional[Dict[str, str]] = None,
        # one of these paths must provide names/types:
        requirement_names: Optional[List[str]] = None,
        is_objective: Optional[List[int]] = None,  # 1=objective, 0=constraint
        evaluations: Optional[List[callable]] = None,
        data_columns: Optional[List[str]] = None,
        device: str = "cpu",
        summary_order: Optional[List[str]] = None,
    ):
        self.device = device

                # --- file loading mode (uses Benchmarker.save_results format) ---
        if filepaths and (requirement_scores or overall_scores):
            warnings.warn("Both filepaths and in-memory scores provided; ignoring filepaths.")
            filepaths = None

        if filepaths:
            loaded_req, loaded_sum = {}, {}
            # Optional extras (not required by the dashboard rendering)
            self.loaded_result_tens: Dict[str, torch.Tensor] = {}
            self.loaded_detailed_scores: Dict[str, pd.Series] = {}

            for name, d in filepaths.items():
                if not os.path.isdir(d):
                    warnings.warn(f"{name}: results directory missing ({d}), skipping.")
                    continue

                aes = os.path.join(d, "all_evaluation_scores.pt")  # required
                ms  = os.path.join(d, "main_scores.json")          # required
                rt  = os.path.join(d, "result_tens.pt")            # optional
                ds  = os.path.join(d, "detailed_scores.json")      # optional

                if not (os.path.exists(aes) and os.path.exists(ms)):
                    warnings.warn(f"{name}: missing required files (need all_evaluation_scores.pt & main_scores.json), skipping.")
                    continue

                try:
                    loaded_req[name] = torch.load(aes, map_location=device)
                    with open(ms, "r") as f:
                        loaded_sum[name] = pd.Series(json.load(f))

                    if os.path.exists(rt):
                        self.loaded_result_tens[name] = torch.load(rt, map_location=device)
                    if os.path.exists(ds):
                        with open(ds, "r") as f:
                            self.loaded_detailed_scores[name] = pd.Series(json.load(f))
                except Exception as e:
                    warnings.warn(f"{name}: failed to load from {d} ({e}), skipping.")
                    continue

            requirement_scores, overall_scores = loaded_req, loaded_sum

        if not (requirement_scores and overall_scores):
            raise ValueError("Provide either (requirement_scores & overall_scores) or filepaths.")

        # --- normalize incoming tensors & collect model names
        if not requirement_scores:
            raise ValueError("requirement_scores cannot be empty")
        self.model_names = list(requirement_scores.keys())

        # shape-normalize scores to (N, R)
        self.requirement_scores = {}
        for name, t in requirement_scores.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Scores for model {name!r} must be a torch.Tensor")
            t = t.to(device)
            if t.ndim == 2:
                pass
            elif t.ndim == 3:
                # support 100x100x50 -> 10000x50
                t = t.reshape(-1, t.shape[-1])
            else:
                raise ValueError(f"Scores for {name!r} must be 2D or 3D, got shape {tuple(t.shape)}")
            self.requirement_scores[name] = t

        # --- overall scores ingestion
        default_summary = [
            "Design Quality ↑ (HV)",
            # "Binary Validity ↑",
            "Constraint Violation ↓",
            "Sim. to Data ↓ (MMD)",
            # "Novelty ↑",
            "Diversity ↓ (DPP)"
        ]
        self._summary_order = list(summary_order) if summary_order is not None else default_summary

        # validate overall_scores against the chosen list
        required_keys = set(self._summary_order)
        for name in self.model_names:
            s = overall_scores.get(name, None)
            if s is None:
                raise ValueError(f"overall_scores missing entry for model {name!r}")
            missing = required_keys - set(s.index)
            if missing:
                raise ValueError(
                    f"overall_scores for {name!r} missing keys from summary_order: {sorted(missing)}"
                )
        self.overall_scores = overall_scores

        # --- requirement names/types
        if (requirement_names is not None) and (is_objective is not None):
            self.eval_names = list(requirement_names)
            self.eval_types = list(is_objective)
        else:
            # Need to derive from evaluator definition
            if evaluations is None:
                evaluations = get_standard_evaluations(device)
            if data_columns is None:
                data_columns = ordered_columns.bike_bench_columns
            # We only need names/types; the evaluator function is unused here.
            _, req_names, req_types, is_conditional = construct_tensor_evaluator(evaluations, data_columns, device=device)
            self.eval_names = req_names
            self.eval_types = req_types

        R = self.requirement_scores[self.model_names[0]].shape[1]
        if len(self.eval_names) != R or len(self.eval_types) != R:
            raise ValueError(f"Requirement count mismatch: tensors have R={R}, "
                             f"but names={len(self.eval_names)}, types={len(self.eval_types)}")

        # --- color handling
        if model_colors is None:
            base = plt.rcParams['axes.prop_cycle'].by_key()['color']
            model_colors = {n: c for n, c in zip(self.model_names, cycle(base))}
            warnings.warn("No model_colors provided; using Matplotlib cycle.")
        self.model_colors = model_colors

        # --- indices
        self.objective_indices  = [i for i,t in enumerate(self.eval_types) if t == 1]
        self.constraint_indices = [i for i,t in enumerate(self.eval_types) if t == 0]
        self.objective_names    = [self.eval_names[i] for i in self.objective_indices]
        self.constraint_names   = [self.eval_names[i] for i in self.constraint_indices]

        # --- precompute per-requirement violation rates from provided scores
        # constraint is satisfied if score <= 0; violation if > 0
        self.model_const_violation = {}  # dict[model] -> np.array shape (num_constraints,)
        for name, T in self.requirement_scores.items():
            if len(self.constraint_indices) == 0:
                self.model_const_violation[name] = np.zeros((0,), dtype=float)
            else:
                C = T[:, self.constraint_indices]  # (N, C)
                self.model_const_violation[name] = (C.detach().cpu().numpy() > 0).mean(axis=0)

    # -------------- plotting --------------

    def show_model(
        self,
        model_name: Optional[str]       = None,
        objectives_per_row: int         = 5,
        constraints_per_row: int        = 40,
        total_width: float              = 12.0,
        summary_cell_height: float      = 0.4,
        objective_cell_height: float    = 1.0,
        truncate_tails_magnitude: float = 0.01,
        filter_invalid: bool            = True,
        min_kde_samples: int            = 3,
        constraint_height_scale: float  = 1.3,
        summary_title_width_in: float   = 1.6,
        title_wrap_chars: int           = 14,
        save_dir: Optional[str]        = None,
    ):
        """
        Render one model’s scorecard:
          * Summary row uses provided overall_scores
          * Objective KDEs use requirement_scores[:, objective_indices]
            - Optional filtering to only valid samples (all constraints <= 0)
          * Constraint tiles show per-requirement violation rates computed from provided scores
        """
        if model_name is None:
            model_name = self.model_names[0]
            warnings.warn(f"No model_name given; defaulting to {model_name!r}")
        if model_name not in self.model_names:
            #warn and skip
            print(f"Model {model_name!r} not found in dashboard data. Skipping.")
        else:
            color = self.model_colors[model_name]

            # validity mask if requested
            valid_mask_by_model = {}
            if filter_invalid and len(self.constraint_indices) > 0:
                for name, T in self.requirement_scores.items():
                    cons = T[:, self.constraint_indices]
                    valid_mask_by_model[name] = (cons <= 0).all(dim=1).detach().cpu().numpy()
            else:
                for name, T in self.requirement_scores.items():
                    valid_mask_by_model[name] = np.ones((T.shape[0],), dtype=bool)

            # gather raw objective arrays (filtered if requested)
            all_raw = {}
            for name, T in self.requirement_scores.items():
                arr = T[:, self.objective_indices].detach().cpu().numpy()
                mask = valid_mask_by_model[name]
                all_raw[name] = arr[mask]

            # ---------------- layout sizes ----------------
            obj_count = len(self.objective_indices)
            con_count = len(self.constraint_indices)
            obj_rows  = int(np.ceil(max(obj_count, 1) / objectives_per_row))
            con_rows  = int(np.ceil(max(con_count, 1) / constraints_per_row))
            cons_cell = total_width/constraints_per_row * constraint_height_scale
            fig_h     = summary_cell_height \
                    + obj_rows*objective_cell_height \
                    + con_rows*cons_cell

            fig = plt.figure(figsize=(total_width, fig_h), dpi=500)
            fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, hspace=0.6)
            outer = fig.add_gridspec(
                3, 1,
                height_ratios=[summary_cell_height, obj_rows*objective_cell_height, con_rows*cons_cell]
            )

            # ---------------- summary row ----------------
            md = pd.DataFrame(self.overall_scores).T  # index=model, columns=metrics

            # Use exactly the order configured in __init__
            summary_cols = [c for c in self._summary_order if c in md.columns]

            def _low_best_from_arrow(label: str) -> bool:
                # True => lower is better (ascending rank), False => higher is better (descending rank)
                if "↓" in label:
                    return True
                if "↑" in label:
                    return False
                raise ValueError(f"Cannot infer direction from label (missing ↑/↓): {label!r}")

            summary_defs = [(col, _low_best_from_arrow(col)) for col in summary_cols]
            single_model = (len(self.model_names) == 1)

            # dynamic grid sized by number of summary metrics
            nsum = len(summary_defs) + 1

            if summary_title_width_in is not None:
                # Convert desired inch-width to a fraction of the figure width
                title_frac = max(0.05, min(summary_title_width_in / total_width, 0.8))
                # Remaining width split evenly across metric panels
                if len(summary_defs) == 0:
                    ratios = [1.0]  # only title
                else:
                    metric_frac = (1.0 - title_frac) / len(summary_defs)
                    ratios = [title_frac] + [metric_frac] * len(summary_defs)
            else:
                # fallback to equal-ish ratios (old behavior)
                ratios = [0.16] + [0.28] * len(summary_defs)

            gs_sum = outer[0].subgridspec(1, len(ratios), width_ratios=ratios, wspace=0.1)

            ax0 = fig.add_subplot(gs_sum[0, 0])

            # Wrap long model names (drop "Scorecard")
            title_text = str(model_name)
            if title_wrap_chars is not None and title_wrap_chars > 0:
                title_text = fill(title_text, width=title_wrap_chars)

            ax0.text(
                0, 0.5, title_text,
                ha='left', va='center',
                fontsize=12, fontweight='bold',
                wrap=True  # let Matplotlib break long words if needed
            )
            ax0.axis('off')

            for i, (col, low_best) in enumerate(summary_defs):
                ax = fig.add_subplot(gs_sum[0, i+1])
                sr = md[col]
                lo, hi = sr.min(), sr.max()
                x0, x1 = lo, hi
                val    = sr.loc[model_name]
                rk     = int(sr.rank(method='min', ascending=low_best).loc[model_name])
                # Special case: only one model — keep the metric title but hide axis chrome
                single_or_uniform = single_model or (hi == lo)
                if single_or_uniform:
                    # No axis chrome
                    ax.set_xticks([]); ax.set_yticks([])
                    for loc in ['left', 'right', 'top', 'bottom']:
                        ax.spines[loc].set_visible(False)
                    ax.grid(False)

                    # Keep metric title
                    ax.text(
                        0.5, 0.75, col,
                        transform=ax.transAxes,
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold'
                    )

                    # Centered value (+ rank; ties show rank 1 via pandas' method='min')
                    ax.text(
                        0.5, 0.35, f"{_format_num(val)} ({_ordinal(rk)})",
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=10, color=color
                    )

                    continue  # skip baseline/ticks/extremes/annotation logic


                ax.hlines(0, x0, x1, color='black', linewidth=1)
                ax.plot([x0, x1], [0, 0], '|k', markersize=4)
                for other_val in sr.values:
                    ax.plot(other_val, 0, '|', color='gray', markersize=6)
                ax.plot(val, 0, '|', color=color, markersize=10)

                ax.text(x0, 0.02, _format_num(x0), ha='center', va='bottom', fontsize=7)
                ax.text(x1, 0.02, _format_num(x1), ha='center', va='bottom', fontsize=7)
                if x1 > x0:
                    frac = (val - x0) / (x1 - x0)
                else:
                    frac = 0.5  # degenerate case; center it

                edge_margin = 0.15  # keep 15% of axes width clear at left/right
                frac = max(edge_margin, min(1.0 - edge_margin, frac))

                ax.text(
                    frac, -0.12, f"{_format_num(val)} ({_ordinal(rk)})",
                    transform=ax.transAxes,  # position in axes coords so clamping works
                    ha='center', va='top',
                    fontsize=8, color=color,
                    clip_on=False  # allow slight draw below axis
                )
                ax.text(0.5, 0.45, col, ha='center', va='bottom',
                        transform=ax.transAxes, fontsize=9, fontweight='bold')
                ax.set_ylim(-0.01, 0.05)
                ax.axis('off')


            # ---------------- objective KDEs ----------------
            gs_obj = outer[1].subgridspec(max(obj_rows, 1), objectives_per_row, wspace=0.05, hspace=0.8)

            if obj_count == 0:
                # No objectives—show a placeholder
                ax = fig.add_subplot(gs_obj[0, 0])
                ax.axis('off')
                ax.text(0.5, 0.5, "No objective metrics found.", ha='center', va='center')
            else:
                for idx in range(obj_count):
                    r, c = divmod(idx, objectives_per_row)
                    ax = fig.add_subplot(gs_obj[r, c])

                    # Gather arrays for this objective across models
                    valid_raws = []
                    for m in self.model_names:
                        raw_m = all_raw[m][:, idx]
                        if raw_m.size >= min_kde_samples:
                            valid_raws.append(raw_m)
                    if not valid_raws:
                        ax.axis('off')
                        continue

                    # global percentile bounds (keep lower bound at 0.0 to match your spec)
                    pooled = np.concatenate(valid_raws, axis=0)
                    pmin = np.min(pooled)
                    high = np.percentile(pooled, 100 * (1 - truncate_tails_magnitude))

                    # If the effective range is less than half the (trimmed) max, lift the floor to the min
                    if (high - pmin) < 0.5 * high:
                        low = pmin
                    else:
                        low = 0.0

                    # prepare trimmed per-model data
                    data_for_kde = {}
                    for m in self.model_names:
                        raw = all_raw[m][:, idx]
                        trimmed = raw[(raw >= low) & (raw <= high)]
                        if trimmed.size >= min_kde_samples:
                            data_for_kde[m] = trimmed

                    if not data_for_kde:
                        ax.axis('off'); continue

                    # plot KDEs and collect means
                    means = {}
                    for m, trimmed in data_for_kde.items():
                        is_focal = (m == model_name)
                        sns.kdeplot(
                            data=trimmed,
                            ax=ax,
                            clip=(low, high),
                            bw_adjust=0.5,
                            color=(self.model_colors[m] if is_focal else 'gray'),
                            alpha=(0.6 if is_focal else 0.2),
                            linewidth=1,
                            fill=is_focal,
                            gridsize=1000,
                            warn_singular=False
                        )
                        means[m] = trimmed.mean()

                    # baseline ticks
                    for m, mv in means.items():
                        ax.plot(
                            mv, 0, '|',
                            color=(self.model_colors[m] if m == model_name else 'gray'),
                            markersize=(10 if m == model_name else 6)
                        )

                    # adjust y-axis
                    ys = []
                    for l in ax.get_lines():
                        yd = l.get_ydata()
                        if len(yd):
                            ys.append(np.nanmax(yd))
                    for coln in ax.collections:
                        get_paths = getattr(coln, "get_paths", None)
                        if callable(get_paths):
                            for p in get_paths():
                                verts = p.vertices
                                if verts.size:
                                    ys.append(np.nanmax(verts[:, 1]))
                    vmax = max(ys, default=0.0)
                    ax.set_ylim(0, vmax * 1.05 if vmax > 0 else 1.0)

                    # focal rank by mean (ascending). If you want direction-aware ranks, add a map here.
                    if model_name in means:
                        mean_val = means[model_name]
                        sorted_models = sorted(means.keys(), key=lambda m: means[m])
                        rk = _ordinal(sorted_models.index(model_name) + 1)
                        y0 = 0.16 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                        ax.text(mean_val, y0, f"({rk})", ha='center', va='bottom', fontsize=7)
                    else:
                        midx = 0.5 * (low + high)
                        midy = 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                        ax.text(midx, midy, "Not enough valid samples!",
                                ha='center', va='center', fontsize=7, color='gray')

                    ax.set_xlim(low, high)
                    ax.set_xticks([low, high])
                    ax.set_xticklabels([_format_num(low), _format_num(high)], fontsize=7)
                    labels = ax.get_xticklabels()
                    if labels:
                        labels[0].set_ha('left')
                        if len(labels) > 1:
                            labels[1].set_ha('right')
                    ax.set_yticks([])
                    ax.set_ylabel("")
                    ax.set_title(self.objective_names[idx], fontsize=9, pad=2)
                    for loc in ['top', 'right', 'left']:
                        ax.spines[loc].set_visible(False)
                    ax.spines['bottom'].set_visible(True)

                # blank any unused axes
                for j in range(obj_count, max(obj_rows, 1) * objectives_per_row):
                    r, c = divmod(j, objectives_per_row)
                    fig.add_subplot(gs_obj[r, c]).axis('off')

            # ---------------- constraints tiles ----------------
            gs_con = outer[2].subgridspec(max(con_rows, 1), constraints_per_row, wspace=0.02)
            white = np.array([1.0, 1.0, 1.0])

            if con_count == 0:
                ax = fig.add_subplot(gs_con[0, 0])
                ax.axis('off')
                ax.text(0.5, 0.5, "No constraints found.", ha='center', va='center')
            else:
                # build matrix for ranking (models x constraints)
                arr = np.stack([self.model_const_violation[m] for m in self.model_names], axis=0)

                for idx in range(con_count):
                    r, c = divmod(idx, constraints_per_row)
                    ax  = fig.add_subplot(gs_con[r, c])

                    rate = self.model_const_violation[model_name][idx]  # violation rate
                    adj  = np.sqrt(rate)
                    face = white*(1-adj) + np.array(to_rgb(color))*adj
                    ax.patch.set_facecolor(tuple(face))

                    # rank lower violation as better (ascending=True)
                    rank = int(pd.Series(arr[:, idx]).rank(method='min', ascending=True)
                            .iloc[self.model_names.index(model_name)])

                    ax.text(0.5, 0.65, f"{rate*100:.0f}%", ha='center', va='center', fontsize=8)
                    ax.text(0.5, 0.22, f"({_ordinal(rank)})", ha='center', va='center', fontsize=7)
                    ax.set_title(f"C{idx+1}", fontsize=9, pad=2)
                    ax.set_xticks([]); ax.set_yticks([])
                    for loc in ax.spines:
                        ax.spines[loc].set_visible(False)

                for j in range(con_count, max(con_rows, 1)*constraints_per_row):
                    r, c = divmod(j, constraints_per_row)
                    fig.add_subplot(gs_con[r, c]).axis('off')

            plt.show()
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"{model_name}_scorecard.png"))
