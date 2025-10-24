from abc import abstractmethod, ABC
from typing import List, Dict

import pandas as pd
import torch


# TODO: write validation functions to [optionally] be able to grab values from the default bike when not found?

class ValidationFunction(ABC):
    @abstractmethod
    def friendly_name(self) -> str:
        """
        Should return a user-friendly and easily comprehensible name for the validation in question.
        """
        pass

    @abstractmethod
    def variable_names(self) -> List[str]:
        """
        Should return a list of variable names used in the validation.
        """
        pass

    @abstractmethod
    def validate(self, designs: torch.Tensor) -> torch.Tensor:
        """
        Should return a PyTorch tensor with shape (len(designs), 1) or (len(designs),).
        The values in the tensor represent validity. 1 is invalid, 0 is valid.
        """
        pass

class FeatureStore:
    def __init__(self, designs: torch.Tensor, name_to_idx: Dict[str, int]):
        self.X = designs
        self.idx = name_to_idx
        self.cache: Dict[str, torch.Tensor] = {}

    def col(self, name: str) -> torch.Tensor:
        t = self.cache.get(name)
        if t is None:
            t = self.X[:, self.idx[name]]
            self.cache[name] = t
        return t

    # Angles (radians)
    @property
    def theta_ht(self):
        k = "_theta_ht"
        t = self.cache.get(k)
        if t is None:
            t = torch.deg2rad(self.col("Head angle"))
            self.cache[k] = t
        return t

    @property
    def theta_st(self):
        k = "_theta_st"
        t = self.cache.get(k)
        if t is None:
            t = torch.deg2rad(self.col("Seat angle"))
            self.cache[k] = t
        return t

    @property
    def sin_ht(self):
        k = "_sin_ht"
        t = self.cache.get(k)
        if t is None:
            t = torch.sin(self.theta_ht); self.cache[k] = t
        return t

    @property
    def cos_ht(self):
        k = "_cos_ht"
        t = self.cache.get(k)
        if t is None:
            t = torch.cos(self.theta_ht); self.cache[k] = t
        return t

    @property
    def tan_ht(self):
        k = "_tan_ht"
        t = self.cache.get(k)
        if t is None:
            t = torch.tan(self.theta_ht); self.cache[k] = t
        return t

    @property
    def sin_st(self):
        k = "_sin_st"
        t = self.cache.get(k)
        if t is None:
            t = torch.sin(self.theta_st); self.cache[k] = t
        return t

    @property
    def cos_st(self):
        k = "_cos_st"
        t = self.cache.get(k)
        if t is None:
            t = torch.cos(self.theta_st); self.cache[k] = t
        return t

    @property
    def tan_st(self):
        k = "_tan_st"
        t = self.cache.get(k)
        if t is None:
            t = torch.tan(self.theta_st); self.cache[k] = t
        return t

    # Common junctions / pieces
    @property
    def DTJY(self):
        k = "_DTJY"; t = self.cache.get(k)
        if t is None:
            stack = self.col("Stack")
            htl   = self.col("Head tube length textfield")
            htlx  = self.col("Head tube lower extension2")
            t = stack - (htl - htlx) * self.sin_ht
            self.cache[k] = t
        return t

    @property
    def DTJX(self):
        k = "_DTJX"; t = self.cache.get(k)
        if t is None:
            dt_len = self.col("DT Length")
            t = torch.sqrt(torch.clamp_min(dt_len**2 - self.DTJY**2, 0.0))
            self.cache[k] = t
        return t

    @property
    def TTJX(self):
        k = "_TTJX"; t = self.cache.get(k)
        if t is None:
            htl  = self.col("Head tube length textfield")
            htlx = self.col("Head tube lower extension2")
            htux = self.col("Head tube upper extension2")
            t = self.DTJX - (htl - htlx - htux) * self.cos_ht
            self.cache[k] = t
        return t

    @property
    def TTJY(self):
        k = "_TTJY"; t = self.cache.get(k)
        if t is None:
            stack = self.col("Stack")
            htux  = self.col("Head tube upper extension2")
            t = stack - htux * self.sin_ht
            self.cache[k] = t
        return t

    @property
    def STJX(self):
        k = "_STJX"; t = self.cache.get(k)
        if t is None:
            stl  = self.col("Seat tube length")
            stux = self.col("Seat tube extension2")
            t = (stl - stux) * self.cos_st
            self.cache[k] = t
        return t

    @property
    def STJY(self):
        k = "_STJY"; t = self.cache.get(k)
        if t is None:
            stl  = self.col("Seat tube length")
            stux = self.col("Seat tube extension2")
            t = (stl - stux) * self.sin_st
            self.cache[k] = t
        return t

    @property
    def z_bb(self):
        """sqrt(max(CS^2 - BB^2, 0)) reused in rear-wheel/seat-stay logic."""
        k = "_z_bb"; t = self.cache.get(k)
        if t is None:
            CS = self.col("CS textfield")
            BB = self.col("BB textfield")
            t = torch.sqrt(torch.clamp_min(CS**2 - BB**2, 0.0))
            self.cache[k] = t
        return t

    # Unified front-axle location (relative to BB)
    def front_axle_xy(self):
        """
        Returns (FWX, FBBD), where:
          - FBBD: front axle Y relative to BB (aka BB vertical offset to FW axle)
          - FWX : front axle X relative to BB
        Construction matches both the down-tube and foot clearance checks.
        """
        theta   = self.theta_ht
        DTJY    = self.DTJY
        DTJX    = self.DTJX
        bb_drop = self.col("BB textfield")
        wdf     = self.col("Wheel diameter front")
        wdr     = self.col("Wheel diameter rear")
        fork0r  = self.col("FORK0R")

        FBBD = bb_drop - wdr*0.5 + wdf*0.5
        # Clamp sin(theta) to avoid division by zero for pathological inputs
        s = torch.clamp_min(torch.sin(theta), 1e-9)
        c = torch.cos(theta)
        y = DTJY - FBBD + fork0r * c
        L = y / s
        x_add = L * c + fork0r * torch.sin(theta)
        FWX = DTJX + x_add
        return FWX, FBBD

def construct_tensor_validator(validation_functions: List[ValidationFunction],
                               column_names: List[str]):
    """
    Preflight-check required columns once; build a shared FeatureStore per call;
    call each validator with ctx.
    """
    column_names = list(column_names)
    name_to_idx: Dict[str, int] = {c: i for i, c in enumerate(column_names)}

    # Preflight: ensure all columns exist
    for vf in validation_functions:
        for col in vf.variable_names():
            if col not in name_to_idx:
                raise KeyError(f"Column '{col}' required by '{vf.friendly_name()}' not in provided column_names.")

    all_return_names = [v.friendly_name() for v in validation_functions]

    def validate_tensor(designs: torch.Tensor) -> torch.Tensor:
        n = designs.shape[0]
        v = len(validation_functions)
        out = torch.zeros((n, v), dtype=designs.dtype, device=designs.device)
        ctx = FeatureStore(designs, name_to_idx)
        for i, vf in enumerate(validation_functions):
            res = vf.validate(ctx)
            out[:, i] = res.reshape(-1)
        return out

    return validate_tensor, all_return_names


def construct_dataframe_validator(validation_functions: List[ValidationFunction]):
    """
    Constructs a function that applies multiple validation functions to a Pandas DataFrame of designs.
    
    Parameters:
        validation_functions (List[ValidationFunction]): List of validation function instances.

    Returns:
        A function that takes a Pandas DataFrame of designs and returns a DataFrame of validation results.
    """

    # First, construct the tensor-based validator (this one doesn't need column mapping)
    def validate_dataframe(designs: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the DataFrame to a tensor, applies validation, and converts the result back to a DataFrame.
        
        Parameters:
            designs (pd.DataFrame): A DataFrame where each row represents a design.

        Returns:
            pd.DataFrame: A DataFrame of shape (n, v), where:
                - Rows correspond to designs (original DataFrame index is preserved).
                - Columns correspond to validation function names.
                - Values: 1 indicates invalid, 0 indicates valid.
        """
        # Convert DataFrame to a PyTorch tensor (float32)
        designs_tensor = torch.tensor(designs.to_numpy(), dtype=torch.float32)

        # Use the tensor validator (construct it dynamically based on DataFrame columns)
        tensor_validator, all_return_names = construct_tensor_validator(validation_functions, list(designs.columns))
        results_tensor = tensor_validator(designs_tensor)

        # Convert results back to a DataFrame
        results_df = pd.DataFrame(
            results_tensor.numpy(),  # Convert tensor to NumPy
            columns=all_return_names,
            index=designs.index  # Preserve original index
        )

        return results_df

    return validate_dataframe
