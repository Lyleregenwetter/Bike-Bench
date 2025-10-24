from abc import abstractmethod, ABC
from typing import List
import torch
import pandas as pd
import numpy as np
import pygmo as pg
from sklearn.preprocessing import StandardScaler
import os
from bikebench.conditioning import conditioning
from bikebench.resource_utils import resource_path
from bikebench.data_loading import data_loading
from bikebench.design_evaluation.design_evaluation import construct_tensor_evaluator, EvaluationFunction
from bikebench.transformation import one_hot_encoding


class ScoringFunction(ABC):
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def return_names(self) -> List[str]:
        pass

    @abstractmethod
    def evaluate(self, designs: torch.Tensor, conditioning: dict = {}) -> torch.Tensor:
        pass



def get_ref_point(evaluator, objective_names, eval_names, reduction = "max", device = "cpu"):
    if reduction=="max":
        path = resource_path("misc/ref_point.csv")
    elif reduction=="meanabs":
        path = resource_path("misc/default_weights.csv")
    else:
        raise ValueError("Invalid reduction method. Use 'max' or 'meanabs'.")
    if not os.path.exists(path):
        #throw error if file does not exist
        raise FileNotFoundError(f"Reference point file not found at {path}")

        # ref_point_df = recompute_ref_point(evaluator, eval_names, path, reduction, device)
    else:
        ref_point_df = pd.read_csv(path, index_col=0, header=None)
        ref_point_columns = ref_point_df.index.values
        if not np.all(np.isin(objective_names, ref_point_columns)):
            raise ValueError("Reference point does not include all objective names. Please provide a valid reference point file.")

            # print("Reference point does not include all objective names. Recomputing...")
            # ref_point_df = recompute_ref_point(evaluator, eval_names, path, reduction, device)
    ref_point_df = ref_point_df.loc[objective_names]
    ref_point = ref_point_df.values.flatten()
    return ref_point

class Hypervolume(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Design Quality ↑ (HV)"]

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        # 1) keep only feasible points
        feas_mask = np.all(constraint_scores <= 0, axis=1)
        objs = objective_scores[feas_mask]
        if objs.size == 0:
            return 0.0

        # 2) drop invalid (non-finite) rows to save time
        valid_rows = np.all(np.isfinite(objs), axis=1)
        objs = objs[valid_rows]
        if objs.size == 0:
            return 0.0

        # 3) clip each objective to its component-wise upper bound (reference)
        ref = np.asarray(obj_ref_point, dtype=float)
        objs = np.minimum(objs, ref)

        # 4) normalize to [0,1] for MINIMIZATION (0 = ideal, 1 = at ref)
        norm = objs / ref

        # 5) compute HV in unit cube with ref point = 1...1 (minimization)
        hv = pg.hypervolume(norm)
        hv_value = float(hv.compute(ref_point=np.ones_like(ref)))
        return hv_value


class BinaryValidity(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return ["Binary Validity ↑"]
    
    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        return np.mean(np.all(constraint_scores <=0, axis=1))


class MMD(ScoringFunction): 

    def __init__(self, batch_size = 1024, gamma=None):
        super().__init__()
        raw_ref  = data_loading.load_bike_bench_test().values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.scaler.fit(raw_ref)
        self.reference_designs = self.scaler.transform(raw_ref)

        self.batch_size = batch_size

        if gamma is None:
            gamma = self.compute_gamma(self.reference_designs)
        self.gamma = gamma
        

    def return_names(self) -> List[str]:
        return ["Sim. to Data ↓ (MMD)"]

    def compute_gamma(self, ref: np.ndarray) -> float:
        dists = np.sum((ref[:, None, :] - ref[None, :, :])**2, axis=2)
        med = np.median(dists)
        return 1.0 / (2 * med) if med > 0 else 1.0

    def rbf_kernel_sum(self, A: np.ndarray, B: np.ndarray, gamma: float) -> float:
        """
        Compute sum_{i,j} exp(-gamma * ||A[i] - B[j]||^2)
        by blocking through rows of A and B in chunks of size batch_size.
        """
        total = 0.0
        for i in range(0, A.shape[0], self.batch_size):
            Ai = A[i : i + self.batch_size]
            for j in range(0, B.shape[0], self.batch_size):
                Bj = B[j : j + self.batch_size]
                # compute squared‐distances of shape (len(Ai), len(Bj))
                D2 = np.sum((Ai[:, None, :] - Bj[None, :, :])**2, axis=2)
                total += np.exp(-gamma * D2).sum()
        return total

    def mmd(self, gen: np.ndarray, ref: np.ndarray) -> float:
        K_GG = self.rbf_kernel_sum(gen, gen, self.gamma)
        K_RR = self.rbf_kernel_sum(ref, ref, self.gamma)
        K_GR = self.rbf_kernel_sum(gen, ref, self.gamma)

        n, m = gen.shape[0], ref.shape[0]
        return (K_GG / (n * n)) + (K_RR / (m * m)) - (2 * K_GR / (n * m))

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        scaled_designs = self.scaler.transform(designs)
        return self.mmd(scaled_designs, self.reference_designs)
    
class AverageConstraintViolation(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Constraint Violation ↓"] #counts average number of violated constraints per design
        validity_boolean = constraint_scores > 0
        return np.mean(np.sum(validity_boolean, axis=1))
    
class AverageNovelty(ScoringFunction):
    def __init__(self):
        super().__init__()
        raw_ref  = data_loading.load_bike_bench_test().values.astype(np.float32)
        
        self.scaler = StandardScaler()
        self.scaler.fit(raw_ref)
        self.reference_designs = self.scaler.transform(raw_ref)

    def return_names(self) -> List[str]:
        return ["Novelty ↑"]

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        scaled_designs = self.scaler.transform(designs)
        dists = np.sqrt(np.sum((scaled_designs[:, None, :] - self.reference_designs[None, :, :])**2, axis=2))
        min_dists = np.min(dists, axis=1)
        return np.mean(min_dists)
        


class DPPDiversity(ScoringFunction):
    def __init__(self):
        super().__init__()
        # Fit scaler on the same reference data as AverageNovelty
        raw_ref = data_loading.load_bike_bench_test().values.astype(np.float32)
        self.scaler = StandardScaler()
        self.scaler.fit(raw_ref)
        # Not strictly needed later, but keeping for parity / potential reuse
        self.reference_designs = self.scaler.transform(raw_ref)

    def return_names(self) -> List[str]:
        # Lower is better (more diverse), matching your wrapper's behavior
        return ["Diversity ↓ (DPP)"]

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        #deduplicate designs
        designs = np.unique(designs, axis=0)

        # Convert to numpy and scale like AverageNovelty
        X = np.asarray(designs, dtype=np.float64)
        n = X.shape[0]
        if n <= 1:
            return 0.0

        Xs = self.scaler.transform(X).astype(np.float64)  # float64 for eig stability

        # Pairwise squared distances via r - 2XX^T + r^T
        r = np.sum(Xs * Xs, axis=1, keepdims=True)            # (n,1)
        D = r - 2.0 * (Xs @ Xs.T) + r.T                       # (n,n), squared Euclidean distances

        D = D / X.shape[1] / X.shape[1]  # normalize by dimension

        # Quartic RBF (as in your wrapper): exp(-0.5 * D^2)
        # Add tiny jitter to help PD-ness when points are near-duplicates
        S = np.exp(-0.5 * np.square(D))
        np.fill_diagonal(S, S.diagonal() + 1e-12)

        # Eigenvalues and negative mean log-eigenvalue
        try:
            eig_val, _ = np.linalg.eigh(S)
        except Exception:
            # Match wrapper fallback semantics
            eig_val = np.ones(n, dtype=np.float64)

        eig_val = np.maximum(eig_val, 1e-7)
        loss = -float(np.mean(np.log(eig_val)))
        return loss



class MinimumObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Individual Min Objective Score ↓: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * obj_ref_point
        minscores = np.min(valid_objective_scores, axis=0)
        return minscores
    
class MeanObjective(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Individual Mean Objective Score ↓: {name}" for name in objective_names]
        validity_mask = np.all(constraint_scores <= 0, axis=1)
        valid_objective_scores = objective_scores[validity_mask]
        if valid_objective_scores.size == 0:
            return np.ones_like(objective_scores[0]) * obj_ref_point
        meanscores = np.mean(valid_objective_scores, axis=0)
        return meanscores
    
class ConstraintViolationRate(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Individual Constraint Violation Rate ↓: {name}" for name in constraint_names]
        validity_boolean = constraint_scores > 0
        return np.mean(validity_boolean, axis=0)
    
class MeanConstraintViolationMagnitude(ScoringFunction):
    def __init__(self):
        super().__init__()

    def return_names(self) -> List[str]:
        return self.names

    def evaluate(self, designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point):
        self.names = [f"Individual Mean Constraint Violation Magnitude ↓: {name}" for name in constraint_names]
        constraint_scores = np.clip(constraint_scores, a_min=0, a_max=None)
        meanscores = np.mean(constraint_scores, axis=0)
        return meanscores

def construct_scorer(scoring_functions: List[ScoringFunction], evaluation_functions: List[EvaluationFunction], column_names: List[str], device: str = "cpu") -> callable:
    evaluator, requirement_names, is_objective, is_conditional = construct_tensor_evaluator(evaluation_functions, column_names, device=device)
    requirement_names = np.array(requirement_names)
    isobjective = torch.tensor(is_objective, dtype=bool)
    objective_names = requirement_names[isobjective]
    constraint_names = requirement_names[~isobjective]

    obj_ref_point = get_ref_point(evaluator, objective_names, requirement_names, "max", device) #1D numpy array
    def scorer(designs: torch.Tensor, condition: dict = {}, preevaluated_scores = None) -> pd.Series:
        device = designs.device
        score_names = []
        scores = []
        if preevaluated_scores is None:
            designs = designs.detach().cpu().numpy()
            designs_df = pd.DataFrame(designs, columns=column_names)
            designs_reverse_oh = one_hot_encoding.decode_to_mixed(designs_df)
            designs_continuous_mapped = one_hot_encoding.encode_to_continuous(designs_reverse_oh)
            designs_mapped_tens = torch.tensor(designs_continuous_mapped.values, dtype=torch.float32).to(device)
            evaluation_scores = evaluator(designs_mapped_tens, condition)
        else:
            evaluation_scores = preevaluated_scores
        objective_scores = evaluation_scores[:, isobjective].detach().cpu().numpy()
        ref_point_exp = np.expand_dims(obj_ref_point, axis=0)
        ref_point_exp = np.repeat(ref_point_exp, objective_scores.shape[0], axis=0)
        objective_scores[np.isnan(objective_scores)] = ref_point_exp[np.isnan(objective_scores)]
        constraint_scores = evaluation_scores[:, ~isobjective].detach().cpu().numpy()
        
        for scoring_function in scoring_functions:    
            raw = scoring_function.evaluate(designs, objective_scores, constraint_scores, objective_names, constraint_names, obj_ref_point)

            arr = np.atleast_1d(raw)

            names = scoring_function.return_names()

            for n, val in zip(names, arr):
                score_names.append(n)
                scores.append(val)
        scores = np.array(scores)
        scores = pd.Series(scores, index=score_names)
        return scores
    return scorer

MainScores: List[ScoringFunction] = [
    Hypervolume(),
    AverageConstraintViolation(),
    MMD(),
    AverageNovelty(),
    BinaryValidity(),
    DPPDiversity(),
]

DetailedScores: List[ScoringFunction] = [
    MinimumObjective(),
    MeanObjective(),
    ConstraintViolationRate(),
    MeanConstraintViolationMagnitude(),
]



