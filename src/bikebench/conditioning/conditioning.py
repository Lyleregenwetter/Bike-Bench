import numpy as np
import pandas as pd
import torch

from bikebench.data_loading import data_loading
from bikebench.resource_utils import datasets_path

# ----------------------------
# Lazy CPU caches (per split)
# ----------------------------
_RIDER_CPU = {"train": None, "test": None}   # torch.FloatTensor on CPU
_EMBED_CPU = {"train": None, "test": None}   # torch.FloatTensor on CPU
_TEXT_CACHE = {"train": None, "test": None}  # list[str]

RIDER_COLS = [
    "upper_leg","lower_leg","arm_length",
    "torso_length","neck_and_head_length","torso_width"
]

# ----------------------------------
# Per-device caches (by split)
# cache["rider"]["train"][device] -> tensor on that device
# ----------------------------------
_DEVICE_CACHE = {
    "rider": {"train": {}, "test": {}},
    "embed": {"train": {}, "test": {}},
}

# ---------- internal helpers ----------

def _ensure_rider_cpu(split: str):
    if _RIDER_CPU[split] is None:
        if split == "train":
            df, _ = data_loading.load_aero_train()
        elif split == "test":
            df, _ = data_loading.load_aero_test()
        else:
            raise ValueError("split must be 'train' or 'test'")
        _RIDER_CPU[split] = torch.tensor(df[RIDER_COLS].values, dtype=torch.float32)
    return _RIDER_CPU[split]

def _ensure_embed_cpu(split: str):
    if _EMBED_CPU[split] is None:
        if split == "train":
            emb = np.load(datasets_path("Conditioning/emb_train.npy"))
            cpu = torch.tensor(emb, dtype=torch.float32)
        elif split == "test":
            emb = np.load(datasets_path("Conditioning/emb_test.npy"))
            cpu = torch.tensor(emb, dtype=torch.float32)
        else:
            raise ValueError("split must be 'train' or 'test'")
        _EMBED_CPU[split] = cpu
    return _EMBED_CPU[split]

def _ensure_text(split: str):
    if _TEXT_CACHE[split] is None:
        if split == "test":
            p = datasets_path("Conditioning/text_test.txt")
        elif split == "train":
            p = datasets_path("Conditioning/text_train.txt")
        else:
            raise ValueError("split must be 'train' or 'test'")
        with open(p, "r") as f:
            _TEXT_CACHE[split] = [line.strip() for line in f.readlines()]
    return _TEXT_CACHE[split]

def _to_device_cached(cpu_tensor: torch.Tensor, cache_dict: dict, device: torch.device = None):
    if device is None:
        return cpu_tensor
    if device not in cache_dict:
        cache_dict[device] = cpu_tensor.to(device, non_blocking=True)
    return cache_dict[device]

def _get_rider_tensor(split: str, device: torch.device = None):
    cpu = _ensure_rider_cpu(split)
    return _to_device_cached(cpu, _DEVICE_CACHE["rider"][split], device)

def _get_embed_tensor(split: str, device: torch.device = None):
    cpu = _ensure_embed_cpu(split)
    return _to_device_cached(cpu, _DEVICE_CACHE["embed"][split], device)

def get_indices(N, num_samples, randomize=False):
    if randomize:
        idx = torch.randint(0, N, (num_samples,))
    else:
        reps = num_samples // N + 1
        idx = torch.arange(N).repeat(reps)[:num_samples]
    return idx

def sample_riders(num_samples: int, split="test",
                  randomize=False, device: torch.device = None):

    if split=="test" and randomize:
        print("Warning: Randomizing order of test data when benchmark expects fixed order!")
    data = _get_rider_tensor(split, device)
    N = data.size(0)
    idx = get_indices(N, num_samples, randomize)
    idx.to(device)
    return data[idx]

def sample_embedding(num_samples: int, split="test", randomize = False, device: torch.device = None):
    #warn if split is test and randomize is true
    if split=="test" and randomize:
        print("Warning: Randomizing order of test data when benchmark expects fixed order!")

    data = _get_embed_tensor(split, device)
    N = data.size(0)
    idx = get_indices(N, num_samples, randomize)
    idx.to(device)
    return data[idx]

def sample_use_case(num_samples: int, split=None, randomize = False, device: torch.device = None):
    if split=="test" and randomize:
        print("Warning: Randomizing order of test data when benchmark expects fixed order!")
    onehot = torch.eye(3, dtype=torch.float32)
    idx = get_indices(3, num_samples, randomize)
    idx.to(device)
    return onehot[idx]

def sample_text(num_samples, split="test", randomize=False):
    text_data = _ensure_text(split)
    if not text_data:
        return []
    N = len(text_data)
    idx = get_indices(N, num_samples, randomize)
    return [text_data[i] for i in idx]