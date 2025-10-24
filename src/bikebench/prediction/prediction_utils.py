import torch
import dill
from torch import nn
from bikebench.resource_utils import models_and_scalers_path


class TorchStandardScaler(nn.Module):
    def __init__(self):
        super().__init__()
        # these will be set in .fit()
        self.register_buffer('mean', torch.tensor([]))
        self.register_buffer('std', torch.tensor([]))
        self.fitted = False

    def fit(self, x: torch.Tensor):
        """
        Compute per‑feature mean and std from a [N, F]-shaped tensor.
        """
        # flatten any extra dims into the batch
        N = x.shape[0]
        feats = x.view(N, -1) if x.dim() > 2 else x
        self.mean = feats.mean(dim=0)
        self.std  = feats.std(dim=0, unbiased=False).clamp(min=1e-6)
        self.fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet")
        # preserve any extra dimensions beyond the feature-dim
        original_shape = x.shape
        N = x.shape[0]
        feats = x.view(N, -1) if x.dim() > 2 else x
        scaled = (feats - self.mean) / self.std
        return scaled.view(original_shape)

    # alias
    transform = forward

class Preprocessor(nn.Module):
    def __init__(self, scaler_path, preprocess_fn, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.scaler: TorchStandardScaler = torch.load(scaler_path, map_location=self.device, weights_only=False)
        self.scaler.to(self.device)
        self.preprocess_fn = preprocess_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preprocess_fn:
            x = self.preprocess_fn(x, self.device)
        return self.scaler(x)

    __call__ = forward
    
class DNN(nn.Module):
    def __init__(self, input_dim, layer_sizes=[256, 128], output_dim=1, dropout_rate=0.2, classification=False):
        super(DNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = size
        
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.classification = classification

    def forward(self, x):
        x = self.network(x)
        if self.classification:
            x = torch.sigmoid(x)
        return x