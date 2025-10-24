# noinspection PyPackageRequirements
import __main__

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from bikebench.embedding.ordered_columns import ORDERED_COLUMNS

from bikebench.resource_utils import resource_path


def _get_pickled_scaler() -> StandardScaler:
    with open(resource_path("scaler.pk"), "rb") as file:
        return dill.load(file)


def _load_scaled():
    model = _ResidualNetwork(96, 512, 256, 2, 3)
    model.load_state_dict(torch.load(_SCALED_FUNCTION_PATH, map_location=_DEVICE))
    model.eval()
    return model


class _ResidualBlock(nn.Module):
    def __init__(self, input_size, layer_size, num_layers):
        super(_ResidualBlock, self).__init__()
        self.layers = self._make_layers(input_size, layer_size, num_layers)

    def _make_layers(self, input_size, layer_size, num_layers):
        layers = [nn.Linear(input_size, layer_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(layer_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.layers(x)
        total = out + residual
        return total


class _ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_size, layers_per_block, num_blocks):
        super(_ResidualNetwork, self).__init__()
        self.initial_layer = nn.Linear(input_size, layer_size)
        self.blocks = self._make_blocks(layer_size, layers_per_block, num_blocks)
        self.final_layer = nn.Linear(layer_size, output_size)

    def _make_blocks(self, layer_size, layers_per_block, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(_ResidualBlock(layer_size, layer_size, layers_per_block))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.initial_layer(x)
        out = self.blocks(out)
        out = self.final_layer(out)
        return out


__main__.ResidualNetwork = _ResidualNetwork
__main__.ResidualBlock = _ResidualBlock

_DEVICE = torch.device('cpu')
_SCALED_FUNCTION_PATH = resource_path("model_small.pt")
_SCALED_FUNCTION = _load_scaled()
_SCALER = _get_pickled_scaler()


class EmbeddingPredictor:

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self._predict(self._get_scaled(x), _SCALED_FUNCTION)

    def _predict(self, x_data: pd.DataFrame, prediction_function: callable) -> np.ndarray:
        tensor = torch.tensor(x_data.values, dtype=torch.float32)
        result_tensor = prediction_function(tensor).cpu()
        return result_tensor.detach().numpy()

    def _get_ordered(self, x: pd.DataFrame):
        return pd.DataFrame(x, columns=ORDERED_COLUMNS)

    def _get_scaled(self, x):
        ordered = self._get_ordered(x)
        return pd.DataFrame(_SCALER.transform(ordered), columns=ORDERED_COLUMNS)
