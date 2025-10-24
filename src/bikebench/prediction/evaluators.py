import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from bikebench.data_loading.data_loading import (
    load_validity_test_oh,
    load_structure_test_oh,
    load_aero_test,
    load_usability_cont_test,
    load_aesthetics_test,
)

def evaluate_validity(model, preprocessing_fn, device="cpu"):
    X_test, Y_test = load_validity_test_oh()
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    predictions = predictions >= 0.5
    return f1_score(Y_test, predictions)

def evaluate_structure(model, preprocessing_fn, device="cpu"):
    X_test, Y_test = load_structure_test_oh()
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_aero(model, preprocessing_fn, device="cpu"):
    X_test, Y_test = load_aero_test()
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()
    return r2_score(Y_test, predictions)

def evaluate_usability(model, preprocessing_fn, device="cpu", target_type='cont'):
    if target_type == 'cont':
        X_test, Y_test = load_usability_cont_test()
        X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
        X_test_tensor = preprocessing_fn(X_test_tensor)
        predictions = model(X_test_tensor).detach().cpu().numpy()
        return r2_score(Y_test, predictions)
    elif target_type == 'binary':
        from bikebench.resource_utils import datasets_path
        X_test = pd.read_csv(datasets_path('Predictive_Modeling_Datasets/usability_binary_X_test.tab'), index_col=0, sep='\t')
        Y_test = pd.read_csv(datasets_path('Predictive_Modeling_Datasets/usability_binary_Y_test.tab'), index_col=0, sep='\t')
        X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
        X_test_tensor = preprocessing_fn(X_test_tensor)
        predictions = model(X_test_tensor).detach().cpu().numpy()
        predictions = predictions >= 0.5
        return f1_score(Y_test, predictions)

def evaluate_aesthetics(model, preprocessing_fn, device="cpu"):
    X_test, Y_test = load_aesthetics_test()
    X_test_tensor = torch.tensor(X_test.values.astype(float), dtype=torch.float32).to(device)
    X_test_tensor = preprocessing_fn(X_test_tensor)
    predictions = model(X_test_tensor).detach().cpu().numpy()

    cosine_sim = cosine_similarity(predictions, Y_test)
    diag = np.diag(cosine_sim)
    worse_than_diag = cosine_sim <= diag[:, np.newaxis]
    matchperc = np.mean(worse_than_diag)
    print(f"Predicted embedding more similar to GT than : {100 * matchperc:.2f}% of test set designs, on average.")

    return mean_squared_error(Y_test, predictions)
