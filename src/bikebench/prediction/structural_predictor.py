from bikebench.prediction.prediction_utils import DNN
def get_structural_model(dropout_on = False):
    if dropout_on:
        model = DNN(39, layer_sizes=(128, 256), dropout_rate = 0.2, output_dim = 6, classification=False)
    else:
        model = DNN(39, layer_sizes=(128, 256), dropout_rate = 0.0, output_dim = 6, classification=False)
    return model
