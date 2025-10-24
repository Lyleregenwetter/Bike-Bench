from bikebench.prediction.prediction_utils import DNN
def get_validity_model(dropout_on = False):
    if dropout_on:
        model = DNN(39, layer_sizes=(128, 128), dropout_rate = 0.6, classification=True)
    else:
        model = DNN(39, layer_sizes=(128, 128), dropout_rate = 0.0, classification=True)
    return model
