from bikebench.prediction.prediction_utils import DNN
def get_usability_model(dropout_on = False):
    if dropout_on:
        model = DNN(3, layer_sizes = [160, 512], dropout_rate = 0.1404)
    else:
        model = DNN(3, layer_sizes = [160, 512], dropout_rate = 0.0)
    return model