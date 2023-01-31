"""
simple Linear implementation
[short description]
[link to the paper]
"""
import torch
from torch import nn

from prescyent.predictor.lightning.module import BaseLightningModule, allow_unbatched


class Linear(nn.Module):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, feature_size, input_size, output_size):
        super(Linear, self).__init__()
        self.feature_size = feature_size
        self.input_size = input_size
        self.output_size = output_size

        self.linear = nn.Linear(input_size * feature_size, output_size * feature_size)

    @allow_unbatched
    def forward(self, x):
        """"""
        # save input shape
        shape = x.shape
        # flatten input
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])
        predictions = self.linear(x)
        # reshape output
        predictions = predictions.view(shape[0], shape[1], shape[2])
        return predictions


class LinearModule(BaseLightningModule):
    """[short description]
       [usage]
       [detail of the implementation]
    """
    def __init__(self, feature_size, input_size, output_size):
        super().__init__()
        self.torch_model = Linear(feature_size, input_size, output_size)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        linear_module = cls(model.feature_size, model.input_size, model.output_size)
        linear_module.torch_model = model
        return linear_module
