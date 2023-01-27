"""
simple LSTM implementation
[short description]
[link to the paper]
"""
import torch
import pytorch_lightning as pl
from torch import nn
import torch.optim as optim

from prescyent.evaluator.metrics import get_ade, get_fde


class LSTM(nn.Module):
    """
    feature_size - The number of dimensions to predict in parrallel
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size
        self.num_layers = num_layers

        # input -> output
        # batch_first=False : (seq_len, batch_size, features) -> (seq_len, batch_size, hidden_size).
        # batch_first=True: (batch_size, seq_len, features) -> (batch_size, seq_len, hidden_size).
        # unbatched: (seq_length, feature_size) -> (seq_len, hidden_size)
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0)

        # we use the sequence of all the hidden state to predict the output
        # linear expect [batch_size, *, nb_features]
        self.linear = nn.Linear(hidden_size, feature_size)

    @property
    def input_size(self):
        return self.feature_size

    def forward(self, x):
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        unbatched = len(x.shape) == 2
        if unbatched:
            x = torch.unsqueeze(x, dim=0)
        lstm_out, hidden = self.lstm(x)
        predictions = self.linear(lstm_out)  # self.linear(lstm_out.view(len(x), -1))
        if unbatched:
            predictions = torch.squeeze(predictions, dim=0)
        return predictions


class LSTMModule(pl.LightningModule):
    """[short description]
       [usage]
       [detail of the implementation]
    """
    def __init__(self, feature_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.torch_model = LSTM(feature_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    @classmethod
    def load_from_state_dict(cls, path: str):
        """Retrieve model infos from state dict"""
        raise NotImplementedError("TODO ?")

    @classmethod
    def load_from_binary(cls, path: str):
        """Retrieve model infos from torch binary"""
        model = torch.load(path)
        lstm_module = cls(model.input_size, model.hidden_size, model.output_size, model.num_layers)
        lstm_module.torch_model = model
        return lstm_module

    def save(self, save_path: str):
        """Export model to state_dict and torch binary"""
        torch.save(self.torch_model.state_dict(), save_path / "state_dict.pt")
        torch.save(self.torch_model, save_path / "model.pb")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_metrics(self, batch, prefix: str = ""):
        """get loss and accuracy metrics from batch"""
        sample, truth = batch
        pred = self.torch_model(sample)
        loss = self.criterion(pred, truth)
        ade = get_ade(truth, pred)
        fde = get_fde(truth, pred)
        self.log(f"{prefix}/loss", loss)
        return {"loss": loss, "ADE": ade, "FDE": fde}

    def log_accuracy(self, outputs, prefix: str = ""):
        """log accuracy metrics from epoch"""
        mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
        fde = torch.stack([x["FDE"] for x in outputs]).mean()
        ade = torch.stack([x["ADE"] for x in outputs]).mean()
        self.logger.experiment.add_scalar(f"{prefix}/epoch_loss", mean_loss, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/FDE", fde, self.current_epoch)
        self.logger.experiment.add_scalar(f"{prefix}/ADE", ade, self.current_epoch)

    def training_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Train")

    def test_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Test")

    def validation_step(self, batch, batch_idx):
        return self.get_metrics(batch, "Val")

    def test_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Test")

    def training_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Train")

    def validation_epoch_end(self, outputs):
        self.log_accuracy(outputs, "Val")
