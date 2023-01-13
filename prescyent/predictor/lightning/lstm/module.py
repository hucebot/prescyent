"""
simple LSTM implementation
[short description]
[link to the paper]
"""
import torch
import pytorch_lightning as pl
from torch import nn
import torch.optim as optim


class LSTM(nn.Module):
    """
    input_size - will be 1 in this example
                since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediction_periods input to get_x_y_pairs
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        # input -> output
        # batch_first=False : (seq_len, batch_size, features) -> (seq_len, batch_size, hidden_size).
        # batch_first=True: (batch_size, seq_len, features) -> (batch_size, seq_len, hidden_size).
        # unbatched: (seq_length, input_size) -> (seq_len, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=0)

        # we use the sequence of all the hidden state to predict the output
        # linear expect [batch_size, *, nb_features]
        self.linear = nn.Linear(hidden_size * output_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        if hidden is None:
            self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x),
                           torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x))
        else:
            self.hidden = hidden
        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        shape = (batch_size, seq_length, 1)  # we don't have the 1 in the dataset for now
        lstm_out, self.hidden = self.lstm(x.view(shape), self.hidden)
        # the linear predictor can use the list of all the hidden states if it wants
        # (and not only the end)

        # this might be useful because we want to predict all the n next steps
        # we can't use view here because the stride does not match?
        out = lstm_out.reshape(batch_size, -1)
        predictions = self.linear(out)  # self.linear(lstm_out.view(len(x), -1))
        return predictions, self.hidden


class LSTMModule(pl.LightningModule):
    """[short description]
       [usage]
       [detail of the implementation]
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.torch_model = LSTM(input_size, hidden_size, output_size, num_layers)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def save(self, save_path: str):
        torch.save(self.torch_model.state_dict(), save_path / "state_dict.pt")
        torch.save(self.torch_model, save_path / "model.pb")

    def compute_loss(self, batch):
        """get loss from truth and pred"""
        sample, truth = batch
        pred, _ = self.torch_model(sample)
        loss = self.criterion(pred, truth)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # for test loop of the Trainer
        loss = self.compute_loss(batch)
        self.log("test_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # for test loop of the Trainer
        loss = self.compute_loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
