"""LSTM forecaster: reads a window of past load values and predicts one future value."""
import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """`windows` has shape (batch, lookback). Returns shape (batch,)."""
        output, _ = self.lstm(windows.unsqueeze(-1))
        return self.head(output[:, -1, :]).squeeze(-1)


def train_epochs(
    model: LSTMForecaster,
    inputs: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    generator: torch.Generator,
) -> None:
    """Train `model` in place with Adam and mean squared error."""
    x = torch.as_tensor(inputs, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            optimiser.step()


def predict(model: LSTMForecaster, inputs: np.ndarray) -> np.ndarray:
    """Return predictions as a numpy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(inputs, dtype=torch.float32)).numpy().astype(float)
