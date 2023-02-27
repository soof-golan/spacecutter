from typing import Any, Callable, List

import torch
from torch import nn

from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import OrdinalLogisticHead

SEED = 666


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 10,
) -> List[float]:
    """
    Train the model for a given number of epochs.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    optimizer: torch.optim.Optimizer
        The optimizer to use.
    X: torch.Tensor
        The features.
    y: torch.Tensor
        The targets.
    num_epochs: int
        The number of epochs to train for.

    Returns
    -------
    losses: List[float]
        The loss on each epoch.
    """

    on_batch_end_callbacks: List[Callable[[nn.Module], Any]] = [AscensionCallback()]
    loss_fn = CumulativeLinkLoss()
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.no_grad():
            for callback in on_batch_end_callbacks:
                model.apply(callback)
    return losses


def is_descending(values: List[float]) -> bool:
    """
    Check if the values are in descending order.

    Parameters
    ----------
    values: List[float]
        The values to check.

    Returns
    -------
    is_descending: bool
        True if the values are in descending order, False otherwise.
    """
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def test_loss_lowers_on_each_epoch():
    torch.manual_seed(SEED)
    num_classes = 5
    num_features = 5
    size = 200
    y = torch.randint(0, num_classes, (size, 1), dtype=torch.long)
    X = torch.rand((size, num_features))

    model = nn.Sequential(
        nn.Linear(num_features, num_features),
        nn.ReLU(),
        nn.Linear(num_features, 1),
        OrdinalLogisticHead(num_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = train(model, optimizer, X, y)
    assert is_descending(losses), "Loss lowers on each epoch"
