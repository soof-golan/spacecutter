# spacecutter-torch

`spacecutter-torch` is a library for implementing ordinal regression models in PyTorch. The library consists of models and loss functions.

## Installation

```bash
pip install spacecutter-torch
```

## Usage

### Models

Define any PyTorch model you want that generates a single, scalar prediction value. This will be our `predictor` model. This model can then be wrapped with `spacecutter.models.OrdinalLogisticModel` which will convert the output of the `predictor` from a single number to an array of ordinal class probabilities. The following example shows how to do this for a two layer neural network `predictor` for a problem with three ordinal classes.

```python
import torch
from torch import nn

from spacecutter.models import OrdinalLogisticHead


X = torch.tensor([[0.5, 0.1, -0.1],
              [1.0, 0.2, 0.6],
              [-2.0, 0.4, 0.8]]).float()

y = torch.tensor([0, 1, 2]).reshape(-1, 1).long()

num_features = X.shape[1]
num_classes = len(torch.unique(y))

model = nn.Sequential(
    nn.Linear(num_features, num_features),
    nn.ReLU(),
    nn.Linear(num_features, 1),
    OrdinalLogisticHead(num_classes),
)

y_pred = model(X)

print(y_pred)

# tensor([[0.2325, 0.2191, 0.5485],
#         [0.2324, 0.2191, 0.5485],
#         [0.2607, 0.2287, 0.5106]], grad_fn=<CatBackward>)

```

### Training

The following shows how to train the model from the previous section using cumulative link loss:

```python
import torch
from spacecutter.callbacks import AscensionCallback
from spacecutter.losses import CumulativeLinkLoss

def train(model, optimizer, X, y, num_epochs = 10) -> list:
    """
    you can bring your own training loop if you want, but we provide a very simple one here. 
    """
    model.train()
    on_batch_end_callbacks = [AscensionCallback()]
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = train(model, optimizer, X, y)

```

Note that we must add the `AscensionCallback`. This ensures that the ordinal cutpoints stay in ascending order. While ideally this constraint would be factored directly into the model optimization, `spacecutter` currently hacks an SGD-compatible solution by utilizing a post-backwards-pass callback to clip the cutpoint values.
