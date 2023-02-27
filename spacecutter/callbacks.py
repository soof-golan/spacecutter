import torch
from torch import nn

from spacecutter.models import LogisticCumulativeLink


class AscensionCallback:
    """
    Ensure that each cutpoint is ordered in ascending value.
    e.g.

    .. < cutpoint[i - 1] < cutpoint[i] < cutpoint[i + 1] < ...

    This is done by clipping the cutpoint values at the end of a batch gradient
    update. By no means is this an efficient way to do things, but it works out
    of the box with stochastic gradient descent.

    Parameters
    ----------
    margin : float, (default=0.0)
        The minimum value between any two adjacent cutpoints.
        e.g. enforce that cutpoint[i - 1] + margin < cutpoint[i]
    min_val : float, (default=-1e6)
        Minimum value that the smallest cutpoint may take.
    """

    def __init__(self, margin: float = 0.0, min_val: float = -1.0e6) -> None:
        self.margin = margin
        self.min_val = min_val

    def clip_cutpoints(self, module: nn.Module) -> None:
        if isinstance(module, LogisticCumulativeLink):
            assert hasattr(
                module, "cutpoints"
            ), "Module must have a cutpoints attribute."
            assert isinstance(
                module.cutpoints, nn.Parameter
            ), "Module.cutpoints must be a torch.nn.Parameter."
            cutpoints: torch.Tensor = module.cutpoints.data
            min_val = torch.tensor(self.min_val).to(cutpoints.device)
            max_val = cutpoints[1:] - self.margin
            cutpoints[:-1].clamp_(min_val, max_val)

    def __call__(self, *args, **kwargs):
        return self.clip_cutpoints(*args, **kwargs)
