import torch
from torch import nn

from spacecutter import callbacks
from spacecutter.models import OrdinalLogisticHead


def test_clip_ensures_sorted_cutpoints():
    backbone = nn.Linear(5, 1)
    head = OrdinalLogisticHead(4, init_cutpoints="ordered")
    model = nn.Sequential(backbone, head)
    # model.link.cutpoints = [-1.5, -0.5, 0.5]

    # The following is necessary to be able to manually modify the cutpoints.
    for p in model.parameters():
        p.requires_grad = False
    ascension = callbacks.AscensionCallback()

    # Make cutpoints not in sorted order
    head.link.cutpoints += torch.FloatTensor([0, 5, 0])
    # model.link.cutpoints = [-1.5, 4.5, 0.5]

    # Apply the clipper
    model.apply(ascension)

    assert torch.allclose(head.link.cutpoints.data, torch.FloatTensor([-1.5, 0.5, 0.5]))


def test_margin_is_satisfied():
    backbone = nn.Linear(5, 1)
    head = OrdinalLogisticHead(4, init_cutpoints="ordered")
    model = nn.Sequential(backbone, head)
    # model.link.cutpoints = [-1.5, -0.5, 0.5]

    # The following is necessary to be able to manually modify the cutpoints.
    for p in model.parameters():
        p.requires_grad = False
    ascension = callbacks.AscensionCallback(margin=0.5)

    # Make cutpoints not in sorted order
    head.link.cutpoints += torch.FloatTensor([0, 5, 0])
    # model.link.cutpoints = [-1.5, 4.5, 0.5]

    # Apply the clipper
    model.apply(ascension)

    assert torch.allclose(head.link.cutpoints.data, torch.FloatTensor([-1.5, 0.0, 0.5]))
