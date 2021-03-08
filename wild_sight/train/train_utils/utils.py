import pathlib
from typing import Any, Dict, List

import torch

from third_party.adabelief import ranger_adabelief
from third_party.adabelief import adabelief
from third_party import ranger


def save_model(model: torch.nn.Module, save_path: pathlib.Path) -> None:
    torch.save(model.state_dict(), save_path)


def create_optimizer(optim_cfg: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Take in optimizer config and create the optimizer for training.

    Args:
        optim_cfg: The parameters for the optimizer generation.
        model: The model to create an optimizer for. We need to read this model's
            parameters.

    Returns:
        An optimizer for the model.
    """

    name = optim_cfg.get("type", "sgd")  # Default to SGD, most reliable.
    if name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            momentum=float(optim_cfg["momentum"]),
            weight_decay=0,
            nesterov=True,
        )
    elif name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            momentum=float(optim_cfg["momentum"]),
            weight_decay=0,
        )
    elif name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            weight_decay=0,
        )
    elif name.lower() == "ranger_adabelief":
        optimizer = ranger_adabelief.RangerAdaBelief(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            weight_decay=0,
        )
    elif name.lower() == "adabelief":
        optimizer = adabelief.AdaBelief(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            weight_decay=0,
        )
    elif name.lower() == "adam":
        optimizer = torch.optim.Adam(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            weight_decay=0,
        )
    elif name.lower() == "ranger":
        optimizer = ranger.Ranger(
            add_weight_decay(model, float(optim_cfg["weight_decay"])),
            lr=float(optim_cfg["lr"]),
            weight_decay=0,
        )
    else:
        raise ValueError(f"Improper optimizer supplied: {name}.")

    return optimizer


def add_weight_decay(
    model: torch.nn.Module, weight_decay: float
) -> List[Dict[str, Any]]:
    """Add weight decay to only the convolutional kernels. Do not add weight decay
    to biases and BN. TensorFlow does this automatically, but for some reason this is
    the PyTorch default.

    Args:
        model: The model where the weight decay is applied.
        weight_decay: How much decay to apply.

    Returns:
        A list of dictionaries encapsulating which params need weight decay.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


# TODO(alex): Unwrap DDP models.
def unwrap_model(model: Any) -> torch.nn.Module:
    ...
