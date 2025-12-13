# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import torch
import numpy as np
import torch.nn as nn

from torch import Tensor


def ff_module(
    node_size,
    num_layers,
    input_size,
    with_bias=True,
    output_size=None,
    activation=nn.SiLU(),
    final_activation=None,
):
    layers = []
    for idl in range(num_layers):
        if idl == 0:
            layers.append(nn.Linear(input_size, node_size, bias=with_bias))
        else:
            layers.append(nn.Linear(node_size, node_size, bias=with_bias))
        layers.append(activation)
    if output_size is not None:
        layers.append(nn.Linear(node_size, output_size, bias=False))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


def scalar_product(x, y, keepdim: bool = True):
    return (x * y).sum(dim=-1, keepdim=keepdim)


def pdist_sq_unsafe(A: Tensor):
    A_norm = torch.square(A).sum(dim=-1, keepdim=True)
    return A_norm - 2 * torch.bmm(A, A.permute(0, 2, 1)) + A_norm.transpose(2, 1)


def cdist(A: Tensor, B: Tensor):
    A_norm = torch.square(A).sum(dim=-1, keepdim=True)
    B_norm = torch.square(B).sum(dim=-1, keepdim=True).transpose(2, 1)
    return torch.sqrt(
        torch.clip(A_norm - 2 * torch.bmm(A, B.permute(0, 2, 1)) + B_norm, 0.0)
    )


def detrace(RxR):
    diagonal = torch.tile(
        RxR.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True), (1, 3)
    )
    return RxR - torch.diag_embed(diagonal)


def build_Rx2(Rx1):
    return detrace(Rx1.unsqueeze(-1) * Rx1.unsqueeze(-2))


def write_xyz(coords, symbols, file_name="test.xyz"):
    num_atoms = len(symbols)
    assert len(coords) == num_atoms
    with open(file_name, "w") as file:
        file.write(str(num_atoms) + "\n")
        file.write("\n")
        for ida in range(num_atoms):
            file.write(
                symbols[ida]
                + " "
                + str(coords[ida][0])
                + " "
                + str(coords[ida][1])
                + " "
                + str(coords[ida][2])
                + "\n"
            )
    return file_name
