# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Optional
import time
import yaml
from datastructures_calibration import Graph


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

def load_parameters(filename: str):
    file = open(filename, "r")
    PARAMETERS = yaml.load(file, yaml.Loader)
    PARAMETERS["time"] = int(time.time())
    PARAMETERS["device"] = torch.device(PARAMETERS["device_name"])
    if PARAMETERS["dtype_name"] == "float32":
        torch.set_default_dtype(torch.float32)
        PARAMETERS["dtype"] = torch.float
    else:
        torch.set_default_dtype(torch.float64)
        PARAMETERS["dtype"] = torch.double
    return PARAMETERS

def batch_to_graph(
    batch, cutoff, cutoff_esp, cutoff_qmmm_esp, cutoff_qmmm_pol, n_channels
):
    graph = build_graph(
        Z=batch.Z,
        coords_qm=batch.coords[:, : batch.Z.shape[0]],
        coords_mm=batch.coords[:, batch.Z.shape[0] :],
        charges_mm=batch.charges_mm,
        mol_charge=batch.charge,
        cutoff=cutoff,
        cutoff_esp=cutoff_esp,
        cutoff_qmmm_esp=cutoff_qmmm_esp,
        cutoff_qmmm_pol=cutoff_qmmm_pol,
        n_channels=n_channels,
    )
    return graph

def build_graph(
    Z,
    coords_qm,
    coords_mm,
    charges_mm,
    mol_charge: int = 0,
    cutoff: float = 5.0,
    cutoff_esp: float = 14.0,
    cutoff_qmmm_esp: float = 500.0,
    cutoff_qmmm_pol: float = 9.0,
    n_channels: int = 32,
):
    dmat_qm_sq = pdist_sq_unsafe(coords_qm)
    R1, R2, Rx1, Rx2, senders, receivers = prepare_qm_indices(
        dmat_qm_sq, coords_qm, cutoff
    )
    R1_esp, R2_esp, senders_esp, receivers_esp, batch_index_esp = prepare_es_indices(
        dmat_qm_sq, cutoff, cutoff_esp
    )
    # Set high cutoff during training to ensure that all QM particles interact electrostatically with all MM particles
    dmat_qmmm = cdist(coords_qm, coords_mm)
    (
        R1_qmmm_esp,
        Rx1_qmmm_esp,
        Rx2_qmmm_esp,
        receivers_qmmm_esp,
        R1_qmmm_pol,
        Rx1_qmmm_pol,
        Rx2_qmmm_pol,
        receivers_qmmm_pol,
        qm_indices_qmmm_esp,
        mm_monos_esp,
        mm_monos_pol,
    ) = prepare_features_qmmm(
        dmat_qmmm,
        coords_qm,
        coords_mm,
        charges_mm,
        cutoff_esp=10000.0,
        cutoff_pol=cutoff_qmmm_pol,
    )
    batch_size, mol_size = coords_qm.shape[:2]
    if Z.shape[0] == mol_size:
        Z = Z.tile([coords_qm.shape[0]])
    mol_size = torch.full([batch_size], mol_size, device=R1.device, dtype=torch.int64)
    mol_charge = torch.full([batch_size], mol_charge, device=R1.device, dtype=R1.dtype)
    graph = Graph(
        Z=Z,
        nodes=Z,
        coords_qm=coords_qm,
        mm_monos_esp=mm_monos_esp,
        mm_monos_pol=mm_monos_pol,
        mol_charge=mol_charge,
        mol_size=mol_size,
        R1=R1,
        R2=R2,
        Rx1=Rx1,
        Rx2=Rx2,
        senders=senders,
        receivers=receivers,
        R1_esp=R1_esp,
        R2_esp=R2_esp,
        senders_esp=senders_esp,
        receivers_esp=receivers_esp,
        batch_index_esp=batch_index_esp,
        R1_qmmm_esp=R1_qmmm_esp,
        Rx1_qmmm_esp=Rx1_qmmm_esp,
        Rx2_qmmm_esp=Rx2_qmmm_esp,
        receivers_qmmm_esp=receivers_qmmm_esp,
        qm_indices_qmmm_esp=qm_indices_qmmm_esp,
        R1_qmmm_pol=R1_qmmm_pol,
        Rx1_qmmm_pol=Rx1_qmmm_pol,
        Rx2_qmmm_pol=Rx2_qmmm_pol,
        receivers_qmmm_pol=receivers_qmmm_pol,
        md_mode=False,
        n_channels=n_channels,
    )
    return graph

def prepare_qm_indices(dmat_sq, coordinates, cutoff: float = 5.0):
    mol_size = dmat_sq.shape[-1]
    mol_id, senders, receivers = torch.where(
        torch.logical_and(dmat_sq < cutoff**2, dmat_sq > 1e-2)
    )
    R2 = dmat_sq[mol_id, senders, receivers].unsqueeze(-1)
    R1 = torch.sqrt(R2)
    Rx1 = (coordinates[mol_id, receivers] - coordinates[mol_id, senders]) / R1
    Rx2 = build_Rx2(Rx1)
    shift = mol_size * mol_id
    Rx1, Rx2 = Rx1.unsqueeze(1), Rx2.unsqueeze(1)
    senders, receivers = senders + shift, receivers + shift
    return R1, R2, Rx1, Rx2, senders, receivers

def prepare_es_indices(dmat_sq, cutoff_qm: float = 5.0, cutoff_esp: float = 14.0):
    mol_size = dmat_sq.shape[-1]
    triu_indices = torch.triu_indices(
        int(dmat_sq.shape[1]), int(dmat_sq.shape[1]), offset=1, device=dmat_sq.device
    )
    R2_esp = dmat_sq[:, triu_indices[0], triu_indices[1]]
    batch_index_esp, cutoff_index_esp = torch.where(R2_esp < cutoff_esp**2)
    R2_esp = R2_esp[batch_index_esp, cutoff_index_esp].unsqueeze(-1)
    R1_esp = torch.sqrt(R2_esp)
    triu_indices_cutoff = triu_indices[:, cutoff_index_esp]
    shift = mol_size * batch_index_esp
    senders_esp, receivers_esp = (
        triu_indices_cutoff[0] + shift,
        triu_indices_cutoff[1] + shift,
    )
    return R1_esp, R2_esp, senders_esp, receivers_esp, batch_index_esp

def prepare_features_qmmm(
    dmat,
    qm_coords,
    mm_coords,
    charges_mm,
    cutoff_esp: float = 500.0,
    cutoff_pol: float = 9.0,
):
    indices_qmmm = torch.where(dmat < cutoff_esp)
    batch_indices_qmmm, receivers_qmmm, senders_qmmm = indices_qmmm
    qm_indices_qmmm = torch.stack((batch_indices_qmmm, receivers_qmmm), dim=0)
    mm_indices_qmmm = torch.stack((batch_indices_qmmm, senders_qmmm), dim=0)
    coords_1 = qm_coords[batch_indices_qmmm, receivers_qmmm]
    coords_2 = mm_coords[batch_indices_qmmm, senders_qmmm]
    R1_qmmm_esp = dmat[batch_indices_qmmm, receivers_qmmm, senders_qmmm].unsqueeze(-1)
    Rx1_qmmm_esp = coords_2 - coords_1
    Rx2_qmmm_esp = build_Rx2(Rx1_qmmm_esp)
    # Indices of atoms in the QM Zone, unidirectional interaction
    receivers_qmmm_esp = indices_qmmm[1] + indices_qmmm[0] * qm_coords.shape[1]
    qm_indices_qmmm_esp, mm_indices_qmmm_esp = qm_indices_qmmm, mm_indices_qmmm
    cutoff_indices = torch.where(R1_qmmm_esp[:, 0] < cutoff_pol)[0]
    R1_qmmm_pol = R1_qmmm_esp[cutoff_indices]
    Rx1_qmmm_pol = Rx1_qmmm_esp[cutoff_indices] / R1_qmmm_pol
    Rx2_qmmm_pol = build_Rx2(Rx1_qmmm_pol)
    receivers_qmmm_pol = receivers_qmmm_esp[cutoff_indices]
    qm_indices_qmmm_pol = qm_indices_qmmm_esp[:, cutoff_indices]
    mm_indices_qmmm_pol = mm_indices_qmmm_esp[:, cutoff_indices]
    mm_monos_esp = charges_mm[mm_indices_qmmm_esp[0], mm_indices_qmmm_esp[1]].unsqueeze(
        -1
    )
    mm_monos_pol = charges_mm[mm_indices_qmmm_pol[0], mm_indices_qmmm_pol[1]].unsqueeze(
        -1
    )
    return (
        R1_qmmm_esp,
        Rx1_qmmm_esp,
        Rx2_qmmm_esp,
        receivers_qmmm_esp,
        R1_qmmm_pol,
        Rx1_qmmm_pol,
        Rx2_qmmm_pol,
        receivers_qmmm_pol,
        qm_indices_qmmm_esp,
        mm_monos_esp,
        mm_monos_pol,
    )

"""basic scatter_sum operations from torch_scatter from
https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter_sum.py
Using code from https://github.com/rusty1s/pytorch_scatter, but cut down to avoid a dependency.
PyTorch plans to move these features into the main repo, but until then,
to make installation simpler, we need this pure python set of wrappers
that don't require installing PyTorch C++ extensions.
See https://github.com/pytorch/pytorch/issues/63780.

From:
https://github.com/ACEsuit/mace/blob/7543f162cfc02b41c1654c73d8c7a393b8b9e9d5/mace/tools/scatter.py
"""


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out

