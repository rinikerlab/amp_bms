import time
import torch
import yaml

from datastructures.Graphs import Graph
from utilities.Utilities import (
    scalar_product,
    ff_module,
    build_Rx2,
    cdist,
    pdist_sq_unsafe,
)


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
