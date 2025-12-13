import os
import torch

from utilities.Scatter import scatter_sum as scatter
from datastructures.Graphs import Graph
from utilities.Utilities import build_Rx2


def compute_com(coords, masses):
    masses = masses.reshape((coords.shape[0], coords.shape[1], 1))
    masses = masses / masses.sum(-2, keepdim=True)
    return (masses * coords).sum(-2, keepdim=True)


# Redistributes excess charge to achieve charge conservation over the whole molecule.
def conserve_charges(graph: Graph, monopoles):
    if graph.md_mode:
        monopoles = monopoles.squeeze()
        charge_residuals = (monopoles.sum() - graph.mol_charge) / graph.mol_size
        charge_residuals = charge_residuals.repeat_interleave(graph.mol_size)
        return (monopoles - charge_residuals).unsqueeze(-1)
    else:
        monopoles = monopoles.squeeze()
        charge_residuals = scatter(
            monopoles,
            graph.batch_ids,
            dim=0,
            dim_size=graph.mol_size.shape[0],
            reduce="sum",
        )
        charge_residuals = (
            (charge_residuals - graph.mol_charge.squeeze()) / graph.mol_size
        ).repeat_interleave(graph.mol_size)
        return (monopoles - charge_residuals).unsqueeze(-1)


class Mu(torch.nn.Module):
    def __init__(self):
        super(Mu, self).__init__()
        param_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )
        path_masses = os.path.join(param_folder, "MASSES.pt")
        self.register_buffer("MASSES", torch.load(path_masses, weights_only=True))

    def forward(self, graph: Graph, include_dipo_term: bool = True):
        qm_coords = graph.coords_qm - compute_com(graph.coords_qm, self.MASSES[graph.Z])
        contribution_monopoles = (
            graph.monos.reshape(qm_coords.shape[:2]).unsqueeze(-1) * qm_coords
        )
        if include_dipo_term:
            contribution_dipoles = graph.dipos.reshape(qm_coords.shape)
            return (contribution_dipoles + contribution_monopoles).sum(dim=1)
        return contribution_monopoles.sum(dim=1)


class Theta(torch.nn.Module):
    def __init__(self):
        super(Theta, self).__init__()
        param_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )
        path_masses = os.path.join(param_folder, "MASSES.pt")
        self.register_buffer("MASSES", torch.load(path_masses, weights_only=True))

    def forward(self, graph: Graph, include_quad_term: bool = True):
        qm_coords = graph.coords_qm - compute_com(graph.coords_qm, self.MASSES[graph.Z])
        monos = graph.monos.reshape(qm_coords.shape[:2]).unsqueeze(-1).unsqueeze(-1)
        contribution_monopoles = monos * build_Rx2(qm_coords)
        if include_quad_term:
            contribution_quadrupoles = graph.quads.reshape(
                (qm_coords.shape[0], qm_coords.shape[1], qm_coords.shape[2], 3)
            )
            return (contribution_quadrupoles + contribution_monopoles).sum(dim=1)
        return contribution_monopoles.sum(dim=1)
