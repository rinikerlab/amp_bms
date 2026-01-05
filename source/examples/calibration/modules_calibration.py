# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

from utilities_calibration import scatter_sum as scatter
import torch
from datastructures_calibration import Graph
from utilities_calibration import scalar_product, ff_module, build_Rx2
from torch import Tensor
import numpy as np
import os
import torch.nn as nn


class BesselKernel(torch.nn.Module):
    def __init__(self, cutoff: float = 5.0, n_bessel: int = 8, trainable=False, p: float = 6):
        super(BesselKernel, self).__init__()
        frequencies = np.pi * torch.linspace(1, n_bessel, n_bessel, dtype=torch.get_default_dtype()).unsqueeze(0)
        if trainable:
            self.frequencies = torch.nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.cutoff = cutoff
        self.p = p
        self.a = -(p + 1) * (p + 2) / 2
        self.b = p * (p + 2)
        self.c = -p * (p + 1) / 2

    def forward(self, R1: Tensor):
        R1_scaled = R1 / self.cutoff
        envelope = self.envelope_bessel(R1_scaled)
        return envelope * torch.sin(self.frequencies * R1_scaled), envelope

    def envelope_bessel(self, R1):
        Rp_ = torch.pow(R1, self.p - 1)
        Rp = Rp_ * R1
        return torch.reciprocal(R1) + self.a * Rp_ + self.b * Rp + self.c * Rp * R1

class AtomicEmbedding(torch.nn.Module):
    def __init__(self, node_size: int = 128, max_z: int = 54):
        super(AtomicEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=max_z, embedding_dim=node_size)

    def forward(self, Z):
        return self.embedding(Z)

class NodePotential(torch.nn.Module):
    def __init__(self, node_size=128, n_layers=2, activation=torch.nn.SiLU()):
        super(NodePotential, self).__init__()
        self.potential = ff_module(node_size, n_layers, node_size, output_size=1, activation=activation)

    def forward(self, graph: Graph):
        node_terms = self.potential(graph.nodes)
        if graph.md_mode:
            graph.V_nodes = node_terms.sum(dim=0, keepdim=True)
        else:
            scatter(node_terms, graph.batch_ids, dim=0, out=graph.V_nodes)
        return graph

"""
Short-range repulsive potential described in Ziegler, J.F.,
Biersack, J.P., and Littmark, U., "The stopping and range of ions in solids".

Adapted from SpookyNet: https://github.com/OUnke/SpookyNet/blob/main/spookynet/modules/zbl_repulsion_energy.py
"""
class ZBL(torch.nn.Module):
    def __init__(self, cutoff: float = 5.0):
        super(ZBL, self).__init__()
        self._kc = 1389.35457644382
        self._adiv = 1 / (0.8854 * 0.5291772105638411)
        self._cutoff = cutoff
        self._cutoff_switch = self._cutoff - 1
        self._switch_diff = self._cutoff - self._cutoff_switch
        self._apow = 0.23
        self._c1 = 0.18180
        self._c2 = 0.50990
        self._c3 = 0.28020
        self._c4 = 0.02817
        self._a1 = 3.20000
        self._a2 = 0.94230
        self._a3 = 0.40280
        self._a4 = 0.20160
        self.a, self.b, self.c = 6.0, 15.0, 10.0

    def forward(self, graph: Graph):
        cutoff_indices = torch.where(graph.R1_esp.squeeze() < self._cutoff)[0]
        indices_i, indices_j = graph.senders_esp[cutoff_indices], graph.receivers_esp[cutoff_indices]
        R1 = graph.R1_esp[cutoff_indices]
        Z = graph.Z.unsqueeze(-1)
        Za = torch.pow(graph.Z, self._apow).unsqueeze(-1)
        Zij = Z[indices_i] * Z[indices_j]
        a = (Za[indices_i] + Za[indices_j]) * self._adiv
        a1 = self._a1 * a
        a2 = self._a2 * a
        a3 = self._a3 * a
        a4 = self._a4 * a
        phi = (
            self._c1 * torch.exp(-a1 * R1)
            + self._c2 * torch.exp(-a2 * R1)
            + self._c3 * torch.exp(-a3 * R1)
            + self._c4 * torch.exp(-a4 * R1)
        )
        switching = self.switching_fn(R1)
        ZBL_terms = self._kc * switching * phi * (Zij / R1)
        if graph.md_mode:
            graph.V_ZBL = ZBL_terms.sum(dim=0, keepdim=True)
        else:
            scatter(ZBL_terms, graph.batch_index_esp[cutoff_indices], dim=0, out=graph.V_ZBL)
        return graph

    def switching_fn(self, R1):
        X = (R1 - self._cutoff_switch) / self._switch_diff
        X3 = torch.pow(X, 3)
        X4 = X3 * X
        return torch.clip(1 - self.a * X4 * X + self.b * X4 - self.c * X3, 0.0, 1.0)

class Coulomb_QM(torch.nn.Module):

    def __init__(self, cutoff: float = 14, cutoff_qm: float = 4, eps_rf: float = 78.4, mrf: int = 4, nrf: int = 6):

        super(Coulomb_QM, self).__init__()

        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
        arfm = (3 * cutoff ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * ((2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf))
        arfn = (3 * cutoff ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * ((2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * cutoff**mrf + arfn * cutoff**nrf
        self.a, self.b, self.c = 6.0, 15.0, 10.0
        self.k_eps = 1389.35457644382
        self.krf = krf
        self.crf = crf
        self.arfm = arfm
        self.arfn = arfn
        self.cutoff_qm = cutoff_qm

    def forward(self, graph: Graph):

        coulomb_terms_qm = self.coulomb_qm(graph)

        if graph.md_mode:

            graph.V_coulomb_qm = coulomb_terms_qm.sum(dim=0, keepdim=True)
        
        else:
            
            coulomb_terms = scatter(coulomb_terms_qm, graph.batch_index_esp, dim=0, out=graph.V_coulomb_qm)
        
        graph.V_coulomb_qm = graph.V_coulomb_qm*self.k_eps

        return graph

    def switching_fn0(self, R1):
        X = R1 / self.cutoff_qm
        X3 = torch.pow(X, 3)
        X4 = X3 * X
        return torch.clip(self.a * X4 * X - self.b * X4 + self.c * X3, 0.0, 1.0)

    def coulomb_qm(self, graph: Graph):
        switching_esp = self.switching_fn0(graph.R1_esp)
        R4 = graph.R2_esp * graph.R2_esp
        R6 = graph.R2_esp * R4
        RF_weights = (
            torch.reciprocal(graph.R1_esp) + self.krf * graph.R2_esp + self.arfm * R4 + self.arfn * R6 - self.crf
        )
        coulomb_weights = switching_esp * RF_weights
        monos_1 = torch.index_select(graph.monos, dim=0, index=graph.senders_esp)
        monos_2 = torch.index_select(graph.monos, dim=0, index=graph.receivers_esp)
        return coulomb_weights * monos_1 * monos_2

class Coulomb_QMMM(torch.nn.Module):

    def __init__(self, cutoff: float = 14, eps_rf: float = 78.4, mrf: int = 4, nrf: int = 6):
        super(Coulomb_QMMM, self).__init__()
        krf = ((eps_rf - 1) / (1 + 2 * eps_rf)) * (1 / cutoff**3)
        arfm = (3 * cutoff ** (-(mrf + 1)) / (mrf * (nrf - mrf))) * ((2 * eps_rf + nrf - 1) / (1 + 2 * eps_rf))
        arfn = (3 * cutoff ** (-(nrf + 1)) / (nrf * (mrf - nrf))) * ((2 * eps_rf + mrf - 1) / (1 + 2 * eps_rf))
        crf = ((3 * eps_rf) / (1 + 2 * eps_rf)) * (1 / cutoff) + arfm * cutoff**mrf + arfn * cutoff**nrf
        self.a, self.b, self.c = 6.0, 15.0, 10.0
        self.k_eps = 1389.35457644382
        self.krf = krf
        self.crf = crf
        self.arfm = arfm
        self.arfn = arfn


    def forward(self, graph: Graph):

        coulomb_terms_qmmm = self.coulomb_qmmm(graph)

        if graph.md_mode:

            graph.V_coulomb_qmmm = coulomb_terms_qmmm.sum(dim=0, keepdim=True)

        else:

            scatter(coulomb_terms_qmmm, graph.qm_indices_qmmm_esp[0], dim=0, out=graph.V_coulomb_qmmm)

        graph.V_coulomb_qmmm = graph.V_coulomb_qmmm * self.k_eps

        return graph


    def coulomb_qmmm(self, graph: Graph):
        B0, B1, B2 = self.B_matrices_ESP(graph)
        G0, G1, G2 = self.G_matrices_ESP(graph)
        return G0 * B0 + G1 * B1 + G2 * B2

    def G_matrices_ESP(self, graph: Graph):
        qm_monos = torch.index_select(graph.monos, dim=0, index=graph.receivers_qmmm_esp)
        qm_dipos = torch.index_select(graph.dipos, dim=0, index=graph.receivers_qmmm_esp)
        qm_quads = torch.index_select(graph.quads, dim=0, index=graph.receivers_qmmm_esp)
        D1_Rx1 = scalar_product(qm_dipos, graph.Rx1_qmmm_esp)
        Q1_Rx2 = torch.einsum("bjk, bjk -> b", qm_quads, graph.Rx2_qmmm_esp).unsqueeze(-1)
        G0 = qm_monos * graph.mm_monos_esp
        G1 = D1_Rx1 * graph.mm_monos_esp
        G2 = Q1_Rx2 * graph.mm_monos_esp
        return G0, G1, G2

    def B_matrices_ESP(self, graph: Graph):
        R2 = torch.square(graph.R1_qmmm_esp)
        B0 = torch.reciprocal(graph.R1_qmmm_esp)
        if graph.md_mode:
            R4 = R2 * R2
            R6 = R2 * R4
            B0 = B0 + self.krf * R2 + self.arfm * R4 + self.arfn * R6 - self.crf
        B1 = B0 / R2
        B2 = 3 * B1 / R2
        return B0, B1, B2


"""
computes D4 dispersion energy
wB97M    s6=1.0  s8=0.7761  a1=0.7514  a2=2.7099

Adapted from SpookyNet: https://github.com/OUnke/SpookyNet/blob/main/spookynet/modules/d4_dispersion_energy.py
"""
class D4(nn.Module):
    def __init__(self, s6: float = 1.0, s8: float = 0.7761, a1: float = 0.7514, a2: float = 2.7099, max_z: int = 87):
        """Initializes the D4DispersionEnergy class."""
        super(D4, self).__init__()
        # Grimme's D4 dispersion is only parametrized up to Rn (Z=86)
        assert max_z <= 87
        # D4 constants
        Bohr = 0.5291772105638411
        convert2Bohr = 1 / Bohr
        self.H_TO_KJ = 627.509474 * 4.184
        self.convert2Bohr2 = convert2Bohr**2
        self.convert2kJAngstrom6 = self.H_TO_KJ * Bohr**6
        self.s6 = s6
        self.s8 = s8
        self.a1 = a1
        self.a2 = a2
        # load D4 data
        param_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        path_c6 = os.path.join(param_folder, "ref_C6.npy")
        path_r4r2 = os.path.join(param_folder, "sqrt_r4r2.pth")
        ref_c6 = torch.from_numpy(np.load(path_c6)[:max_z, 0])        
        sqrt_r4r2 = 1.3160740129524924 * torch.load(path_r4r2, weights_only=True)[:max_z].unsqueeze(-1)
        self.register_buffer("C6_0", ref_c6.to(dtype=torch.get_default_dtype()))
        self.register_buffer("sqrt_r4r2", sqrt_r4r2.to(dtype=torch.get_default_dtype()))

    def _calc_disp_term(self, C6ij, R2, idx_i, Zi, Zj):
        sqrt_r4r2ij = self.sqrt_r4r2[Zi] * self.sqrt_r4r2[Zj]
        R0_2 = torch.square(self.a1 * sqrt_r4r2ij + self.a2)
        R0_6 = torch.pow(R0_2, 3)
        R0_8 = R0_2 * R0_6
        R6 = torch.pow(R2, 3)
        R8 = R6 * R2
        return C6ij * (self.s6 / (R6 + R0_6) + self.s8 * sqrt_r4r2ij**2 / (R8 + R0_8))

    def forward(self, graph: Graph):
        n_atoms = graph.Z.shape[0]
        R2 = graph.R2_esp * self.convert2Bohr2
        Zi = torch.index_select(graph.Z, dim=0, index=graph.senders_esp)
        Zj = torch.index_select(graph.Z, dim=0, index=graph.receivers_esp)
        C6_0i = torch.index_select(self.C6_0, dim=0, index=Zi)
        C6_0j = torch.index_select(self.C6_0, dim=0, index=Zj)
        C6i = C6_0i * torch.index_select(graph.C6_factors, dim=0, index=graph.senders_esp)
        C6j = C6_0j * torch.index_select(graph.C6_factors, dim=0, index=graph.receivers_esp)
        C6ij = torch.sqrt(C6i * C6j)
        D4_terms = self._calc_disp_term(C6ij, R2, graph.senders_esp, Zi, Zj)
        if graph.md_mode:
            graph.V_D4 = -self.H_TO_KJ * D4_terms.sum(dim=0, keepdim=True)
        else:
            scatter(-self.H_TO_KJ * D4_terms, graph.batch_index_esp, dim=0, out=graph.V_D4)
            graph.C6 = (graph.C6_factors * torch.index_select(self.C6_0, dim=0, index=graph.Z))[:, 0] * self.convert2kJAngstrom6
        return graph

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
        charge_residuals = scatter(monopoles, graph.batch_ids, dim=0, dim_size=graph.mol_size.shape[0], reduce="sum")
        charge_residuals = ((charge_residuals - graph.mol_charge.squeeze()) / graph.mol_size).repeat_interleave(
            graph.mol_size
        )
        return (monopoles - charge_residuals).unsqueeze(-1)

class Mu(torch.nn.Module):
    def __init__(self):
        super(Mu, self).__init__()
        param_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        path_masses = os.path.join(param_folder, "MASSES.pt")
        self.register_buffer("MASSES", torch.load(path_masses, weights_only=True))

    def forward(self, graph: Graph, include_dipo_term: bool = True):
        qm_coords = graph.coords_qm - compute_com(graph.coords_qm, self.MASSES[graph.Z])
        contribution_monopoles = graph.monos.reshape(qm_coords.shape[:2]).unsqueeze(-1) * qm_coords
        if include_dipo_term:
            contribution_dipoles = graph.dipos.reshape(qm_coords.shape)
            return (contribution_dipoles + contribution_monopoles).sum(dim=1)
        return contribution_monopoles.sum(dim=1)

class Theta(torch.nn.Module):
    def __init__(self):
        super(Theta, self).__init__()
        param_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
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