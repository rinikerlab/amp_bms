# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import os
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from typing import Final
from utilities.Scatter import scatter_sum as scatter

from datastructures.Graphs import Graph

"""
computes D4 dispersion energy
wB97M    s6=1.0  s8=0.7761  a1=0.7514  a2=2.7099

Adapted from SpookyNet: https://github.com/OUnke/SpookyNet/blob/main/spookynet/modules/d4_dispersion_energy.py
"""


class D4(nn.Module):
    def __init__(
        self,
        s6: float = 1.0,
        s8: float = 0.7761,
        a1: float = 0.7514,
        a2: float = 2.7099,
        max_z: int = 87,
    ):
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
        param_folder = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data"
        )
        path_c6 = os.path.join(param_folder, "ref_C6.npy")
        path_r4r2 = os.path.join(param_folder, "sqrt_r4r2.pth")
        ref_c6 = torch.from_numpy(np.load(path_c6)[:max_z, 0])
        sqrt_r4r2 = 1.3160740129524924 * torch.load(path_r4r2, weights_only=True)[
            :max_z
        ].unsqueeze(-1)
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
        C6i = C6_0i * torch.index_select(
            graph.C6_factors, dim=0, index=graph.senders_esp
        )
        C6j = C6_0j * torch.index_select(
            graph.C6_factors, dim=0, index=graph.receivers_esp
        )
        C6ij = torch.sqrt(C6i * C6j)
        D4_terms = self._calc_disp_term(C6ij, R2, graph.senders_esp, Zi, Zj)
        if graph.md_mode:
            graph.V_D4 = -self.H_TO_KJ * D4_terms.sum(dim=0, keepdim=True)
        else:
            scatter(
                -self.H_TO_KJ * D4_terms, graph.batch_index_esp, dim=0, out=graph.V_D4
            )
            graph.C6 = (
                graph.C6_factors * torch.index_select(self.C6_0, dim=0, index=graph.Z)
            )[:, 0] * self.convert2kJAngstrom6
        return graph
