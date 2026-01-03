# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import torch
import numpy as np


@torch.jit.script
class Graph:
    def __init__(
        self,
        Z,
        nodes,
        coords_qm,
        mm_monos_esp,
        mm_monos_pol,
        mol_charge,
        mol_size,
        R1,
        R2,
        Rx1,
        Rx2,
        senders,
        receivers,
        R1_esp,
        R2_esp,
        senders_esp,
        receivers_esp,
        batch_index_esp,
        R1_qmmm_esp,
        Rx1_qmmm_esp,
        Rx2_qmmm_esp,
        receivers_qmmm_esp,
        qm_indices_qmmm_esp,
        R1_qmmm_pol,
        Rx1_qmmm_pol,
        Rx2_qmmm_pol,
        receivers_qmmm_pol,
        md_mode: bool,
        n_channels: int,
    ):
        _ = torch.empty(0)
        batch_size = mol_size.shape[0]
        self.coords_qm = coords_qm
        self.md_mode = md_mode
        self.device = R1.device
        self.dtype = R1.dtype
        self.Z = Z
        self.nodes = nodes
        self.C6 = _
        self.C6_factors = _
        self.batch_ids = _
        self.mol_size = mol_size
        self.mol_charge = mol_charge
        self.n_nodes = self.Z.shape[0]
        self.batch_index_esp = batch_index_esp
        self.qm_indices_qmmm_esp = qm_indices_qmmm_esp
        if not self.md_mode:
            self.batch_ids = torch.arange(batch_size, device=self.device).repeat_interleave(self.mol_size)
        self.R1, self.R2, self.Rx1, self.Rx2 = R1, R2, Rx1, Rx2
        self.senders, self.receivers = senders, receivers
        self.R1_esp, self.R2_esp = R1_esp, R2_esp
        self.senders_esp, self.receivers_esp = senders_esp, receivers_esp
        # Initialize attributes for later construction.
        self.edges = _
        self.aniso_weights = _
        self.envelope = _
        self.monos = _  # torch.zeros((self.n_nodes, 1), device=self.device, dtype=self.dtype)
        self.dipos = _  # torch.zeros((self.n_nodes, n_channels, 3), device=self.device, dtype=self.dtype)
        self.quads = _  # torch.zeros((self.n_nodes, n_channels, 3, 3), device=self.device, dtype=self.dtype)
        self.dipos_qmmm = _  # torch.zeros((self.n_nodes, 3), device=self.device, dtype=self.dtype)
        self.quads_qmmm = _  # torch.zeros((self.n_nodes, 3, 3), device=self.device, dtype=self.dtype)
        self.V_total = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.V_nodes = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.V_coulomb_qm = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.V_coulomb_qmmm = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.V_D4 = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.V_ZBL = torch.zeros((batch_size, 1), device=self.device, dtype=self.dtype)
        self.edges_qmmm = _
        self.envelope_qmmm = _
        self.R1_qmmm_esp, self.Rx1_qmmm_esp, self.Rx2_qmmm_esp = R1_qmmm_esp, Rx1_qmmm_esp, Rx2_qmmm_esp
        self.receivers_qmmm_esp = receivers_qmmm_esp
        self.R1_qmmm_pol, self.Rx1_qmmm_pol, self.Rx2_qmmm_pol = R1_qmmm_pol, Rx1_qmmm_pol, Rx2_qmmm_pol
        self.receivers_qmmm_pol = receivers_qmmm_pol
        self.mm_monos_esp = mm_monos_esp
        self.mm_monos_pol = mm_monos_pol
