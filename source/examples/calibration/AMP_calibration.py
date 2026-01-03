# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import torch
import torch.nn as nn
from AMPHelpers_calibration import aniso_features_qmmm, aniso_features, build_poles
from datastructures_calibration import Graph
from utilities_calibration import scatter_sum as scatter
from utilities_calibration import ff_module
from modules_calibration import (Coulomb_QM, Coulomb_QMMM, AtomicEmbedding, BesselKernel,
NodePotential, ZBL, D4, conserve_charges, Mu, Theta)

class AMP(nn.Module):
    def __init__(self, config, activation=nn.SiLU(), **kwargs):
        super(AMP, self).__init__()
        self.activation = activation
        self.device = torch.device(config["device_name"])
        # Cutoffs and constants
        self.k_eps_sqrt = 37.27404695554026
        self.cutoff = config["cutoff"]
        self.cutoff_esp = config["cutoff_esp"]
        self.cutoff_qmmm_esp = config["cutoff_qmmm_esp"]
        self.cutoff_qmmm_pol = config["cutoff_qmmm_pol"]
        self.aniso_pol = config["aniso_pol"]
        # Model and training parameters
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.zeta = config["zeta"]
        self.max_grad_norm = config["max_grad_norm"]
        self.p = config["p"]
        self.eps_rf = config["eps_rf"]
        self.mrf, self.nrf = config["mrf"], config["nrf"]
        self.max_z = config["max_z"]
        self.order = config["order"]
        self.n_steps = config["n_steps"]
        self.n_bessel = config["n_bessel"]
        self.n_bessel_pol = config["n_bessel_pol"]
        self.node_size = config["node_size"]
        self.edge_size = config["edge_size"]
        self.n_channels = config["n_channels"]
        self.n_samples = config["n_samples"]
        self.trainable_bessel = config["trainable_bessel"]
        self.pol_scaling = config["pol_scaling"]
        self._init_layers()
        self._init_modules()

    def _init_layers(self):
        in_update_layers, in_message_layers, eq_message_layers = [], [], []
        n_inputs_coeff = 2 * self.node_size + self.edge_size
        for idl in range(self.n_steps):
            n_inputs = 2 * self.node_size + 9 * self.n_channels + self.edge_size
            n_inputs_eq = n_inputs
            n_inputs_update = 2 * self.node_size
            if idl == (self.n_steps - 1):
                n_inputs = 2 * self.node_size + 9 * (self.n_channels + 1) + self.edge_size
            if idl == 0:
                n_inputs_eq = n_inputs_coeff
            eq_layer = ff_module(
                self.node_size, 
                2, 
                n_inputs_eq, 
                output_size=(self.order) * self.n_channels, 
                activation=self.activation
            )
            in_message_layer = ff_module(
                self.node_size, 
                2, 
                n_inputs, 
                activation=self.activation
            )
            in_update_layer = ff_module(
                self.node_size, 
                2, 
                n_inputs_update, 
                activation=self.activation
            )
            in_message_layers.append(in_message_layer)
            eq_message_layers.append(eq_layer)
            in_update_layers.append(in_update_layer)
        self.edge_embedding = ff_module(
            self.edge_size, 
            1, 
            self.n_bessel + 2 * self.node_size, 
            output_size=self.edge_size, 
            activation=self.activation
        )
        self.eq_message_layers = nn.ModuleList(eq_message_layers)
        self.in_message_layers = nn.ModuleList(in_message_layers)
        self.in_update_layers = nn.ModuleList(in_update_layers)
        self.QM_monos = ff_module(
            self.node_size // 2, 
            1, 
            self.node_size, 
            output_size=1, 
            activation=self.activation
        )
        self.QM_coeffs = ff_module(
            self.node_size // 2,
            1,
            self.node_size,
            output_size=3,
            activation=self.activation,
            final_activation=nn.Softplus(),
        )
        if self.aniso_pol:
            self.B_coefficients = ff_module(
                4,
                1, 
                2 + self.n_bessel_pol,
                output_size=2, 
                activation=self.activation
            )
        else:
            self.B_coefficients = ff_module(
                self.n_bessel_pol, 
                1, 
                self.n_bessel_pol, 
                output_size=2, 
                activation=self.activation
            )

    def _init_modules(self):
        self.node_embedding = AtomicEmbedding(
            node_size=self.node_size, 
            max_z=self.max_z
        )
        self.radial_embedding = BesselKernel(
            cutoff=self.cutoff, 
            n_bessel=self.n_bessel, 
            trainable=self.trainable_bessel, 
            p=self.p
        )
        self.radial_embedding_qmmm = BesselKernel(
            cutoff=self.cutoff_qmmm_pol, 
            n_bessel=self.n_bessel_pol, 
            trainable=self.trainable_bessel, 
            p=self.p
        )

        self.coulomb_qm = Coulomb_QM(
            cutoff=self.cutoff_esp, # Cutoff QM-QM electrostatic interaction 
            cutoff_qm=self.cutoff, # Cutoff QM-QM edge definition
            eps_rf=self.eps_rf, 
            mrf=self.mrf, 
            nrf=self.nrf
        )
        
        self.coulomb_qmmm = Coulomb_QMMM(
            cutoff=self.cutoff_qmmm_esp, # Cutoff QM-MM electrostatic interaction (potential)
            eps_rf=self.eps_rf, 
            mrf=self.mrf, 
            nrf=self.nrf
        )


        self.V = NodePotential(
            node_size=self.node_size, 
            n_layers=2, 
            activation=self.activation
        )
        self.D4 = D4(max_z=self.max_z)
        self.ZBL = ZBL(cutoff=self.cutoff)
        self.mu = Mu()
        self.theta = Theta()

    def forward(self, graph: Graph):
        graph = self._process_graph(graph)
        graph = self._calculate_energy_terms(graph)
        return graph

    def _embed(self, graph: Graph):
        if not graph.md_mode:
            graph.nodes = self.node_embedding(graph.Z)
        graph.edges, graph.envelope = self.radial_embedding(graph.R1)
        features_i = torch.index_select(graph.nodes, dim=0, index=graph.senders)
        features_j = torch.index_select(graph.nodes, dim=0, index=graph.receivers)
        edge_features = torch.cat((graph.edges, features_i, features_j), dim=-1)
        graph.edges = self.edge_embedding(edge_features)
        graph.edges_qmmm, graph.envelope_qmmm = self.radial_embedding_qmmm(graph.R1_qmmm_pol)
        return graph

    def _process_graph(self, graph: Graph):
        graph = self._embed(graph)
        graph = self._pass_messages(graph)
        graph = self._build_multipoles_esp(graph)
        return graph

    def _calculate_energy_terms(self, graph: Graph):
        graph = self.V(graph)
        graph = self.D4(graph)
        graph = self.coulomb_qm(graph)
        graph = self.coulomb_qmmm(graph)
        graph = self.ZBL(graph)
        graph.V_total = (graph.V_nodes + graph.V_coulomb_qm + graph.V_coulomb_qmmm + graph.V_D4 + graph.V_ZBL)
        return graph

    def _build_multipoles_esp(self, graph: Graph):
        graph.monos = self.QM_monos(graph.nodes) / self.k_eps_sqrt
        graph.monos = conserve_charges(graph, graph.monos)
        graph.dipos = (graph.dipos[:, 0] + graph.dipos_qmmm) / self.k_eps_sqrt
        graph.quads = (graph.quads[:, 0] + graph.quads_qmmm) / self.k_eps_sqrt
        return graph

    def _include_mm_polarization(self, graph: Graph):
        if self.aniso_pol:
            QMMM_edge_features = torch.cat((aniso_features_qmmm(graph), graph.edges_qmmm), dim=-1)
        else:
            QMMM_edge_features = graph.edges_qmmm
        # coeffs = self.B_coefficients(QMMM_edge_features) * graph.envelope_qmmm
        coeffs = self.pol_scaling * self.B_coefficients(QMMM_edge_features) * graph.envelope_qmmm
        field = graph.mm_monos_pol / torch.square(graph.R1_qmmm_pol)
        coeffs_d, coeffs_q = (field * coeffs).tensor_split(2, dim=-1)
        graph.dipos_qmmm = scatter(
            coeffs_d * graph.Rx1_qmmm_pol, 
            graph.receivers_qmmm_pol, 
            dim=0, 
            dim_size=graph.n_nodes
        )
        graph.quads_qmmm = scatter(
            coeffs_q.unsqueeze(-1) * graph.Rx2_qmmm_pol, 
            graph.receivers_qmmm_pol, 
            dim=0, 
            dim_size=graph.n_nodes
        )
        QM_coeffs = self.QM_coeffs(graph.nodes)
        # Predict C6 scaling factors before including QM/MM information.
        graph.C6_factors, alphas_d, alphas_q = QM_coeffs.tensor_split(3, dim=-1)
        graph.dipos_qmmm = alphas_d * graph.dipos_qmmm
        graph.quads_qmmm = alphas_q.unsqueeze(-1) * graph.quads_qmmm
        graph.dipos = torch.cat((graph.dipos, graph.dipos_qmmm.unsqueeze(1)), dim=1)
        graph.quads = torch.cat((graph.quads, graph.quads_qmmm.unsqueeze(1)), dim=1)
        return graph

    def _pass_messages(self, graph: Graph):
        edge_features = graph.edges
        for step, (eq_message_layer, in_message_layer, in_update_layer) in enumerate(
            zip(self.eq_message_layers, self.in_message_layers, self.in_update_layers)
        ):
            features_i = torch.index_select(graph.nodes, dim=0, index=graph.senders)
            features_j = torch.index_select(graph.nodes, dim=0, index=graph.receivers)
            coefficients = eq_message_layer(torch.cat((features_i, features_j, edge_features), dim=-1))
            graph = build_poles(graph, coefficients)
            if step == (self.n_steps - 1):
                graph = self._include_mm_polarization(graph)
            edge_features = torch.cat((aniso_features(graph), graph.edges), dim=-1)
            messages = in_message_layer(torch.cat((features_i, features_j, edge_features), dim=-1))
            messages = scatter(messages * graph.envelope, graph.receivers, dim=0, dim_size=graph.n_nodes)
            graph.nodes = graph.nodes + in_update_layer(torch.cat((graph.nodes, messages), dim=-1))
        return graph
