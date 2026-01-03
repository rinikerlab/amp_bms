# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import torch
from datastructures_calibration import Graph
from utilities_calibration import scatter_sum as scatter
from utilities_calibration import scalar_product


def get_norms(graph: Graph):
    q_norm = torch.linalg.norm(graph.quads, dim=[-1, -2])
    d_norm = torch.linalg.norm(graph.dipos, dim=[-1])
    return torch.cat((d_norm, q_norm), dim=-1)

def build_poles(graph: Graph, coefficients):
    dipos, quads = (coefficients * graph.envelope).tensor_split(2, dim=-1)
    graph.dipos = scatter(dipos.unsqueeze(-1) * graph.Rx1, graph.receivers, dim=0, dim_size=graph.n_nodes)
    graph.quads = scatter(quads[..., None, None] * graph.Rx2, graph.receivers, dim=0, dim_size=graph.n_nodes)
    return graph

def aniso_features(graph: Graph):
    dipos_1 = torch.index_select(graph.dipos, dim=0, index=graph.senders)
    dipos_2 = torch.index_select(graph.dipos, dim=0, index=graph.receivers)
    quads_1 = torch.index_select(graph.quads, dim=0, index=graph.senders)
    quads_2 = torch.index_select(graph.quads, dim=0, index=graph.receivers)
    dipo_dipo = scalar_product(dipos_1, dipos_2)
    D1_Rx1 = scalar_product(dipos_1, graph.Rx1)
    D2_Rx1 = scalar_product(dipos_2, graph.Rx1)
    Q1_Rx1 = torch.einsum("bcjk, bck -> bcj", quads_1, graph.Rx1)
    Q2_Rx1 = torch.einsum("bcjk, bck -> bcj", quads_2, graph.Rx1)
    Q1_Rx2 = torch.einsum("bcjk, bcjk -> bc", quads_1, graph.Rx2).unsqueeze(-1)
    Q2_Rx2 = torch.einsum("bcjk, bcjk -> bc", quads_2, graph.Rx2).unsqueeze(-1)
    quad_dipo = scalar_product(Q1_Rx1, dipos_2)
    dipo_quad = scalar_product(Q2_Rx1, dipos_1)
    quad_quad = torch.einsum("bcjk, bcjk -> bc", quads_1, quads_2).unsqueeze(-1)
    quad_R = scalar_product(Q1_Rx1, Q2_Rx1)
    aniso_features = torch.cat((D1_Rx1, D2_Rx1, dipo_dipo, Q1_Rx2, Q2_Rx2, quad_dipo, dipo_quad, quad_quad, quad_R), dim=-1)
    return aniso_features.view(dipos_1.shape[0], dipos_1.shape[1] * 9)

def aniso_features_qmmm(graph: Graph):
    qm_dipos = torch.index_select(graph.dipos[:, 0], dim=0, index=graph.receivers_qmmm_pol)
    qm_quads = torch.index_select(graph.quads[:, 0], dim=0, index=graph.receivers_qmmm_pol)
    D1_Rx1 = scalar_product(qm_dipos, graph.Rx1_qmmm_pol, keepdim=False)
    Q1_Rx2 = torch.einsum("bjk, bjk -> b", qm_quads, graph.Rx2_qmmm_pol)
    return torch.stack((D1_Rx1, Q1_Rx2), dim=-1)
