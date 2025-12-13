# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import os
import time

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric

from amp.AMP import AMP
from utilities.Helpers import load_parameters, batch_to_graph
from datastructures.Data import Dataset


def initialize_model(model_type="MIN", script=False):
    PARAMETERS = load_parameters(f"parameters/PARAMETERS_{model_type}.yaml")
    model = AMP(PARAMETERS).to(PARAMETERS["device"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized Model with {n_parameters} trainable parameters.")
    if script:
        return torch.jit.script(model)
    return model


def initialize_datasets(folder_path_train, folder_path_test=None, pin_memory=False):
    dataset_train = Dataset(folder_path=folder_path_train)
    loader_train = DataLoader(
        dataset_train, collate_fn=lambda x: x[0], shuffle=True, pin_memory=pin_memory
    )
    if folder_path_test is not None:
        dataset_test = Dataset(folder_path=folder_path_test)
        loader_test = DataLoader(
            dataset_test, collate_fn=lambda x: x[0], shuffle=True, pin_memory=pin_memory
        )
        return loader_train, loader_test
    return loader_train


def get_loss(model, batch, loss_fn, metrics_dict, train=True):
    graph = batch_to_graph(
        batch,
        model.cutoff,
        model.cutoff_esp,
        model.cutoff_qmmm_esp,
        model.cutoff_qmmm_pol,
        model.n_channels,
    )
    graph = model(graph)
    if train:
        gradients = torch.autograd.grad(
            graph.V_total,
            batch.coords,
            grad_outputs=torch.ones_like(graph.V_total),
            retain_graph=True,
            create_graph=True,
        )[0]
    else:
        gradients = torch.autograd.grad(
            graph.V_total, batch.coords, grad_outputs=torch.ones_like(graph.V_total)
        )[0]
    qm_gradients = gradients[:, : batch.Z.shape[0]]
    mm_gradients = gradients[:, batch.Z.shape[0] :]
    dipo_pred = model.mu(graph, include_dipo_term=True)
    quad_pred = model.theta(graph, include_quad_term=True)
    dipo_pred_mono = model.mu(graph, include_dipo_term=False)
    quad_pred_mono = model.theta(graph, include_quad_term=False)
    V_offset_pred, V_offset_ref = graph.V_total[0], batch.e_qmmm[0]
    loss_energy = (1 - model.alpha) * loss_fn(
        graph.V_total - V_offset_pred, batch.e_qmmm - V_offset_ref
    )
    loss_gradient_qm = model.alpha * loss_fn(qm_gradients, batch.grad_qmmm)
    loss_gradient_mm = model.beta * loss_fn(mm_gradients, batch.grad_mm)
    loss_gradient = loss_gradient_qm + loss_gradient_mm
    loss_C6 = model.zeta * loss_fn(batch.C6, graph.C6)
    loss_dipo = loss_fn(batch.dipo_qmmm, dipo_pred)
    loss_quad = loss_fn(batch.quad_qmmm, quad_pred)
    loss_dipo_mono = loss_fn(batch.dipo_qmmm, dipo_pred_mono)
    loss_quad_mono = loss_fn(batch.quad_qmmm, quad_pred_mono)
    loss_multi = model.gamma * (loss_dipo + loss_quad + loss_dipo_mono + loss_quad_mono)
    loss = loss_gradient + loss_energy + loss_multi + loss_C6
    metrics_dict["mae_energy"].update(
        (batch.e_qmmm - V_offset_ref).detach(), (graph.V_total - V_offset_pred).detach()
    )
    metrics_dict["mae_gradient_qm"].update(
        batch.grad_qmmm.detach(), qm_gradients.detach()
    )
    metrics_dict["mae_gradient_mm"].update(
        batch.grad_mm.detach(), mm_gradients.detach()
    )
    metrics_dict["mae_dipoles"].update(batch.dipo_qmmm.detach(), dipo_pred.detach())
    metrics_dict["mae_quadrupoles"].update(batch.quad_qmmm.detach(), quad_pred.detach())
    metrics_dict["mae_c6"].update(batch.C6.detach(), graph.C6.detach())
    metrics_dict["loss"].update(loss.detach())
    if train:
        loss.backward()
    return metrics_dict


def train_step(model, optimizer, batch, loss_fn, metrics_dict):
    optimizer.zero_grad()
    metrics_dict = get_loss(model, batch, loss_fn, metrics_dict, train=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=model.max_grad_norm)
    if all([torch.isfinite(p.grad).all() for p in model.parameters()]):
        optimizer.step()
    return metrics_dict


def validation_step(model, batch, loss_fn, metrics_dict):
    metrics_dict = get_loss(model, batch, loss_fn, metrics_dict, train=False)
    return metrics_dict


def train_epoch(model, optimizer, dataloader, loss_fn, writer, epoch):
    start = time.time()
    metrics_dict = initialize_metrics_dict(model.device)
    for idb, batch in enumerate(dataloader):
        train_step(model, optimizer, batch, loss_fn, metrics_dict)
        # if idb == int(model.n_samples * 16):
        #    break
    print_out(metrics_dict, writer, start, epoch, "Train")


def validate_epoch(model, dataloader, loss_fn, writer, epoch):
    start = time.time()
    metrics_dict = initialize_metrics_dict(model.device)
    for idb, batch in enumerate(dataloader):
        metrics_dict = validation_step(model, batch, loss_fn, metrics_dict)
        if idb == model.n_samples:
            break
    val_loss = metrics_dict["loss"].compute().cpu().detach().numpy()
    print_out(metrics_dict, writer, start, epoch, set_name="Validation")
    return val_loss


def print_out(metrics_dict, writer, start, epoch, set_name="Train"):
    print("Epoch {}".format(epoch))
    print(time.time() - start)
    print(f'MAE Energy {set_name} [kJ/mol]: {metrics_dict["mae_energy"].compute()}')
    print(
        f'MAE QM Gradient {set_name}  [kJ/molA]: {metrics_dict["mae_gradient_qm"].compute()}'
    )
    print(
        f'MAE MM Gradient {set_name}  [kJ/molA]: {metrics_dict["mae_gradient_mm"].compute()}'
    )
    print(f'MAE Dipoles {set_name} [eA]: {metrics_dict["mae_dipoles"].compute()}')
    print(
        f'MAE Quadrupoles {set_name} [eA2]: {metrics_dict["mae_quadrupoles"].compute()}'
    )
    print(f'MAE C6 {set_name} [kJA6/mol]: {metrics_dict["mae_c6"].compute()}')
    writer.add_scalar(
        f"MAE QM Energies {set_name} [kJ/mol]",
        metrics_dict["mae_energy"].compute(),
        epoch,
    )
    writer.add_scalar(
        f"MAE QM Gradients {set_name} [kJ/molA]",
        metrics_dict["mae_gradient_qm"].compute(),
        epoch,
    )
    writer.add_scalar(
        f"MAE MM Gradients {set_name} [kJ/molA]",
        metrics_dict["mae_gradient_mm"].compute(),
        epoch,
    )
    writer.add_scalar(
        f"MAE Dipoles {set_name} [eA]", metrics_dict["mae_dipoles"].compute(), epoch
    )
    writer.add_scalar(
        f"MAE Quadrupoles {set_name} [eA]",
        metrics_dict["mae_quadrupoles"].compute(),
        epoch,
    )
    writer.add_scalar(
        f"MAE C6 {set_name} [kJA6/mol]", metrics_dict["mae_c6"].compute(), epoch
    )
    writer.add_scalar(f"Loss {set_name} [1]", metrics_dict["loss"].compute(), epoch)


def initialize_metrics_dict(device):
    metrics_dict = {}
    metrics_dict["mae_energy"] = MAE().to(device)
    metrics_dict["mae_gradient_qm"] = MAE().to(device)
    metrics_dict["mae_gradient_mm"] = MAE().to(device)
    metrics_dict["mae_c6"] = MAE().to(device)
    metrics_dict["mae_dipoles"] = MAE().to(device)
    metrics_dict["mae_quadrupoles"] = MAE().to(device)
    metrics_dict["loss"] = MeanMetric().to(device)
    return metrics_dict


def initialize_optimizer(model, model_type, model_id=""):
    PARAMETERS = load_parameters(f"parameters/PARAMETERS_{model_type}.yaml")
    loss_fn = torch.nn.MSELoss()
    decay_rate = np.exp(np.log(PARAMETERS["decay_factor"]) / PARAMETERS["n_epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMETERS["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    writer = SummaryWriter(
        f"summaries/{model_id}_{model_type}model_{PARAMETERS['n_epochs']}"
    )
    try:
        os.mkdir(f"weights/{model_id}_{model_type}/")
    except:
        pass
    return loss_fn, optimizer, scheduler, writer, PARAMETERS["n_epochs"]
