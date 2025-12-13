# Copyright (C) 2025 ETH Zurich, Moritz Thürlemann, and other AMP contributors

import os
import torch
import numpy as np

from dataclasses import dataclass, fields
from torch.utils.data import Dataset
from torch import Tensor

from utilities.Utilities import detrace


class Dataset(Dataset):
    def __init__(self, folder_path="data/batches/", cuda=True):
        self.folder_path = folder_path
        self.dtype = torch.get_default_dtype()
        self.file_names = [
            os.path.join(folder_path, file)
            for file in os.listdir(folder_path)
            if "batch" in file
        ]
        self.cuda = cuda

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index, device=None):
        batch_np = np.load(self.file_names[index], allow_pickle=True).item()
        batch = {}
        for key, item in batch_np.items():
            if key not in ["coords_qm", "coords_mm"]:
                if key in ["Z", "charge"]:
                    batch[key] = torch.as_tensor(item, dtype=torch.int64)
                elif key in ["quad_qm", "quad_qmmm"]:
                    batch[key] = detrace(torch.as_tensor(item, dtype=self.dtype))
                else:
                    batch[key] = torch.as_tensor(item, dtype=self.dtype)
        coords = np.concatenate((batch_np["coords_qm"], batch_np["coords_mm"]), axis=1)
        batch["coords"] = torch.as_tensor(coords, dtype=self.dtype).requires_grad_()
        if self.cuda:
            return Batch(**batch).cuda()
        return Batch(**batch)


@dataclass
class Batch:
    Z: Tensor
    coords: Tensor | None = None
    charge: Tensor | None = None
    charges_mm: Tensor | None = None
    grad_mm: Tensor | None = None
    e_qm: Tensor | None = None
    e_qmmm: Tensor | None = None
    grad_qm: Tensor | None = None
    grad_qmmm: Tensor | None = None
    dipo_qm: Tensor | None = None
    dipo_qmmm: Tensor | None = None
    quad_qm: Tensor | None = None
    quad_qmmm: Tensor | None = None
    C6: Tensor | None = None

    def keys(self):
        return [
            attribute.name
            for attribute in fields(self)
            if getattr(self, attribute.name) is not None
        ]

    def cuda(self, device=None, non_blocking=True):
        return self.apply(lambda x: x.cuda(device=device, non_blocking=non_blocking))

    def cpu(self):
        return self.apply(lambda x: x.cpu())

    def apply(self, func):
        for key, item in self.__dict__.items():
            if item is not None:
                self.__dict__[key] = func(item)
        return self

    def pin_memory(self):
        return self.apply(lambda x: x.pin_memory())
