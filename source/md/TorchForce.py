import torch

from datastructures.Graphs import Graph
from utilities.Utilities import build_Rx2


class ForceModule(torch.nn.Module):
    def __init__(
        self,
        amp,
        simulator,
        n_nlist=64,
        pairlist_padding=4.0,
        chunk_size=10000000,
        block_size=4000,
        dtype=torch.float32,
        device=torch.device("cuda"),
    ):
        super(ForceModule, self).__init__()
        self.simulator = simulator
        self.device = device
        self.dtype = dtype
        Z = [
            atom.element.atomic_number
            for ida, atom in enumerate(simulator.modeller.topology.atoms())
            if ida in simulator.qm_zone
        ]
        self.Z = torch.tensor(Z, device=self.device)
        self.register_buffer("nodes", amp.node_embedding(self.Z).detach())
        self.register_buffer(
            "charges_mm",
            torch.tensor(
                simulator.charges_mm, device=self.device, dtype=self.dtype
            ).unsqueeze(-1),
        )
        self.register_buffer(
            "mol_charge",
            torch.tensor(simulator.mol_charge, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "mol_size",
            torch.tensor([self.Z.shape[0]], device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "mm_zone_charges",
            torch.tensor(simulator.mm_zone_charges, device=self.device),
        )
        self.register_buffer(
            "qm_zone", torch.tensor(simulator.qm_zone, device=self.device)
        )
        self.register_buffer(
            "ion_zone", torch.tensor(simulator.ion_zone, device=self.device)
        )
        self.n_qm = self.qm_zone.shape[0]
        self.n_charges = self.mm_zone_charges.shape[0]
        self.register_buffer(
            "cutoff", torch.tensor(amp.cutoff, device=self.device, dtype=self.dtype)
        )
        self.register_buffer(
            "cutoff_esp",
            torch.tensor(amp.cutoff_esp, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_qmmm_esp",
            torch.tensor(amp.cutoff_qmmm_esp, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_qmmm_pol",
            torch.tensor(amp.cutoff_qmmm_pol, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "cutoff_nlist",
            torch.tensor(
                amp.cutoff_esp + pairlist_padding, device=self.device, dtype=self.dtype
            ),
        )
        self.register_buffer(
            "cutoff_qmmm_nlist",
            torch.tensor(
                amp.cutoff_qmmm_esp + pairlist_padding,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.register_buffer(
            "chunk_size",
            torch.tensor(chunk_size, device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "index_block_size",
            torch.tensor(block_size, device=self.device, dtype=torch.int64),
        )
        self.register_buffer(
            "step_count", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "n_nlist", torch.tensor(n_nlist, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_qm", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_mm", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_senders", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.register_buffer(
            "nlist_receivers", torch.tensor(0, device=self.device, dtype=torch.int64)
        )
        self.amp = amp.to(device=device, dtype=dtype).eval()

    def forward(self, positions, boxvectors):
        boxsize = (
            torch.diag(boxvectors * 10)
            .unsqueeze(0)
            .to(dtype=self.dtype, device=self.device)
        )  #
        positions = (positions * 10).to(dtype=self.dtype, device=self.device)
        graph = self._build_graph(positions, boxsize)
        graph = self.amp(graph)
        self.step_count = self.step_count + 1
        return graph.V_total.squeeze()

    def _build_graph(self, positions, boxsize):
        coords_qm = positions[self.qm_zone]
        coords_mm = positions[self.mm_zone_charges]
        if self.step_count % self.n_nlist == 0:
            self.nlist_qm, self.nlist_mm = self.build_nlist_qmmm_iteratively(
                coords_qm, coords_mm, boxsize
            )
            trius = torch.triu_indices(
                self.n_qm, self.n_qm, offset=1, dtype=torch.long, device=self.device
            )
            senders_qm, receivers_qm = trius[0], trius[1]
            self.nlist_senders, self.nlist_receivers = self.build_nlist(
                coords_qm, coords_qm, boxsize, senders_qm, receivers_qm
            )
        R1_qm, Rx1_qm, senders_qm, receivers_qm = self.prepare_distances_qm(
            coords_qm, boxsize, self.nlist_senders, self.nlist_receivers
        )
        R1, R2, Rx1, Rx2, senders, receivers = self.prepare_qm_indices(
            R1_qm, Rx1_qm, senders_qm, receivers_qm
        )
        R1_esp, R2_esp, Rx1_esp, Rx2_esp, senders_esp, receivers_esp = (
            self.prepare_esp_indices(R1_qm, Rx1_qm, senders_qm, receivers_qm)
        )
        R1_qmmm, Rx1_qmmm, indices_qm, indices_mm = self.prepare_distances_qmmm(
            coords_qm, coords_mm, boxsize, self.nlist_qm, self.nlist_mm
        )
        (
            R1_qmmm_esp,
            Rx1_qmmm_esp,
            Rx2_qmmm_esp,
            indices_qm_esp,
            indices_mm_esp,
            R1_qmmm_pol,
            Rx1_qmmm_pol,
            Rx2_qmmm_pol,
            indices_qm_pol,
            indices_mm_pol,
        ) = self.prepare_qmmm_indices(R1_qmmm, Rx1_qmmm, indices_qm, indices_mm)
        mm_monos_esp, mm_monos_pol = (
            self.charges_mm[indices_mm_esp],
            self.charges_mm[indices_mm_pol],
        )
        graph = Graph(
            Z=self.Z,
            nodes=self.nodes,
            coords_qm=coords_qm,
            mm_monos_esp=mm_monos_esp,
            mm_monos_pol=mm_monos_pol,
            mol_charge=self.mol_charge,
            mol_size=self.mol_size,
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
            batch_index_esp=torch.empty(0),
            R1_qmmm_esp=R1_qmmm_esp,
            Rx1_qmmm_esp=Rx1_qmmm_esp,
            Rx2_qmmm_esp=Rx2_qmmm_esp,
            receivers_qmmm_esp=indices_qm_esp,
            qm_indices_qmmm_esp=torch.empty(0),
            R1_qmmm_pol=R1_qmmm_pol,
            Rx1_qmmm_pol=Rx1_qmmm_pol,
            Rx2_qmmm_pol=Rx2_qmmm_pol,
            receivers_qmmm_pol=indices_qm_pol,
            md_mode=True,
            n_channels=self.amp.n_channels,
        )
        return graph

    def prepare_distances_qm(self, coords_qm, boxsize, senders, receivers):
        R1_qm, Rx1_qm = min_image(coords_qm, boxsize, senders, receivers)
        return R1_qm, Rx1_qm, senders, receivers

    def prepare_distances_qmmm(
        self, coords_qm, coords_mm, boxsize, indices_qm, indices_mm
    ):
        R1_qmmm, Rx1_qmmm = min_image_qmmm(
            coords_qm, coords_mm, boxsize, indices_qm, indices_mm
        )
        return R1_qmmm, Rx1_qmmm, indices_qm, indices_mm

    def prepare_qm_indices(self, R1_qm, Rx1_qm, senders_qm, receivers_qm):
        cutoff_indices = torch.where(R1_qm < self.amp.cutoff)[0]
        R1 = torch.index_select(R1_qm, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1_qm, dim=0, index=cutoff_indices)
        senders_qm = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers_qm = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R1 = torch.cat((R1, R1))
        R2 = torch.square(R1)
        Rx1 = torch.cat((Rx1, -Rx1), dim=0) / R1
        Rx2 = build_Rx2(Rx1)
        Rx1, Rx2 = Rx1.unsqueeze(1), Rx2.unsqueeze(1)
        senders = torch.cat((senders_qm, receivers_qm))
        receivers = torch.cat((receivers_qm, senders_qm))
        return R1, R2, Rx1, Rx2, senders, receivers

    def prepare_esp_indices(self, R1, Rx1, senders_qm, receivers_qm):
        cutoff_indices = torch.where(R1 < self.amp.cutoff_esp)[0]
        R1 = torch.index_select(R1, dim=0, index=cutoff_indices)
        Rx1 = torch.index_select(Rx1, dim=0, index=cutoff_indices)
        senders = torch.index_select(senders_qm, dim=0, index=cutoff_indices)
        receivers = torch.index_select(receivers_qm, dim=0, index=cutoff_indices)
        R2 = torch.square(R1)
        Rx2 = build_Rx2(Rx1)
        return R1, R2, Rx1, Rx2, senders, receivers

    def prepare_qmmm_indices(self, R1, Rx1, indices_qm, indices_mm):
        cutoff_indices_esp = torch.where(R1 < self.amp.cutoff_qmmm_esp)[0]
        R1_qmmm_esp = torch.index_select(R1, dim=0, index=cutoff_indices_esp)
        Rx1_qmmm_esp = torch.index_select(Rx1, dim=0, index=cutoff_indices_esp)
        Rx2_qmmm_esp = build_Rx2(Rx1_qmmm_esp)
        indices_qm_esp = torch.index_select(indices_qm, dim=0, index=cutoff_indices_esp)
        indices_mm_esp = torch.index_select(indices_mm, dim=0, index=cutoff_indices_esp)
        cutoff_indices_pol = torch.where(R1_qmmm_esp < self.amp.cutoff_qmmm_pol)[0]
        R1_qmmm_pol = torch.index_select(R1_qmmm_esp, dim=0, index=cutoff_indices_pol)
        Rx1_qmmm_pol = (
            torch.index_select(Rx1_qmmm_esp, dim=0, index=cutoff_indices_pol)
            / R1_qmmm_pol
        )
        Rx2_qmmm_pol = build_Rx2(Rx1_qmmm_pol)
        indices_qm_pol = torch.index_select(
            indices_qm_esp, dim=0, index=cutoff_indices_pol
        )
        indices_mm_pol = torch.index_select(
            indices_mm_esp, dim=0, index=cutoff_indices_pol
        )
        return (
            R1_qmmm_esp,
            Rx1_qmmm_esp,
            Rx2_qmmm_esp,
            indices_qm_esp,
            indices_mm_esp,
            R1_qmmm_pol,
            Rx1_qmmm_pol,
            Rx2_qmmm_pol,
            indices_qm_pol,
            indices_mm_pol,
        )

    def build_nlist_qmmm_iteratively(self, positions_a, positions_b, boxsize):
        with torch.no_grad():
            nlist_a, nlist_b = [], []
            a, b = min(self.n_qm, self.index_block_size), min(
                self.n_charges, self.index_block_size
            )
            index_matrix = torch.full((a, b), 1, device=self.device, dtype=torch.bool)
            block_indices_qm = torch.arange(0, self.n_qm, self.index_block_size)
            block_indices_mm = torch.arange(0, self.n_charges, self.index_block_size)
            for offset_index_qm in block_indices_qm:
                end_index_qm = self.index_block_size
                if (offset_index_qm + self.index_block_size) > self.n_qm:
                    end_index_qm = self.n_qm % self.index_block_size
                for offset_index_mm in block_indices_mm:
                    end_index_mm = self.index_block_size
                    if (offset_index_mm + self.index_block_size) > self.n_charges:
                        end_index_mm = self.n_charges % self.index_block_size
                    indices_qm, indices_mm = torch.where(
                        index_matrix[:end_index_qm, :end_index_mm]
                    )
                    indices_qm = indices_qm + offset_index_qm
                    indices_mm = indices_mm + offset_index_mm
                    R1, Rx1 = min_image_block(
                        positions_a[indices_qm], positions_b[indices_mm], boxsize
                    )
                    cutoff_indices = torch.where(R1.squeeze() < self.cutoff_qmmm_nlist)[
                        0
                    ]
                    indices_qm = torch.index_select(
                        indices_qm, dim=0, index=cutoff_indices
                    )
                    indices_mm = torch.index_select(
                        indices_mm, dim=0, index=cutoff_indices
                    )
                    nlist_a.append(indices_qm)
                    nlist_b.append(indices_mm)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)

    def build_nlist(self, positions_a, positions_b, boxsize, indices_a, indices_b):
        with torch.no_grad():
            chunks_a, chunks_b = chunkify(indices_a, self.chunk_size), chunkify(
                indices_b, self.chunk_size
            )
            nlist_a, nlist_b = [], []
            for chunk_a, chunk_b in zip(chunks_a, chunks_b):
                R1, Rx1 = min_image_block(
                    positions_a[chunk_a], positions_b[chunk_b], boxsize
                )
                cutoff_indices = torch.where(R1.squeeze() < self.cutoff_nlist)[0]
                chunk_a_nlist = torch.index_select(chunk_a, dim=0, index=cutoff_indices)
                chunk_b_nlist = torch.index_select(chunk_b, dim=0, index=cutoff_indices)
                nlist_a.append(chunk_a_nlist)
                nlist_b.append(chunk_b_nlist)
        return torch.cat(nlist_a, dim=0), torch.cat(nlist_b, dim=0)


# Assuming an orthorhombic box.
def to_fractional(coords, boxsize):
    return coords / boxsize


def from_fractional(coords, boxsize):
    return coords * boxsize


def min_image(coords, boxsize, senders, receivers):
    coords_a, coords_b = coords[senders], coords[receivers]
    return min_image_block(coords_a, coords_b, boxsize)


def min_image_qmmm(coords_qm, coords_mm, boxsize, indices_qm, indices_mm):
    coords_a, coords_b = coords_qm[indices_qm], coords_mm[indices_mm]
    return min_image_block(coords_a, coords_b, boxsize)


def min_image_block(coords_a, coords_b, boxsize):
    Rx1 = coords_b - coords_a
    Rx1 = Rx1 - from_fractional(torch.round(to_fractional(Rx1, boxsize)), boxsize)
    R1 = torch.linalg.norm(Rx1, dim=-1, keepdim=True)
    return R1, Rx1


def chunkify(indices, chunk_size):
    return torch.split(indices, chunk_size)
