"""LightGCN with differentiable edge deletion on the propagation graph."""

from __future__ import annotations

import numpy as np
import torch

from recbole.model.general_recommender import LightGCN


class LightGCNPerturbed(LightGCN):
    """LightGCN whose propagation graph can have edges deleted differentiably.

    Two properties matter:

    1. One parameter per *undirected* edge. The normalised adjacency stores each
       user-item edge twice, as (u, i) and (i, u). Giving those two entries
       independent weights lets the optimiser delete one direction and keep the
       other, which is not a graph. Here a single mask entry drives both.

    2. Degrees are recomputed after deletion. Eq. (4) perturbs the binary
       adjacency and the GNN normalises *that*, so removing a user's edges
       raises the weight on the ones they keep. Scaling an already-normalised
       matrix freezes the pre-attack degrees and loses a real part of the
       attack's effect.
    """

    def get_norm_adj_mat(self) -> torch.Tensor:
        """Build the graph representation used by both clean and attacked LightGCN.

        Each user-item interaction is one undirected edge, but the sparse adjacency
        matrix stores it twice: user -> item and item -> user. This method records
        both entries and maps them to one shared edge ID. Consequently, one deletion
        mask value always removes both directions of an interaction.

        It then builds the clean normalized adjacency by calling ``perturbed_adj``
        with an all-zero mask. This guarantees that the clean and attacked graphs use
        exactly the same construction and normalization logic.

        Overriding this RecBole method also avoids RecBole 1.2.1's incompatible call
        to SciPy's removed private method ``dok_matrix._update``.
        """
        # self.interaction_matrix comes from the dataset handed in, so this class
        # must be constructed with train_data._dataset
        inter = self.interaction_matrix
        edges = np.unique(
            np.stack(
                [
                    inter.row.astype(np.int64),
                    inter.col.astype(np.int64) + self.n_users,
                ]
            ),
            axis=1,
        )
        src, dst = edges[0], edges[1]
        self.n_edges = int(src.shape[0])

        # Both directions of every edge, sorted into coalesced order once so the
        # sparse tensor can be rebuilt each iteration without re-sorting.
        row = np.concatenate([src, dst])
        col = np.concatenate([dst, src])
        edge_of_entry = np.concatenate(
            [np.arange(self.n_edges), np.arange(self.n_edges)]
        )
        order = np.lexsort((col, row))

        self.register_buffer(
            "pert_index",
            torch.as_tensor(np.stack([row[order], col[order]]), dtype=torch.long),
        )
        self.register_buffer(
            "edge_of_entry", torch.as_tensor(edge_of_entry[order], dtype=torch.long)
        )

        # The clean graph is the empty perturbation, so both come from one path.
        return self.perturbed_adj(torch.zeros(self.n_edges))

    def perturbed_adj(self, deletion_mask: torch.Tensor) -> torch.Tensor:
        """D^-1/2 (A - deleted) D^-1/2, with D taken from the perturbed graph."""
        idx = self.pert_index
        keep = (1.0 - deletion_mask).to(device=idx.device, dtype=torch.float32)
        a = keep.index_select(0, self.edge_of_entry)          # [2E] binary-ish

        n_nodes = self.n_users + self.n_items
        deg = torch.zeros(n_nodes, device=idx.device, dtype=a.dtype).index_add(
            0, idx[0], a
        )
        d_inv_sqrt = (deg + 1e-7).pow(-0.5)
        vals = a * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
        return torch.sparse_coo_tensor(
            idx, vals, (n_nodes, n_nodes), device=idx.device
        ).coalesce()

    def forward(self, deletion_mask: torch.Tensor | None = None):
        """Standard LightGCN propagation; perturbed when a mask is supplied.

        RecBole's ``calculate_loss`` calls ``forward()`` with no arguments during
        training, which takes the clean path and is bit-for-bit the base model.
        """
        ego = self.get_ego_embeddings()
        adj = (
            self.norm_adj_matrix
            if deletion_mask is None
            else self.perturbed_adj(deletion_mask)
        )

        embeddings = [ego]
        h = ego
        for _ in range(self.n_layers):
            h = torch.sparse.mm(adj, h)
            embeddings.append(h)
        all_embeddings = torch.stack(embeddings, dim=1).mean(dim=1)
        return torch.split(all_embeddings, [self.n_users, self.n_items])
