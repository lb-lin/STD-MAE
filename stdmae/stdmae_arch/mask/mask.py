# stdmae/stdmae_arch/mask/mask.py
# 完整、稳健的 Mask 实现（方案 B）：graph random-walk mask + edge reconstruction aux loss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_
from typing import List, Optional

from basicts.utils.serialization import load_adj

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from ..graph_random_walk_masker import GraphRandomWalkMasker
from ..edge_recon_head import EdgeReconHead


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):
    """
    Mask module (STD-MAE) with:
      - graph random-walk spatial mask (per-sample or shared)
      - optional edge reconstruction auxiliary loss (BCE)
    Notes:
      - This module expects forward(history_data) where forward permutes history_data
        into [B, N, C, L] internally (we keep that convention).
    """

    def __init__(
        self,
        patch_size,
        in_channel,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout,
        mask_ratio,
        encoder_depth,
        decoder_depth,
        spatial=False,
        mode="pre-train",
        **kwargs,
    ):
        super().__init__()
        assert mode in ["pre-train", "forecasting"]
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial = spatial
        self.selected_feature = 0

        # graph & aux settings
        self.use_graph_mask = kwargs.get("use_graph_mask", True)
        self.aux_adj_loss = kwargs.get("aux_adj_loss", True)
        self.edge_sample_ratio = float(kwargs.get("edge_sample_ratio", 0.5))
        self.lambda_a = float(kwargs.get("lambda_a", 0.0))

        self.graph_walk_len = int(kwargs.get("graph_walk_len", 4))
        self.graph_num_walks = int(kwargs.get("graph_num_walks", 8))
        self.graph_p = float(kwargs.get("graph_p", 1.0))
        self.graph_q = float(kwargs.get("graph_q", 1.0))
        self.graph_seed = int(kwargs.get("graph_seed", 0))

        self.mask_strategy = kwargs.get("mask_strategy", "graph")  # 'graph' or 'random'
        self.mask_batch_mode = kwargs.get("mask_batch_mode", "per-sample")  # 'per-sample' or 'shared'

        # container for graph components (initialized below if needed)
        self.graph_masker: Optional[GraphRandomWalkMasker] = None
        self.A_np: Optional[np.ndarray] = None
        self.edge_head: Optional[EdgeReconHead] = None
        self.aux_loss = None

        # core modules
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None

        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding()

        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)

        # mask tokens: raw-level (in_channel) and embedding-level (embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, in_channel))      # for raw-level masking (optional)
        self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))   # for decoder filling

        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)

        self.initialize_weights()

        # load adjacency and setup GraphRandomWalkMasker & EdgeReconHead if needed
        if self.spatial and self.use_graph_mask and self.mask_strategy == "graph":
            dataset_name = kwargs.get("dataset_name", "PEMS04")
            adj_path = kwargs.get("graph_adj_path", f"datasets/{dataset_name}/adj_mx.pkl")
            # load_adj returns something like (adj_mx, some_meta). We support both list/ndarray.
            adj_mx, _ = load_adj(adj_path, "doubletransition")
            if isinstance(adj_mx, (list, tuple)):
                adj = adj_mx[0]
            else:
                adj = adj_mx
            adj = np.asarray(adj)
            self.graph_masker = GraphRandomWalkMasker(
                adj=adj,
                mask_ratio=self.mask_ratio,
                walk_len=self.graph_walk_len,
                num_walks=self.graph_num_walks,
                p=self.graph_p,
                q=self.graph_q,
                seed=self.graph_seed,
            )
            self.A_np = (adj > 0).astype(np.float32)
            self.edge_head = EdgeReconHead(d_model=self.embed_dim)

    def initialize_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.mask_token_emb, std=0.02)

    # --------------------------
    # encoding
    # --------------------------
    def encoding(self, long_term_history: torch.Tensor, mask: bool = True):
        """
        Input expected: long_term_history shape [B, N, C, L] (this function assumes that).
        Returns:
          hidden_states_unmasked, unmasked_token_index, masked_token_index, walk_edges_batch
        Note:
          - For spatial branch: unmasked_token_index and masked_token_index are lists of lists (per sample).
          - For temporal branch: they are plain lists (old behavior).
        """
        if mask:
            if self.spatial:
                # patches: [B, N, d, P] -> [B, N, P, d]
                patches = self.patch_embedding(long_term_history)
                patches = patches.transpose(-1, -2)
                patches, self.pos_mat = self.positional_encoding(patches)  # pos_mat: [B, N, P, d]
                batch_size, num_nodes, num_time, num_dim = patches.shape

                # decide masks per-sample or shared
                masked_token_index_batch = []
                unmasked_token_index_batch = []
                walk_edges_batch = []

                if self.use_graph_mask and (self.graph_masker is not None) and (self.mask_strategy == "graph"):
                    if self.mask_batch_mode == "shared":
                        m_nodes, w_edges = self.graph_masker.sample()
                        m_nodes = np.asarray(m_nodes, dtype=np.int64)
                        w_edges = np.asarray(w_edges, dtype=np.int64)
                        mask_list = m_nodes.tolist()
                        unmask_list = [i for i in range(num_nodes) if i not in set(mask_list)]
                        unmask_list = sorted(unmask_list)
                        for _ in range(batch_size):
                            masked_token_index_batch.append(mask_list)
                            unmasked_token_index_batch.append(unmask_list)
                            walk_edges_batch.append(w_edges.copy())
                    else:
                        for _ in range(batch_size):
                            m_nodes, w_edges = self.graph_masker.sample()
                            m_nodes = np.asarray(m_nodes, dtype=np.int64)
                            w_edges = np.asarray(w_edges, dtype=np.int64)
                            mlist = m_nodes.tolist()
                            ulist = [i for i in range(num_nodes) if i not in set(mlist)]
                            ulist = sorted(ulist)
                            masked_token_index_batch.append(mlist)
                            unmasked_token_index_batch.append(ulist)
                            walk_edges_batch.append(w_edges.copy())
                else:
                    # fallback uniform random
                    mg = MaskGenerator(num_nodes, self.mask_ratio)
                    unmask, mask = mg.uniform_rand()
                    masked_token_index_batch = [mask for _ in range(batch_size)]
                    unmasked_token_index_batch = [unmask for _ in range(batch_size)]
                    walk_edges_batch = [np.zeros((0, 2), dtype=np.int64) for _ in range(batch_size)]

                # Apply raw-level mask if you want (optional). Here we do NOT overwrite raw inputs by default.
                # If you want to replace raw time series at masked nodes, uncomment the following:
                # long_term_history_masked = long_term_history.clone()
                # for b in range(batch_size):
                #     idx = torch.as_tensor(masked_token_index_batch[b], device=long_term_history.device, dtype=torch.long)
                #     if idx.numel() > 0:
                #         long_term_history_masked[b, idx, :, :] = self.mask_token  # must match in_channel if used
                # patches = self.patch_embedding(long_term_history_masked)

                # Build encoder input per-sample, then pad to max_keep
                keeps = [len(u) for u in unmasked_token_index_batch]
                max_keep = max(keeps) if len(keeps) > 0 else 0
                B_, P_, _, d_ = patches.shape[0], patches.shape[2], max_keep, patches.shape[3]
                enc_padded = patches.new_zeros((batch_size, patches.shape[2], max_keep, patches.shape[3]))
                enc_key_padding_mask = torch.ones((batch_size, patches.shape[2], max_keep), dtype=torch.bool, device=patches.device)
                for b in range(batch_size):
                    keep_idx = torch.as_tensor(unmasked_token_index_batch[b], device=patches.device, dtype=torch.long)
                    n_k = keep_idx.numel()
                    if n_k > 0:
                        enc_padded[b, :, :n_k, :] = patches[b:b+1, keep_idx, :, :].squeeze(0)
                        enc_key_padding_mask[b, :, :n_k] = False

                # pass to encoder (try key_padding_mask if supported)
                try:
                    hidden_states_unmasked = self.encoder(enc_padded, key_padding_mask=enc_key_padding_mask)
                except TypeError:
                    hidden_states_unmasked = self.encoder(enc_padded)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, patches.shape[2], -1, self.embed_dim)

                return hidden_states_unmasked, unmasked_token_index_batch, masked_token_index_batch, walk_edges_batch

            else:
                # temporal branch (unchanged)
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)
                patches, self.pos_mat = self.positional_encoding(patches)
                batch_size, num_nodes, num_time, num_dim = patches.shape
                mg = MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = mg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :]
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
                return hidden_states_unmasked, unmasked_token_index, masked_token_index, None
        else:
            # forecasting
            batch_size, num_nodes, _, _ = long_term_history.shape
            patches = self.patch_embedding(long_term_history)
            patches = patches.transpose(-1, -2)
            patches, self.pos_mat = self.positional_encoding(patches)
            encoder_input = patches
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)
            hidden_states_unmasked = self.encoder(encoder_input)
            if self.spatial:
                hidden_states_unmasked = hidden_states_unmasked.transpose(-2, -3)
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)
            return hidden_states_unmasked, None, None, None

    # --------------------------
    # decoding
    # --------------------------
    def decoding(self, hidden_states_unmasked, unmasked_token_index, masked_token_index):
        """
        Arguments:
          hidden_states_unmasked: [B, P, N_unmask, d]
          unmasked_token_index, masked_token_index: per-sample lists or shared lists
        Returns:
          reconstruction_full [B, P, N, patch_size], hidden_states_full [B, P, N, d]
        """
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)
        if self.spatial:
            batch_size, num_time, _, _ = hidden_states_unmasked.shape
            pos_perm = self.pos_mat.permute(0, 2, 1, 3)  # [B, P, N, d]
            hidden_states_full_list = []
            for b in range(batch_size):
                # support per-sample lists
                unmask_b = unmasked_token_index[b] if isinstance(unmasked_token_index[0], list) else unmasked_token_index
                mask_b = masked_token_index[b] if isinstance(masked_token_index[0], list) else masked_token_index

                h_unmasked_b = hidden_states_unmasked[b:b+1, :, :, :]            # [1,P,N_unmask,d]
                if len(unmask_b) > 0:
                    pos_unmask = pos_perm[b:b+1, :, unmask_b, :]                 # [1,P,N_unmask,d]
                    h_unmasked_b = h_unmasked_b + pos_unmask

                if len(mask_b) > 0:
                    pos_mask = pos_perm[b:b+1, :, mask_b, :]                     # [1,P,N_mask,d]
                    mask_token = self.mask_token_emb.expand(1, num_time, pos_mask.shape[2], self.embed_dim)
                    h_masked_b = pos_mask + mask_token
                else:
                    h_masked_b = torch.zeros((1, num_time, 0, self.embed_dim), device=hidden_states_unmasked.device)

                h_full_b = torch.cat([h_unmasked_b, h_masked_b], dim=2)          # [1,P,N,d]
                hidden_states_full_list.append(h_full_b)
            hidden_states_full = torch.cat(hidden_states_full_list, dim=0)       # [B,P,N,d]

            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_time, -1, self.embed_dim))
        else:
            # temporal branch (keeps former logic)
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            unmasked_seq = [i for i in range(0, len(masked_token_index) + num_time) if i not in masked_token_index]
            hidden_states_masked = self.pos_mat[:, :, masked_token_index, :] + self.mask_token_emb.expand(batch_size, num_nodes, len(masked_token_index), self.embed_dim)
            hidden_states_unmasked = hidden_states_unmasked + self.pos_mat[:, :, unmasked_seq, :]
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)
            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))
        return reconstruction_full, hidden_states_full

    # --------------------------
    # get reconstructed masked tokens (per-sample)
    # --------------------------
    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        # reconstruction_full: [B, P, N, patch_size]
        if self.spatial:
            batch_size, num_time, num_nodes, _ = reconstruction_full.shape
            recon_list = []
            label_list = []
            # label_full: [B, N, P, L]
            label_full = (real_value_full.permute(0, 3, 1, 2)
                          .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
                          .transpose(1, 2))
            for b in range(batch_size):
                mask_idx_b = masked_token_index[b] if isinstance(masked_token_index[0], list) else masked_token_index
                if len(mask_idx_b) == 0:
                    recon_list.append(reconstruction_full.new_zeros((1, num_time, 0)))
                    label_list.append(reconstruction_full.new_zeros((1, num_time, 0)))
                    continue
                m_len = len(mask_idx_b)
                recon_b = reconstruction_full[b:b+1, :, -m_len:, :].view(1, num_time, -1)
                recon_list.append(recon_b)
                idx = torch.as_tensor(mask_idx_b, device=label_full.device, dtype=torch.long)
                label_b = label_full[b:b+1, idx, :, :].transpose(1, 2).contiguous().view(1, num_time, -1)
                label_list.append(label_b)
            reconstruction_masked_tokens = torch.cat(recon_list, dim=0)
            label_masked_tokens = torch.cat(label_list, dim=0)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape
            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :].view(batch_size, num_nodes, -1).transpose(1, 2)
            label_full = (real_value_full.permute(0, 3, 1, 2)
                          .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
                          .transpose(1, 2))
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()
            label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)
            return reconstruction_masked_tokens, label_masked_tokens

    # --------------------------
    # compute auxiliary edge loss (mapping node ids -> H_pool index)
    # --------------------------
    def _compute_aux_loss(self, hidden_states_full, unmasked_token_index, masked_token_index, walk_edges_batch):
        self.aux_loss = None
        if (not self.training) or (not self.aux_adj_loss):
            return
        if self.graph_masker is None or self.A_np is None:
            return
        if walk_edges_batch is None:
            return

        if hidden_states_full.dim() == 4:
            H_pool = hidden_states_full.mean(dim=1)  # [B, N, D]
        else:
            H_pool = hidden_states_full

        A_t = torch.from_numpy(self.A_np).to(H_pool.device)
        batch_size = H_pool.shape[0]
        total_loss = 0.0
        valid = 0
        for b in range(batch_size):
            edges_np = walk_edges_batch[b]
            if edges_np is None or edges_np.shape[0] == 0:
                continue
            # node order:
            unmask_b = unmasked_token_index[b] if isinstance(unmasked_token_index[0], list) else unmasked_token_index
            mask_b = masked_token_index[b] if isinstance(masked_token_index[0], list) else masked_token_index
            node_order = list(unmask_b) + list(mask_b)
            node_pos = {int(n): idx for idx, n in enumerate(node_order)}

            E = edges_np.shape[0]
            take = max(1, int(E * self.edge_sample_ratio))
            sel = np.random.choice(E, size=take, replace=False)
            edges_sel = edges_np[sel]  # original node ids

            feat_idx = []
            label_idx = []
            for (u, v) in edges_sel:
                u = int(u); v = int(v)
                if (u in node_pos) and (v in node_pos):
                    feat_idx.append([node_pos[u], node_pos[v]])
                    label_idx.append([u, v])
            if len(feat_idx) == 0:
                continue

            edges_pos = torch.as_tensor(feat_idx, device=H_pool.device, dtype=torch.long)
            edges_nodes = torch.as_tensor(label_idx, device=H_pool.device, dtype=torch.long)

            logits = self.edge_head(H_pool[b:b+1, :, :], edges_pos)  # [1, E']
            y = A_t[edges_nodes[:, 0], edges_nodes[:, 1]].unsqueeze(0)  # [1, E']
            loss_A = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss_A
            valid += 1

        if valid > 0:
            self.aux_loss = total_loss / float(valid)
        else:
            self.aux_loss = None

    # --------------------------
    # forward
    # --------------------------
    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # history_data expected [B, L, N, C] as in original repo → permute to [B, N, C, L]
        history_data = history_data.permute(0, 2, 3, 1)
        self.aux_loss = None

        if self.mode == "pre-train":
            hidden_states_unmasked, unmasked_token_index, masked_token_index, walk_edges_batch = self.encoding(history_data)
            reconstruction_full, hidden_states_full = self.decoding(hidden_states_unmasked, unmasked_token_index, masked_token_index)
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full, history_data, unmasked_token_index, masked_token_index
            )
            if self.spatial and self.aux_adj_loss and (self.graph_masker is not None):
                self._compute_aux_loss(hidden_states_full, unmasked_token_index, masked_token_index, walk_edges_batch)
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full


# quick manual test entry
if __name__ == "__main__":
    import sys
    from torchsummary import summary
    GPU = sys.argv[-1] if len(sys.argv) == 2 else "0"
    device = torch.device(f"cuda:{GPU}") if torch.cuda.is_available() else torch.device("cpu")
    model = Mask(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        mask_ratio=0.25,
        encoder_depth=2,
        decoder_depth=1,
        mode="pre-train",
        spatial=True,
    ).to(device)
    summary(model, (10, 100, 1), device=device)
