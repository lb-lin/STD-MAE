import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from timm.models.vision_transformer import trunc_normal_

from basicts.utils.serialization import load_adj

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from ..graph_random_walk_masker import GraphRandomWalkMasker
from ..edge_recon_head import EdgeReconHead


def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):

    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout,  mask_ratio, encoder_depth, decoder_depth,spatial=False, mode="pre-train", **kwargs):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.spatial=spatial
        self.selected_feature = 0
        self.use_graph_mask = kwargs.get("use_graph_mask", True)
        self.aux_adj_loss = kwargs.get("aux_adj_loss", True)
        self.edge_sample_ratio = kwargs.get("edge_sample_ratio", 0.5)
        self.lambda_a = kwargs.get("lambda_a", 0.0)
        self.graph_walk_len = kwargs.get("graph_walk_len", 4)
        self.graph_num_walks = kwargs.get("graph_num_walks", 8)
        self.graph_p = kwargs.get("graph_p", 1.0)
        self.graph_q = kwargs.get("graph_q", 1.0)
        self.graph_seed = kwargs.get("graph_seed", 0)
        self.mask_strategy = kwargs.get("mask_strategy", "graph")
        self.mask_batch_mode = kwargs.get("mask_batch_mode", "per-sample")
        self.graph_masker = None
        self.A_np = None
        self.edge_head = None
        self.aux_loss = None

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat=None
        # encoder specifics
        # # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # # positional encoding
        self.positional_encoding = PositionalEncoding()

        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, in_channel))
        self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

        if self.spatial and self.use_graph_mask:
            dataset_name = kwargs.get("dataset_name", "PEMS04")
            adj_path = kwargs.get("graph_adj_path", f"datasets/{dataset_name}/adj_mx.pkl")
            adj_mx, _ = load_adj(adj_path, "doubletransition")
            adj = adj_mx[0] if isinstance(adj_mx, (list, tuple)) else adj_mx
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
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.mask_token_emb, std=.02)

    def encoding(self, long_term_history, mask=True):
        """

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, C, P * L],
                                                which is used in the Pre-training.
                                                P is the number of patches.
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        # patchify and embed input
        if mask:
            if self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time,num_dim  =  patches.shape
                patches,self.pos_mat = self.positional_encoding(patches)        # mask
                pos_perm = self.pos_mat.permute(0, 2, 1, 3)  # B, P, N, d

                masked_token_index_batch = []
                unmasked_token_index_batch = []
                walk_edges_batch = []

                if self.use_graph_mask and self.graph_masker is not None and self.mask_strategy == "graph":
                    if self.mask_batch_mode == "shared":
                        mask_nodes_np, walk_edges_np = self.graph_masker.sample()
                        mask_nodes_np = np.asarray(mask_nodes_np, dtype=np.int64)
                        walk_edges_np = np.asarray(walk_edges_np, dtype=np.int64)
                        for _ in range(batch_size):
                            masked_token_index_batch.append(mask_nodes_np.tolist())
                            unmasked = [i for i in range(num_nodes) if i not in set(mask_nodes_np.tolist())]
                            unmasked_token_index_batch.append(sorted(unmasked))
                            walk_edges_batch.append(walk_edges_np.copy())
                    else:
                        for _ in range(batch_size):
                            mask_nodes_np, walk_edges_np = self.graph_masker.sample()
                            mask_nodes_np = np.asarray(mask_nodes_np, dtype=np.int64)
                            walk_edges_np = np.asarray(walk_edges_np, dtype=np.int64)
                            masked_token_index_batch.append(mask_nodes_np.tolist())
                            unmasked = [i for i in range(num_nodes) if i not in set(mask_nodes_np.tolist())]
                            unmasked_token_index_batch.append(sorted(unmasked))
                            walk_edges_batch.append(walk_edges_np.copy())
                else:
                    Maskg=MaskGenerator(patches.shape[1], self.mask_ratio)
                    unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                    masked_token_index_batch = [masked_token_index for _ in range(batch_size)]
                    unmasked_token_index_batch = [unmasked_token_index for _ in range(batch_size)]
                    walk_edges_batch = [np.zeros((0, 2), dtype=np.int64) for _ in range(batch_size)]

                encoder_inputs = []
                for b in range(batch_size):
                    encoder_inputs.append(patches[b:b + 1, unmasked_token_index_batch[b], :, :])
                encoder_input = torch.cat(encoder_inputs, dim=0).transpose(-2,-3)
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size,num_time, -1, self.embed_dim)

                return hidden_states_unmasked,  unmasked_token_index_batch, masked_token_index_batch, walk_edges_batch

            if not self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)  # B, N, P, d
                batch_size, num_nodes, num_time,num_dim  =  patches.shape

                # positional embedding
                patches,self.pos_mat = self.positional_encoding(patches)        # mask
                Maskg=MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :]
                #print(encoder_input.shape)
                hidden_states_unmasked = self.encoder(encoder_input)
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)

                return hidden_states_unmasked,  unmasked_token_index, masked_token_index, None

        else:
            batch_size, num_nodes, _, _ = long_term_history.shape
            # patchify and embed input
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)         # B, N, P, d
            # positional embedding
            patches,self.pos_mat = self.positional_encoding(patches)# B, N, P, d
            #print(self.pos_mat.shape)
            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches# B, N, P, d
            if self.spatial:
                encoder_input=encoder_input.transpose(-2,-3)# B,  P,N, d
            hidden_states_unmasked = self.encoder(encoder_input)# B,  P,N, d/# B, N, P, d
            if self.spatial:
                hidden_states_unmasked=hidden_states_unmasked.transpose(-2,-3)# B, N, P, d
            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(batch_size, num_nodes, -1, self.embed_dim)#B, N, P, d
            return hidden_states_unmasked, unmasked_token_index, masked_token_index, None
        # encoding

        return hidden_states_unmasked,  unmasked_token_index, masked_token_index, None

    def decoding(self, hidden_states_unmasked, unmasked_token_index, masked_token_index):

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)# B, N, P, d/# B,P, N,  d
        # B,N*r,P,d
        if self.spatial:
            # TO WORK SPATIAL:
            batch_size,  num_time,num_nodes, _ = hidden_states_unmasked.shape# B,P, N,  d
            pos_perm = self.pos_mat.permute(0, 2, 1, 3)  # B, P, N, d
            hidden_states_full_list = []
            for b in range(batch_size):
                masked_index_b = masked_token_index[b] if isinstance(masked_token_index, list) else masked_token_index
                unmasked_index_b = unmasked_token_index[b] if isinstance(unmasked_token_index, list) else unmasked_token_index
                hidden_states_masked = pos_perm[b:b+1, :, masked_index_b, :]
                hidden_states_masked=hidden_states_masked.transpose(-2,-3)# B, P, N*r, d

                hidden_states_masked+=self.mask_token_emb.expand(1, num_time, len(masked_index_b), hidden_states_unmasked.shape[-1])# B, P, N*r, d
                hidden_states_unmasked_b = hidden_states_unmasked[b:b+1, :, :, :] + pos_perm[b:b+1, :, unmasked_index_b, :].transpose(-2,-3)
                hidden_states_full_list.append(torch.cat([hidden_states_unmasked_b, hidden_states_masked], dim=-2))
            hidden_states_full = torch.cat(hidden_states_full_list, dim=0)

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)# B, P, N, d
            hidden_states_full = self.decoder_norm(hidden_states_full)# B, P, N, d
            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size,num_time, -1,  self.embed_dim))# B, P, N, L
        else:
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            unmasked_token_index=[i for i in range(0,len(masked_token_index)+num_time) if i not in masked_token_index ]
            hidden_states_masked = self.pos_mat[:,:,masked_token_index,:]
            hidden_states_masked+=self.mask_token_emb.expand(batch_size, num_nodes, len(masked_token_index), hidden_states_unmasked.shape[-1])
            hidden_states_unmasked+=self.pos_mat[:,:,unmasked_token_index,:]
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, N, P, d

            # decoding
            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)

            # prediction (reconstruction)
            reconstruction_full = self.output_layer(hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim))

        return reconstruction_full, hidden_states_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index,
                                        masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        if self.spatial:
            batch_size,  num_time,num_nodes, _ = reconstruction_full.shape# B, P, N, L
            mask_len = len(masked_token_index[0]) if isinstance(masked_token_index, list) else len(masked_token_index)
            reconstruction_masked_tokens = reconstruction_full[:, :, -mask_len:, :]     # B, P, r*N, L
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_time, -1)     # B, P, r*N*L

            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_list = []
            for b in range(batch_size):
                masked_idx_b = masked_token_index[b] if isinstance(masked_token_index, list) else masked_token_index
                label_masked = label_full[b:b + 1, masked_idx_b, :, :].transpose(1, 2).contiguous()
                label_masked_list.append(label_masked)
            label_masked_tokens = torch.cat(label_masked_list, dim=0).view(batch_size,  num_time,-1)  # B, P, r*N*L

            return reconstruction_masked_tokens, label_masked_tokens
        else:
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape

            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]     # B, N, r*P, d
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

            label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous() # B, N, r*P, d
            label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*d, N

            return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        self.aux_loss = None

        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index, walk_edges_batch = self.encoding(history_data)
            # decoding
            reconstruction_full, hidden_states_full = self.decoding(hidden_states_unmasked, unmasked_token_index, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)

            if self.training and self.aux_adj_loss and self.graph_masker is not None and walk_edges_batch is not None:
                H = hidden_states_full
                H_pool = H.mean(dim=1) if H.dim() == 4 else H  # [B, N, D]
                A_t = torch.from_numpy(self.A_np).to(H_pool.device)
                batch_size = H_pool.shape[0]
                total_loss = 0.0
                valid = 0
                for b in range(batch_size):
                    edges_np = walk_edges_batch[b]
                    if edges_np is None or edges_np.shape[0] == 0:
                        continue
                    E = edges_np.shape[0]
                    take = max(1, int(E * self.edge_sample_ratio))
                    sel = np.random.choice(E, size=take, replace=False)
                    edges_sel = edges_np[sel]
                    mapped = []
                    for (u, v) in edges_sel:
                        if 0 <= int(u) < H_pool.shape[1] and 0 <= int(v) < H_pool.shape[1]:
                            mapped.append([int(u), int(v)])
                    if len(mapped) == 0:
                        continue
                    edges_pos = torch.as_tensor(mapped, device=H_pool.device, dtype=torch.long)
                    logits = self.edge_head(H_pool[b:b+1, :, :], edges_pos)  # [1, E']
                    y = A_t[edges_pos[:,0], edges_pos[:,1]].unsqueeze(0)      # [1, E']
                    loss_A = F.binary_cross_entropy_with_logits(logits, y)
                    total_loss += loss_A
                    valid += 1
                if valid > 0:
                    self.aux_loss = total_loss / float(valid)

            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full

