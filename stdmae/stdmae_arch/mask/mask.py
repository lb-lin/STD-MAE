import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import trunc_normal_

from basicts.utils.serialization import load_adj

from .patch import PatchEmbedding
from .maskgenerator import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from ..graph_random_walk_masker import GraphRandomWalkMasker
from ..edge_recon_head import EdgeReconHead


def unshuffle(shuffled_tokens):
    """Reverse operation of a permutation (for completeness,保持原实现)."""
    dic = {}
    for k, v in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class Mask(nn.Module):
    """
    STD-MAE 的掩码模块：
    - 时间分支：保持原实现；
    - 空间分支：支持 graph random-walk 掩码 + 边重构辅助任务。
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
        assert mode in ["pre-train", "forecasting"], "Error mode."
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

        # ----------------- 图掩码 & 辅助 loss 相关超参 -----------------
        # 是否启用图掩码（仅 spatial=True 时有意义）
        self.use_graph_mask = kwargs.get("use_graph_mask", True)
        # 是否启用边重构辅助损失
        self.aux_adj_loss = kwargs.get("aux_adj_loss", True)
        # 每个 batch 中从随机游走边集中采样多少比例来算辅助 loss
        self.edge_sample_ratio = kwargs.get("edge_sample_ratio", 0.5)
        # 辅助 loss 的权重，真正加到总 loss 里由外层 masked_mae_with_aux 使用
        self.lambda_a = kwargs.get("lambda_a", 0.0)

        # random walk 参数
        self.graph_walk_len = kwargs.get("graph_walk_len", 4)
        self.graph_num_walks = kwargs.get("graph_num_walks", 8)
        self.graph_p = kwargs.get("graph_p", 1.0)
        self.graph_q = kwargs.get("graph_q", 1.0)
        self.graph_seed = kwargs.get("graph_seed", 0)

        # 掩码策略：graph / random
        self.mask_strategy = kwargs.get("mask_strategy", "graph")
        # 掩码在 batch 维度的共享方式：per-sample / shared
        self.mask_batch_mode = kwargs.get("mask_batch_mode", "per-sample")

        # 图信息
        self.graph_masker = None
        self.A_np = None       # 邻接矩阵 numpy 版本
        self.edge_head = None  # 边重构 head
        self.aux_loss = None   # 辅助 loss（供 loss wrapper 读取）

        # ----------------- 标准 STD-MAE 模块 -----------------
        # norm
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.pos_mat = None

        # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.positional_encoding = PositionalEncoding()

        # encoder / decoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)

        # 注意：保持两个 mask token：
        # - mask_token: 用于 raw 输入上（如果将来想把原始值也盖掉，类似 STMAE）
        # - mask_token_emb: 用于 decoder 侧的 embedding 填充
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, in_channel))
        self.mask_token_emb = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.output_layer = nn.Linear(embed_dim, patch_size)

        self.initialize_weights()

        # ----------------- 加载图结构并构造 graph_masker 和 edge_head -----------------
        if self.spatial and self.use_graph_mask:
            dataset_name = kwargs.get("dataset_name", "PEMS04")
            adj_path = kwargs.get("graph_adj_path", f"datasets/{dataset_name}/adj_mx.pkl")
            # load_adj 返回的 adj_mx 可能是 list/tuple，取第 0 个
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
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.mask_token_emb, std=.02)

    # ------------------------------------------------------------------
    # 编码：返回
    #  - hidden_states_unmasked: encoder 输出（只包含未遮节点）；
    #  - unmasked_token_index / masked_token_index：
    #       spatial=True 时 -> list[list[int]]，每个样本一份；
    #       spatial=False 时 -> list[int]；
    #  - walk_edges_batch: spatial=True 时为 list[np.ndarray[E,2]]，否则为 None。
    # ------------------------------------------------------------------
    def encoding(self, long_term_history, mask=True):
        """
        Args:
            long_term_history (torch.Tensor): [B, N, C, P * L]
            mask (bool): True pre-train / False forecasting
        """
        if mask:
            # =================== 空间分支（graph 掩码主要改动在这里） ===================
            if self.spatial:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)                # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape

                patches, self.pos_mat = self.positional_encoding(patches)  # self.pos_mat: B, N, P, d

                masked_token_index_batch = []
                unmasked_token_index_batch = []
                walk_edges_batch = []

                # --------- 决定掩码节点：graph / random，per-sample / shared ---------
                if self.use_graph_mask and self.graph_masker is not None and self.mask_strategy == "graph":
                    if self.mask_batch_mode == "shared":
                        # 一个掩码共享给整个 batch
                        mask_nodes_np, walk_edges_np = self.graph_masker.sample()
                        mask_nodes_np = np.asarray(mask_nodes_np, dtype=np.int64)
                        walk_edges_np = np.asarray(walk_edges_np, dtype=np.int64)
                        mask_list = mask_nodes_np.tolist()
                        unmask_list = [i for i in range(num_nodes) if i not in set(mask_list)]
                        unmask_list = sorted(unmask_list)
                        for _ in range(batch_size):
                            masked_token_index_batch.append(mask_list)
                            unmasked_token_index_batch.append(unmask_list)
                            walk_edges_batch.append(walk_edges_np.copy())
                    else:
                        # 每个样本各自采一个掩码
                        for _ in range(batch_size):
                            mask_nodes_np, walk_edges_np = self.graph_masker.sample()
                            mask_nodes_np = np.asarray(mask_nodes_np, dtype=np.int64)
                            walk_edges_np = np.asarray(walk_edges_np, dtype=np.int64)
                            mask_list = mask_nodes_np.tolist()
                            unmask_list = [i for i in range(num_nodes) if i not in set(mask_list)]
                            unmask_list = sorted(unmask_list)
                            masked_token_index_batch.append(mask_list)
                            unmasked_token_index_batch.append(unmask_list)
                            walk_edges_batch.append(walk_edges_np.copy())
                else:
                    # fallback：原来的随机节点掩码
                    Maskg = MaskGenerator(patches.shape[1], self.mask_ratio)
                    unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                    masked_token_index_batch = [masked_token_index for _ in range(batch_size)]
                    unmasked_token_index_batch = [unmasked_token_index for _ in range(batch_size)]
                    walk_edges_batch = [np.zeros((0, 2), dtype=np.int64) for _ in range(batch_size)]

                # --------- 构造 encoder 输入：只喂未遮节点 ---------
                encoder_inputs = []
                for b in range(batch_size):
                    idx = torch.as_tensor(
                        unmasked_token_index_batch[b],
                        device=patches.device,
                        dtype=torch.long,
                    )
                    encoder_inputs.append(patches[b:b + 1, idx, :, :])  # 1, N_unmask, P, d
                encoder_input = torch.cat(encoder_inputs, dim=0)        # B, N_unmask, P, d
                encoder_input = encoder_input.transpose(-2, -3)         # B, P, N_unmask, d

                hidden_states_unmasked = self.encoder(encoder_input)   # B, P, N_unmask, d
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(
                    batch_size, num_time, -1, self.embed_dim
                )  # B, P, N_unmask, d

                return hidden_states_unmasked, unmasked_token_index_batch, masked_token_index_batch, walk_edges_batch

            # =================== 时间分支（保持原实现） ===================
            else:
                patches = self.patch_embedding(long_term_history)  # B, N, d, P
                patches = patches.transpose(-1, -2)                # B, N, P, d
                batch_size, num_nodes, num_time, num_dim = patches.shape

                patches, self.pos_mat = self.positional_encoding(patches)
                Maskg = MaskGenerator(patches.shape[2], self.mask_ratio)
                unmasked_token_index, masked_token_index = Maskg.uniform_rand()
                encoder_input = patches[:, :, unmasked_token_index, :]  # B, N, P_unmask, d
                hidden_states_unmasked = self.encoder(encoder_input)    # B, N, P_unmask, d
                hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(
                    batch_size, num_nodes, -1, self.embed_dim
                )
                return hidden_states_unmasked, unmasked_token_index, masked_token_index, None

        else:
            # forecasting 模式：不做掩码
            batch_size, num_nodes, _, _ = long_term_history.shape
            patches = self.patch_embedding(long_term_history)     # B, N, d, P
            patches = patches.transpose(-1, -2)                   # B, N, P, d
            patches, self.pos_mat = self.positional_encoding(patches)  # B, N, P, d

            unmasked_token_index, masked_token_index = None, None
            encoder_input = patches
            if self.spatial:
                encoder_input = encoder_input.transpose(-2, -3)   # B, P, N, d

            hidden_states_unmasked = self.encoder(encoder_input)
            if self.spatial:
                hidden_states_unmasked = hidden_states_unmasked.transpose(-2, -3)  # B, N, P, d

            hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked).view(
                batch_size, num_nodes, -1, self.embed_dim
            )
            return hidden_states_unmasked, unmasked_token_index, masked_token_index, None

    # ------------------------------------------------------------------
    # 解码：补回被 mask 的节点，然后过 decoder → reconstruction_full
    # 返回：
    #   reconstruction_full: 空间分支 [B, P, N, L]；时间分支 [B, N, P, L]
    #   hidden_states_full:  decoder 输出（后面用于边重构）
    # ------------------------------------------------------------------
    def decoding(self, hidden_states_unmasked, unmasked_token_index, masked_token_index):
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        if self.spatial:
            # hidden_states_unmasked: [B, P, N_unmask, d]
            batch_size, num_time, num_nodes_unmask, _ = hidden_states_unmasked.shape
            # self.pos_mat: [B, N, P, d] → [B, P, N, d]
            pos_perm = self.pos_mat.permute(0, 2, 1, 3)

            hidden_states_full_list = []

            for b in range(batch_size):
                # 兼容 per-sample / shared 两种情况
                if isinstance(unmasked_token_index[0], list):
                    unmask_list = unmasked_token_index[b]
                else:
                    unmask_list = unmasked_token_index
                if isinstance(masked_token_index[0], list):
                    mask_list = masked_token_index[b]
                else:
                    mask_list = masked_token_index

                unmask_idx = torch.as_tensor(
                    unmask_list, device=hidden_states_unmasked.device, dtype=torch.long
                )
                mask_idx = torch.as_tensor(
                    mask_list, device=hidden_states_unmasked.device, dtype=torch.long
                )

                # 未遮节点：encoder 输出 + 对应位置编码
                h_unmasked_b = hidden_states_unmasked[b:b + 1, :, :, :]         # [1, P, N_unmask, d]
                pos_unmask = pos_perm[b:b + 1, :, unmask_idx, :]                # [1, P, N_unmask, d]
                h_unmasked_b = h_unmasked_b + pos_unmask                        # [1, P, N_unmask, d]

                # 遮住的节点：只用位置编码 + 一个 learnable mask_token_emb
                if mask_idx.numel() > 0:
                    pos_mask = pos_perm[b:b + 1, :, mask_idx, :]                # [1, P, N_mask, d]
                    mask_token = self.mask_token_emb.expand(
                        1, num_time, pos_mask.shape[2], self.embed_dim
                    )                                                           # [1, P, N_mask, d]
                    h_masked_b = pos_mask + mask_token                          # [1, P, N_mask, d]
                else:
                    h_masked_b = torch.zeros(
                        (1, num_time, 0, self.embed_dim),
                        device=hidden_states_unmasked.device,
                        dtype=hidden_states_unmasked.dtype,
                    )

                # 拼成完整节点序列：顺序为 [unmask..., mask...]
                h_full_b = torch.cat([h_unmasked_b, h_masked_b], dim=2)         # [1, P, N, d]
                hidden_states_full_list.append(h_full_b)

            hidden_states_full = torch.cat(hidden_states_full_list, dim=0)      # [B, P, N, d]

            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)          # [B, P, N, d]

            reconstruction_full = self.output_layer(
                hidden_states_full.view(batch_size, num_time, -1, self.embed_dim)
            )  # [B, P, N, L]
        else:
            # 时间分支：保持原逻辑
            batch_size, num_nodes, num_time, _ = hidden_states_unmasked.shape
            unmasked_token_index = [
                i for i in range(0, len(masked_token_index) + num_time)
                if i not in masked_token_index
            ]
            hidden_states_masked = self.pos_mat[:, :, masked_token_index, :]               # B, N, r*P, d
            hidden_states_masked = hidden_states_masked + self.mask_token_emb.expand(
                batch_size, num_nodes, len(masked_token_index), self.embed_dim
            )
            hidden_states_unmasked = hidden_states_unmasked + self.pos_mat[:, :, unmasked_token_index, :]
            hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)  # B, N, P, d

            hidden_states_full = self.decoder(hidden_states_full)
            hidden_states_full = self.decoder_norm(hidden_states_full)

            reconstruction_full = self.output_layer(
                hidden_states_full.view(batch_size, num_nodes, -1, self.embed_dim)
            )  # B, N, P, L

        return reconstruction_full, hidden_states_full

    # ------------------------------------------------------------------
    # 取出被 mask 的 token 及其 label（和原实现保持兼容）
    # ------------------------------------------------------------------
    def get_reconstructed_masked_tokens(
        self,
        reconstruction_full,
        real_value_full,
        unmasked_token_index,
        masked_token_index,
    ):
        if self.spatial:
            # reconstruction_full: [B, P, N, L]
            batch_size, num_time, num_nodes, _ = reconstruction_full.shape

            # 计算每个样本的 mask 长度（支持 per-sample / shared）
            if isinstance(masked_token_index[0], list):
                mask_lens = [len(m) for m in masked_token_index]
            else:
                mask_lens = [len(masked_token_index) for _ in range(batch_size)]

            # 取出重建的被 mask 节点（在 decoding 里我们保证它们在最后）
            recon_list = []
            for b in range(batch_size):
                m_len = mask_lens[b]
                if m_len == 0:
                    recon_list.append(
                        torch.zeros(
                            (1, num_time, 0),
                            device=reconstruction_full.device,
                            dtype=reconstruction_full.dtype,
                        )
                    )
                else:
                    recon_b = reconstruction_full[b:b + 1, :, -m_len:, :]  # [1, P, m_len, L]
                    recon_b = recon_b.view(1, num_time, -1)               # [1, P, m_len*L]
                    recon_list.append(recon_b)
            reconstruction_masked_tokens = torch.cat(recon_list, dim=0)   # [B, P, r*N*L]

            # label_full: [B, N, P, L]
            label_full = (
                real_value_full.permute(0, 3, 1, 2)
                .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
                .transpose(1, 2)
            )

            label_list = []
            for b in range(batch_size):
                if isinstance(masked_token_index[0], list):
                    mask_idx_b = masked_token_index[b]
                else:
                    mask_idx_b = masked_token_index
                if len(mask_idx_b) == 0:
                    label_list.append(
                        torch.zeros(
                            (1, num_time, 0),
                            device=label_full.device,
                            dtype=label_full.dtype,
                        )
                    )
                else:
                    idx = torch.as_tensor(mask_idx_b, device=label_full.device, dtype=torch.long)
                    label_b = label_full[b:b + 1, idx, :, :].transpose(1, 2).contiguous()
                    label_b = label_b.view(1, num_time, -1)  # [1, P, r*N*L]
                    label_list.append(label_b)
            label_masked_tokens = torch.cat(label_list, dim=0)           # [B, P, r*N*L]

            return reconstruction_masked_tokens, label_masked_tokens

        else:
            # 时间分支：保持原逻辑
            batch_size, num_nodes, num_time, _ = reconstruction_full.shape

            reconstruction_masked_tokens = reconstruction_full[:, :, len(unmasked_token_index):, :]
            reconstruction_masked_tokens = reconstruction_masked_tokens.view(
                batch_size, num_nodes, -1
            ).transpose(1, 2)  # B, r*P*L, N

            label_full = (
                real_value_full.permute(0, 3, 1, 2)
                .unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :]
                .transpose(1, 2)
            )  # B, N, P, L
            label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()
            label_masked_tokens = label_masked_tokens.view(
                batch_size, num_nodes, -1
            ).transpose(1, 2)

            return reconstruction_masked_tokens, label_masked_tokens

    # ------------------------------------------------------------------
    # 计算边重构辅助 loss，结果写到 self.aux_loss 里
    # ------------------------------------------------------------------
    def _compute_aux_loss(self, hidden_states_full, unmasked_token_index, masked_token_index, walk_edges_batch):
        self.aux_loss = None
        if (not self.training) or (not self.aux_adj_loss):
            return
        if self.graph_masker is None or self.A_np is None:
            return
        if walk_edges_batch is None:
            return

        # 对空间分支：hidden_states_full [B, P, N, D] → 节点池化后 [B, N, D]
        if hidden_states_full.dim() == 4:
            H_pool = hidden_states_full.mean(dim=1)
        else:
            H_pool = hidden_states_full
        A_t = torch.from_numpy(self.A_np).to(H_pool.device)

        batch_size, num_nodes, _ = H_pool.shape
        total_loss = 0.0
        valid = 0

        for b in range(batch_size):
            edges_np = walk_edges_batch[b]
            if edges_np is None or edges_np.shape[0] == 0:
                continue

            # 1) 构造当前样本的 node_order：unmask + mask
            if isinstance(unmasked_token_index[0], list):
                unmask_b = unmasked_token_index[b]
            else:
                unmask_b = unmasked_token_index
            if isinstance(masked_token_index[0], list):
                mask_b = masked_token_index[b]
            else:
                mask_b = masked_token_index

            node_order = unmask_b + mask_b
            node_pos = {int(n): idx for idx, n in enumerate(node_order)}

            # 2) 从随机游走边集中采样部分边
            E = edges_np.shape[0]
            take = max(1, int(E * self.edge_sample_ratio))
            sel = np.random.choice(E, size=take, replace=False)
            edges_sel = edges_np[sel]  # 原始节点 id

            feat_idx = []
            label_idx = []
            for (u, v) in edges_sel:
                u = int(u)
                v = int(v)
                if (u in node_pos) and (v in node_pos):
                    feat_idx.append([node_pos[u], node_pos[v]])  # H_pool 用
                    label_idx.append([u, v])                     # 邻接矩阵用
            if len(feat_idx) == 0:
                continue

            edges_pos = torch.as_tensor(feat_idx, device=H_pool.device, dtype=torch.long)
            edges_nodes = torch.as_tensor(label_idx, device=H_pool.device, dtype=torch.long)

            logits = self.edge_head(H_pool[b:b + 1, :, :], edges_pos)     # [1, E']
            y = A_t[edges_nodes[:, 0], edges_nodes[:, 1]].unsqueeze(0)    # [1, E']
            loss_A = F.binary_cross_entropy_with_logits(logits, y)

            total_loss += loss_A
            valid += 1

        if valid > 0:
            self.aux_loss = total_loss / float(valid)

    # ------------------------------------------------------------------
    # forward：接口保持原样：
    #   - pre-train: 返回 (reconstruction_masked_tokens, label_masked_tokens)
    #   - forecasting: 返回 hidden_states_full
    #   self.aux_loss 在 pre-train+training+graph 时被写入，用于 loss wrapper。
    # ------------------------------------------------------------------
    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        epoch: int = None,
        **kwargs,
    ) -> torch.Tensor:
        # reshape: [B, L, N, C] → [B, N, C, L]
        history_data = history_data.permute(0, 2, 3, 1)
        self.aux_loss = None

        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index, walk_edges_batch = self.encoding(history_data)

            # decoding
            reconstruction_full, hidden_states_full = self.decoding(
                hidden_states_unmasked,
                unmasked_token_index,
                masked_token_index,
            )

            # reconstruction + label for MAE
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_full,
                history_data,
                unmasked_token_index,
                masked_token_index,
            )

            # 边重构辅助 loss
            if self.spatial:
                self._compute_aux_loss(
                    hidden_states_full,
                    unmasked_token_index,
                    masked_token_index,
                    walk_edges_batch,
                )

            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _, _ = self.encoding(history_data, mask=False)
            return hidden_states_full


def main():
    import sys
    from torchsummary import summary
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '2'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    model = Mask(
        patch_size=12,
        in_channel=1,
        embed_dim=96,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        mask_ratio=0.75,
        encoder_depth=4,
        decoder_depth=1,
        mode="pre-train",
        spatial=True,
    ).to(device)
    summary(model, (288 * 7, 307, 1), device=device)


if __name__ == '__main__':
    main()
