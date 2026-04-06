"""
oneRec - Phase 3: SASRec 简化版模型 (v3 彻底修复版)
基于 Self-Attention 的序列推荐模型
"""

import torch
import torch.nn as nn
import math


class SASRec(nn.Module):
    """
    简化版 SASRec (Self-Attentive Sequential Recommendation)
    """

    def __init__(self, num_items, max_seq_len, embed_dim=64, num_heads=2, num_layers=2, dropout=0.2):
        super(SASRec, self).__init__()

        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        # ---- Embedding 层 ----
        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # ---- 手动构建 Transformer 组件 ----
        # 不使用 nn.TransformerEncoder，改为手动实现，完全掌控掩码逻辑
        self.attention_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            ))
            self.norm1_layers.append(nn.LayerNorm(embed_dim))
            self.norm2_layers.append(nn.LayerNorm(embed_dim))

        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def _build_attention_mask(self, input_seq):
        """
        构建合并的注意力掩码 (因果 + padding)

        关键思路：
        - 因果掩码：位置 i 只能关注位置 <= i
        - padding 掩码：任何位置都不应该关注 padding 位置
        - 但 padding 位置本身可以关注自己（避免全 -inf 导致 nan）

        返回: (seq_len, seq_len) 的 float 掩码，-inf 表示屏蔽，0.0 表示允许
        """
        batch_size, seq_len = input_seq.shape
        device = input_seq.device

        # 因果掩码: 上三角为 True (要屏蔽的位置)
        causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # Padding 掩码: 哪些位置是 padding
        is_pad = (input_seq == 0)  # (batch, seq_len)

        # 我们需要一个 (batch, seq_len, seq_len) 的掩码
        # mask[b, i, j] = True 表示位置 i 不应该关注位置 j
        # 条件: j > i (因果) 或 j 是 padding
        pad_mask = is_pad.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, seq_len)
        causal_mask = causal.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, seq_len)

        combined = causal_mask | pad_mask  # (batch, seq_len, seq_len)

        # 关键修复：如果某一行全是 True（即该位置没有任何可关注的位置），
        # 让它至少能关注自己，避免 softmax 全 -inf → nan
        all_masked = combined.all(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # 对角线位置 = 自己关注自己
        self_mask = torch.eye(seq_len, device=device).bool().unsqueeze(0).expand(batch_size, -1, -1)
        # 如果某行全被 mask，就把对角线打开
        combined = combined & ~(all_masked & self_mask)

        # 转为 float: True → -inf, False → 0.0
        # MultiheadAttention 需要 (batch * num_heads, seq_len, seq_len)
        float_mask = torch.zeros_like(combined, dtype=torch.float)
        float_mask.masked_fill_(combined, float('-inf'))

        # 扩展到多头: (batch, num_heads, seq_len, seq_len)
        float_mask = float_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # reshape 为 (batch * num_heads, seq_len, seq_len)
        float_mask = float_mask.reshape(batch_size * self.num_heads, seq_len, seq_len)

        return float_mask

    def forward(self, input_seq):
        """
        前向传播
        """
        batch_size, seq_len = input_seq.shape
        device = input_seq.device

        # Step 1: Embedding
        item_emb = self.item_embedding(input_seq)  # (batch, seq, dim)

        # Step 2: 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        hidden = item_emb + pos_emb
        hidden = self.input_dropout(self.input_norm(hidden))

        # Step 3: 构建注意力掩码
        attn_mask = self._build_attention_mask(input_seq)  # (batch*heads, seq, seq)

        # Step 4: 逐层通过自注意力 + FFN
        for i in range(len(self.attention_layers)):
            # Self-Attention (Pre-Norm)
            residual = hidden
            normed = self.norm1_layers[i](hidden)
            attn_output, _ = self.attention_layers[i](
                query=normed,
                key=normed,
                value=normed,
                attn_mask=attn_mask,
                need_weights=False,
            )
            hidden = residual + attn_output

            # FFN (Pre-Norm)
            residual = hidden
            normed = self.norm2_layers[i](hidden)
            ffn_output = self.ffn_layers[i](normed)
            hidden = residual + ffn_output

        # Step 5: 取最后一个有效位置的输出
        non_pad_mask = (input_seq != 0)  # (batch, seq)
        # 找每个样本最后一个非 padding 位置
        last_indices = []
        for i in range(batch_size):
            positions_i = torch.where(non_pad_mask[i])[0]
            if len(positions_i) > 0:
                last_indices.append(positions_i[-1].item())
            else:
                last_indices.append(seq_len - 1)  # fallback
        last_indices = torch.tensor(last_indices, device=device)

        seq_output = hidden[torch.arange(batch_size, device=device), last_indices]  # (batch, dim)

        # Step 6: 计算得分
        logits = torch.matmul(seq_output, self.item_embedding.weight.T)

        return logits

    def predict(self, input_seq, top_k=10):
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_seq)
            logits[:, 0] = float('-inf')
            top_k_scores, top_k_items = torch.topk(logits, top_k, dim=-1)
        return top_k_items, top_k_scores


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    print("🧪 模型结构测试 (v3)...")

    NUM_ITEMS = 27
    MAX_SEQ_LEN = 10
    BATCH_SIZE = 4

    model = SASRec(
        num_items=NUM_ITEMS,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 模型总参数量: {total_params:,}")

    # 模拟短序列（大量 padding）
    fake_input = torch.zeros(BATCH_SIZE, MAX_SEQ_LEN, dtype=torch.long)
    fake_input[0, -3:] = torch.tensor([3, 15, 7])
    fake_input[1, -2:] = torch.tensor([10, 20])
    fake_input[2, -5:] = torch.tensor([1, 5, 9, 12, 27])
    fake_input[3, -1:] = torch.tensor([8])

    print(f"\n📥 模拟输入 (含大量 padding):")
    for i in range(BATCH_SIZE):
        print(f"   用户{i+1}: {fake_input[i].tolist()}")

    logits = model(fake_input)
    print(f"\n📤 输出 logits 形状: {logits.shape}")

    has_nan = torch.isnan(logits).any().item()
    print(f"   包含 nan: {has_nan}  {'❌ 仍有问题' if has_nan else '✅ 无 nan!'}")

    # 检查得分是否不全为 0
    all_zero = (logits == 0).all().item()
    print(f"   全为零:   {all_zero}  {'❌ 得分异常' if all_zero else '✅ 得分正常!'}")

    top_items, top_scores = model.predict(fake_input, top_k=5)
    print(f"\n🎯 Top-5 推荐:")
    for i in range(BATCH_SIZE):
        print(f"   用户{i+1}: {top_items[i].tolist()}  得分: {[f'{s:.4f}' for s in top_scores[i].tolist()]}")

    print(f"\n✅ 模型测试通过！")
