"""
oneRec - Phase 4: 训练与评估
"""

import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model import SASRec

# ============================================================
# 1. 加载预处理好的数据
# ============================================================
print("📂 加载数据...")
with open("../artifacts/data_bundle.pkl", "rb") as f:
    data = pickle.load(f)

train_seqs = data["train_seqs"]
train_targets = data["train_targets"]
test_seqs = data["test_seqs"]
test_targets = data["test_targets"]
test_uids = data["test_uids"]
num_items = data["num_items"]
max_seq_len = data["max_seq_len"]
idx2item = data["idx2item"]

print(f"   训练样本: {len(train_seqs)} 条")
print(f"   测试样本: {len(test_seqs)} 条")
print(f"   商品数: {num_items}")


# ============================================================
# 2. 构建 PyTorch Dataset 和 DataLoader
# ============================================================
class SeqDataset(Dataset):
    """把 numpy 数组包装成 PyTorch Dataset"""
    def __init__(self, seqs, targets):
        self.seqs = torch.LongTensor(seqs)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


BATCH_SIZE = 128

train_dataset = SeqDataset(train_seqs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = SeqDataset(test_seqs, test_targets)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ DataLoader 就绪 (batch_size={BATCH_SIZE})")


# ============================================================
# 3. 创建模型、损失函数、优化器
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  使用设备: {device}")

model = SASRec(
    num_items=num_items,
    max_seq_len=max_seq_len,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    dropout=0.2,
).to(device)

# 交叉熵损失函数：衡量模型预测的概率分布和真实答案之间的差距
criterion = nn.CrossEntropyLoss()

# Adam 优化器：自动调整学习率，是深度学习最常用的优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"✅ 模型已创建，参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 4. 评估函数
# ============================================================
def evaluate(model, data_loader, device, top_k_list=[5, 10]):
    """
    计算推荐系统的标准指标:
    - HR@K (Hit Rate): 前 K 个推荐中是否包含真实答案
    - NDCG@K: 不仅看是否命中，还看排在第几位
    """
    model.eval()
    all_hits = {k: 0 for k in top_k_list}
    all_ndcgs = {k: 0.0 for k in top_k_list}
    total = 0

    with torch.no_grad():
        for seqs, targets in data_loader:
            seqs = seqs.to(device)
            targets = targets.to(device)

            logits = model(seqs)  # (batch, num_items+1)
            # 把 padding 位置的得分设为负无穷
            logits[:, 0] = float('-inf')

            for k in top_k_list:
                # 取 Top-K 的商品 idx
                _, top_k_items = torch.topk(logits, k, dim=-1)  # (batch, k)

                for i in range(len(targets)):
                    target = targets[i].item()
                    recommended = top_k_items[i].tolist()

                    if target in recommended:
                        all_hits[k] += 1
                        # NDCG: 排名越靠前得分越高
                        rank = recommended.index(target) + 1
                        all_ndcgs[k] += 1.0 / np.log2(rank + 1)

            total += len(targets)

    metrics = {}
    for k in top_k_list:
        metrics[f"HR@{k}"] = all_hits[k] / total
        metrics[f"NDCG@{k}"] = all_ndcgs[k] / total

    return metrics


# ============================================================
# 5. 训练循环
# ============================================================
NUM_EPOCHS = 50  # 训练轮数
EVAL_EVERY = 5   # 每 5 轮评估一次

print(f"\n{'=' * 60}")
print(f"🚀 开始训练! 共 {NUM_EPOCHS} 轮")
print(f"{'=' * 60}")

best_hr10 = 0.0
best_epoch = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for seqs, targets in train_loader:
        seqs = seqs.to(device)
        targets = targets.to(device)

        # 前向传播
        logits = model(seqs)          # (batch, num_items+1)
        loss = criterion(logits, targets)  # 计算损失

        # 反向传播 + 更新参数
        optimizer.zero_grad()         # 清空上一轮的梯度
        loss.backward()               # 计算梯度
        optimizer.step()              # 更新参数

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    elapsed = time.time() - start_time

    # 每轮都打印损失
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", end="")

    # 每 EVAL_EVERY 轮评估一次
    if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
        metrics = evaluate(model, test_loader, device)
        hr5 = metrics["HR@5"]
        hr10 = metrics["HR@10"]
        ndcg5 = metrics["NDCG@5"]
        ndcg10 = metrics["NDCG@10"]
        print(f" | HR@5: {hr5:.4f} | HR@10: {hr10:.4f} | NDCG@5: {ndcg5:.4f} | NDCG@10: {ndcg10:.4f}", end="")

        # 记录最佳模型
        if hr10 > best_hr10:
            best_hr10 = hr10
            best_epoch = epoch
            torch.save(model.state_dict(), "../artifacts/best_model.pt")
            print(f" ⭐ 新最佳!", end="")

    print()  # 换行

print(f"\n{'=' * 60}")
print(f"🏆 训练完成!")
print(f"   最佳 HR@10: {best_hr10:.4f} (第 {best_epoch} 轮)")
print(f"   模型已保存到: best_model.pt")
print(f"{'=' * 60}")


# ============================================================
# 6. 加载最佳模型，做最终评估
# ============================================================
print(f"\n📊 加载最佳模型进行最终评估...")
model.load_state_dict(torch.load("../artifacts/best_model.pt", weights_only=True))
final_metrics = evaluate(model, test_loader, device, top_k_list=[1, 3, 5, 10])

print(f"\n{'=' * 60}")
print(f"📋 最终测试集指标:")
print(f"{'=' * 60}")
print(f"   {'指标':<12} {'得分':>8}")
print(f"   {'-' * 22}")
for metric_name, score in final_metrics.items():
    bar = "█" * int(score * 40)  # 简单的进度条可视化
    print(f"   {metric_name:<12} {score:>8.4f}  {bar}")

print(f"\n💡 指标解读:")
print(f"   HR@K  = 前 K 个推荐中命中真实答案的比例 (越高越好)")
print(f"   NDCG@K = 命中时排名越靠前得分越高 (越高越好)")
print(f"   随机猜测的 HR@10 ≈ {10/num_items:.4f}，模型应该远高于此")
