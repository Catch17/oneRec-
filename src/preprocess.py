"""
oneRec - Phase 2: 数据预处理
将原始 CSV 转化为模型可用的序列数据
"""

import pandas as pd
import numpy as np
import pickle

# ============================================================
# 1. 读取数据
# ============================================================
df_items = pd.read_csv("../data/items.csv")
df_users = pd.read_csv("../data/users.csv")
df_interactions = pd.read_csv("../data/interactions.csv")

print(f"📊 读取完毕: {len(df_items)} 商品, {len(df_users)} 用户, {len(df_interactions)} 条行为")

# ============================================================
# 2. 构建 ID 映射
# ============================================================
# 模型需要连续的整数 ID（从 1 开始，0 留给 padding）
# 原始 itemId 是 1001~1027，我们映射成 1~27

all_item_ids = sorted(df_items["itemId"].unique())
item2idx = {item_id: idx + 1 for idx, item_id in enumerate(all_item_ids)}  # 1~27
idx2item = {idx: item_id for item_id, idx in item2idx.items()}              # 反向映射

num_items = len(all_item_ids)  # 27
print(f"✅ 商品 ID 映射完成: {num_items} 个商品, idx 范围 1~{num_items}, 0 保留给 padding")

# 同时构建 category 和 tag 的映射（后面 Phase 3 模型要用）
all_categories = sorted(df_items["category"].unique())
cat2idx = {cat: idx + 1 for idx, cat in enumerate(all_categories)}  # 1~5, 0=padding

all_tags = set()
for tags_str in df_items["tags"]:
    for tag in tags_str.split("|"):
        all_tags.add(tag)
all_tags = sorted(all_tags)
tag2idx = {tag: idx + 1 for idx, tag in enumerate(all_tags)}  # 1~N, 0=padding

print(f"✅ 类别映射: {len(cat2idx)} 个类别")
print(f"✅ 标签映射: {len(tag2idx)} 个标签")

# ============================================================
# 3. 构建商品特征表（每个商品的类别idx + 标签idx列表）
# ============================================================
MAX_TAGS = 2  # 每个商品最多 2 个标签

item_features = {}
for _, row in df_items.iterrows():
    idx = item2idx[row["itemId"]]
    cat_idx = cat2idx[row["category"]]
    tag_indices = [tag2idx[t] for t in row["tags"].split("|")]
    # padding 标签到固定长度
    while len(tag_indices) < MAX_TAGS:
        tag_indices.append(0)
    tag_indices = tag_indices[:MAX_TAGS]
    item_features[idx] = {
        "cat_idx": cat_idx,
        "tag_indices": tag_indices,
    }

print(f"✅ 商品特征表构建完成")

# ============================================================
# 4. 构建用户行为序列
# ============================================================
# 按用户分组，按时间排序，提取 itemId 序列

df_interactions = df_interactions.sort_values(["userId", "timestamp"])

user_sequences = {}
for uid, group in df_interactions.groupby("userId"):
    # 将原始 itemId 转为连续 idx
    seq = [item2idx[iid] for iid in group["itemId"].values]
    user_sequences[uid] = seq

print(f"✅ 用户序列构建完成: {len(user_sequences)} 个用户")

# 看看序列长度的分布
seq_lengths = [len(s) for s in user_sequences.values()]
print(f"   序列长度 - 最短: {min(seq_lengths)}, 最长: {max(seq_lengths)}, "
      f"平均: {np.mean(seq_lengths):.1f}")

# ============================================================
# 5. 构造训练样本和测试样本
# ============================================================
# 策略：
#   - 每个用户的序列，最后一个点击作为【测试目标】
#   - 倒数第二个点击作为【验证目标】（可选，先不用）
#   - 前面的所有点击用于构造【训练样本】
#
# 训练样本的构造方式（滑动窗口）：
#   序列 [A, B, C, D, E, F] →
#     输入 [A, B]       → 目标 C
#     输入 [A, B, C]    → 目标 D
#     输入 [A, B, C, D] → 目标 E
#   最后的 F 留给测试

MAX_SEQ_LEN = 10  # 输入序列的最大长度

train_data = []  # 每条: (user_id, padded_input_seq, target_item_idx)
test_data = []   # 每条: (user_id, padded_input_seq, target_item_idx)

def pad_sequence(seq, max_len):
    """左侧补 0，截断到 max_len"""
    if len(seq) >= max_len:
        return seq[-max_len:]  # 取最近的 max_len 个
    else:
        return [0] * (max_len - len(seq)) + seq  # 左侧补 0

for uid, seq in user_sequences.items():
    if len(seq) < 3:
        # 序列太短，跳过（至少需要 2 个输入 + 1 个目标）
        continue

    # 测试集：用倒数第二个之前的所有作为输入，最后一个作为目标
    test_input = seq[:-1]  # 除了最后一个
    test_target = seq[-1]
    test_data.append((uid, pad_sequence(test_input, MAX_SEQ_LEN), test_target))

    # 训练集：在去掉最后一个的序列上做滑动窗口
    train_seq = seq[:-1]  # 去掉测试目标
    for i in range(2, len(train_seq)):  # 至少 2 个输入
        input_seq = train_seq[:i]
        target = train_seq[i]
        train_data.append((uid, pad_sequence(input_seq, MAX_SEQ_LEN), target))

print(f"\n📦 数据集构造完成:")
print(f"   训练样本: {len(train_data)} 条")
print(f"   测试样本: {len(test_data)} 条")

# ============================================================
# 6. 转为 NumPy 数组并保存
# ============================================================
train_uids = np.array([d[0] for d in train_data], dtype=np.int64)
train_seqs = np.array([d[1] for d in train_data], dtype=np.int64)
train_targets = np.array([d[2] for d in train_data], dtype=np.int64)

test_uids = np.array([d[0] for d in test_data], dtype=np.int64)
test_seqs = np.array([d[1] for d in test_data], dtype=np.int64)
test_targets = np.array([d[2] for d in test_data], dtype=np.int64)

# 打印一条样本看看长什么样
print(f"\n🔍 训练样本示例（第 1 条）:")
print(f"   用户 ID:    {train_uids[0]}")
print(f"   输入序列:   {train_seqs[0]}  (0 是 padding)")
print(f"   预测目标:   {train_targets[0]} → 原始商品ID: {idx2item[train_targets[0]]}")

# 保存所有预处理结果
data_bundle = {
    "train_uids": train_uids,
    "train_seqs": train_seqs,
    "train_targets": train_targets,
    "test_uids": test_uids,
    "test_seqs": test_seqs,
    "test_targets": test_targets,
    "item2idx": item2idx,
    "idx2item": idx2item,
    "cat2idx": cat2idx,
    "tag2idx": tag2idx,
    "item_features": item_features,
    "num_items": num_items,
    "max_seq_len": MAX_SEQ_LEN,
    "num_categories": len(cat2idx),
    "num_tags": len(tag2idx),
}

with open("../artifacts/data_bundle.pkl", "wb") as f:
    pickle.dump(data_bundle, f)

print(f"\n{'=' * 50}")
print(f"🎉 预处理完毕！已保存到: data_bundle.pkl")
print(f"   商品数 (含 padding): {num_items + 1}")
print(f"   类别数 (含 padding): {len(cat2idx) + 1}")
print(f"   标签数 (含 padding): {len(tag2idx) + 1}")
print(f"   最大序列长度: {MAX_SEQ_LEN}")
