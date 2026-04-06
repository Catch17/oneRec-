"""
oneRec - Phase 5: 推荐演示
输入用户浏览过的商品 → 输出 Top-10 推荐列表
"""

import pickle
import torch
import pandas as pd
from src.model import SASRec

# ============================================================
# 1. 加载所有需要的数据和模型
# ============================================================
print("📂 加载数据和模型...\n")

# 加载预处理数据
with open("../artifacts/data_bundle.pkl", "rb") as f:
    data = pickle.load(f)

item2idx = data["item2idx"]
idx2item = data["idx2item"]
num_items = data["num_items"]
max_seq_len = data["max_seq_len"]

# 加载商品信息表（用于展示商品名称）
df_items = pd.read_csv("../data/items.csv")
item_info = {}
for _, row in df_items.iterrows():
    item_info[row["itemId"]] = {
        "title": row["title"],
        "category": row["category"],
        "tags": row["tags"],
    }

# 加载训练好的模型
device = torch.device("cpu")
model = SASRec(
    num_items=num_items,
    max_seq_len=max_seq_len,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    dropout=0.2,
)
model.load_state_dict(torch.load("../artifacts/best_model.pt", weights_only=True, map_location=device))
model.eval()
print("✅ 模型加载完成!\n")


# ============================================================
# 2. 展示商品目录（方便选择）
# ============================================================
def show_catalog():
    """打印所有商品目录"""
    print("=" * 65)
    print("📋 商品目录")
    print("=" * 65)
    current_cat = ""
    for _, row in df_items.iterrows():
        if row["category"] != current_cat:
            current_cat = row["category"]
            print(f"\n  【{current_cat}】")
        print(f"    {row['itemId']}  {row['title']:<16}  标签: {row['tags']}")
    print("\n" + "=" * 65)


# ============================================================
# 3. 推荐函数
# ============================================================
def recommend(browsed_item_ids, top_k=10):
    """
    给定用户浏览过的商品 ID 列表，返回 Top-K 推荐

    参数:
        browsed_item_ids: 用户浏览过的商品原始 ID 列表，如 [1001, 1003, 1005]
        top_k: 推荐数量
    """
    # 转换为模型内部的 idx
    seq = []
    for iid in browsed_item_ids:
        if iid in item2idx:
            seq.append(item2idx[iid])
        else:
            print(f"  ⚠️ 商品 ID {iid} 不存在，已跳过")

    if len(seq) == 0:
        print("  ❌ 没有有效的商品 ID")
        return

    # Padding：左侧补 0，截断到 max_seq_len
    if len(seq) >= max_seq_len:
        seq = seq[-max_seq_len:]
    else:
        seq = [0] * (max_seq_len - len(seq)) + seq

    # 转为 tensor
    input_tensor = torch.LongTensor([seq])  # (1, max_seq_len)

    # 模型推理
    top_items, top_scores = model.predict(input_tensor, top_k=top_k)
    top_items = top_items[0].tolist()
    top_scores = top_scores[0].tolist()

    # 过滤掉用户已经浏览过的商品
    browsed_idx_set = set(item2idx[iid] for iid in browsed_item_ids if iid in item2idx)
    results = []
    for item_idx, score in zip(top_items, top_scores):
        if item_idx in browsed_idx_set:
            continue
        if item_idx in idx2item:
            results.append((item_idx, score))
        if len(results) >= top_k:
            break

    return results


def display_recommendation(browsed_item_ids, top_k=10):
    """完整的推荐展示"""
    print("\n" + "=" * 65)
    print("🛒 用户浏览历史:")
    print("-" * 65)
    for i, iid in enumerate(browsed_item_ids, 1):
        if iid in item_info:
            info = item_info[iid]
            print(f"  {i}. [{iid}] {info['title']}  ({info['category']} | {info['tags']})")
        else:
            print(f"  {i}. [{iid}] ⚠️ 未知商品")

    print("\n🎯 模型推荐 Top-{0}:".format(top_k))
    print("-" * 65)

    results = recommend(browsed_item_ids, top_k=top_k + 5)  # 多取一些，因为要过滤已浏览的
    if not results:
        print("  无推荐结果")
        return

    for rank, (item_idx, score) in enumerate(results[:top_k], 1):
        original_id = idx2item[item_idx]
        info = item_info[original_id]
        score_bar = "█" * int(max(0, (score + 5)) * 3)  # 简易得分条
        print(f"  {rank:>2}. [{original_id}] {info['title']:<16} "
              f"({info['category']:<4} | {info['tags']:<12}) "
              f"得分: {score:>7.3f}  {score_bar}")

    print("=" * 65)


# ============================================================
# 4. 运行演示场景
# ============================================================
if __name__ == "__main__":
    show_catalog()

    # ---- 场景 1: 牛奶爱好者 ----
    print("\n\n🧪 场景 1: 一个爱喝牛奶的用户")
    display_recommendation([1001, 1002, 1003])  # 蒙牛纯牛奶, 伊利高钙奶, 特仑苏有机奶

    # ---- 场景 2: 零食控 ----
    print("\n\n🧪 场景 2: 一个爱吃零食的用户")
    display_recommendation([1007, 1012, 1014])  # 奥利奥, 乐事原味, 品客烧烤

    # ---- 场景 3: 健康饮食 ----
    print("\n\n🧪 场景 3: 一个偏好健康食品的用户")
    display_recommendation([1003, 1016, 1022])  # 特仑苏有机, 好丽友薯愿(非油炸), 三只松鼠每日坚果

    # ---- 场景 4: 饮料达人 ----
    print("\n\n🧪 场景 4: 一个爱喝饮料的用户")
    display_recommendation([1017, 1018, 1019, 1020])  # 可乐, 元气森林, 茶π, 王老吉

    # ---- 场景 5: 混合浏览 ----
    print("\n\n🧪 场景 5: 随便逛逛的用户")
    display_recommendation([1001, 1012, 1019])  # 蒙牛纯牛奶, 乐事原味, 茶π

    print("\n\n✅ 演示完毕!")
    print("💡 你可以修改上面的商品 ID 列表，测试不同的推荐场景")
