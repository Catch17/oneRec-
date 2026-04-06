"""
oneRec - Phase 1: 生成模拟数据
生成三张表: items.csv, users.csv, interactions.csv
场景: 超市零食饮料推荐
"""

import random
import pandas as pd
from datetime import datetime, timedelta

# ============================================================
# 固定随机种子，保证每次运行结果一致（方便调试）
# ============================================================
random.seed(42)

# ============================================================
# 1. 商品表 (Items)
# ============================================================
# 定义商品目录：每个类别下有若干商品，每个商品带有标签
catalog = {
    "牛奶": [
        {"title": "蒙牛纯牛奶",      "tags": ["纯牛奶", "常温"]},
        {"title": "伊利高钙奶",      "tags": ["高钙", "常温"]},
        {"title": "特仑苏有机奶",    "tags": ["有机", "高端"]},
        {"title": "光明鲜牛奶",      "tags": ["鲜奶", "冷藏"]},
        {"title": "安慕希酸奶",      "tags": ["酸奶", "常温"]},
        {"title": "蒙牛酸酸乳",      "tags": ["酸奶", "乳饮料"]},
    ],
    "饼干": [
        {"title": "奥利奥原味",      "tags": ["巧克力", "夹心"]},
        {"title": "好丽友派",        "tags": ["巧克力", "蛋糕"]},
        {"title": "太平苏打饼干",    "tags": ["咸味", "代餐"]},
        {"title": "百草味曲奇",      "tags": ["黄油", "酥脆"]},
        {"title": "嘉士利薄脆",      "tags": ["咸味", "薄脆"]},
    ],
    "薯片": [
        {"title": "乐事原味薯片",    "tags": ["原味", "经典"]},
        {"title": "乐事黄瓜味",      "tags": ["清爽", "黄瓜"]},
        {"title": "品客烧烤味",      "tags": ["烧烤", "罐装"]},
        {"title": "可比克番茄味",    "tags": ["番茄", "膨化"]},
        {"title": "好丽友薯愿",      "tags": ["非油炸", "健康"]},
    ],
    "饮料": [
        {"title": "可口可乐",        "tags": ["碳酸", "经典"]},
        {"title": "元气森林气泡水",  "tags": ["无糖", "气泡"]},
        {"title": "农夫山泉茶π",    "tags": ["茶饮", "果味"]},
        {"title": "王老吉凉茶",      "tags": ["凉茶", "罐装"]},
        {"title": "东鹏特饮",        "tags": ["功能", "提神"]},
        {"title": "美汁源果粒橙",    "tags": ["果汁", "橙味"]},
    ],
    "坚果": [
        {"title": "三只松鼠每日坚果","tags": ["混合", "健康"]},
        {"title": "百草味夏威夷果",  "tags": ["单品", "奶香"]},
        {"title": "洽洽瓜子",        "tags": ["瓜子", "经典"]},
        {"title": "良品铺子腰果",    "tags": ["单品", "进口"]},
        {"title": "沃隆坚果礼盒",    "tags": ["混合", "送礼"]},
    ],
}

# 将目录展开为商品列表，并分配 itemId
items = []
item_id = 1001
category_to_items = {}  # 记录每个类别包含哪些 itemId，后面造行为数据要用

for category, products in catalog.items():
    category_to_items[category] = []
    for product in products:
        items.append({
            "itemId": item_id,
            "title": product["title"],
            "category": category,
            "tags": "|".join(product["tags"]),  # 用 | 分隔多个标签
        })
        category_to_items[category].append(item_id)
        item_id += 1

df_items = pd.DataFrame(items)
print(f"✅ 商品表: {len(df_items)} 条记录")
print(df_items.head(10))
print()

# ============================================================
# 2. 用户表 (Users)
# ============================================================
NUM_USERS = 200  # 生成 200 个用户

all_categories = list(catalog.keys())
all_tags = list(set(tag for products in catalog.values() for p in products for tag in p["tags"]))

users = []
for uid in range(1, NUM_USERS + 1):
    # 每个用户随机偏好 1~2 个类别
    preferred_cats = random.sample(all_categories, k=random.randint(1, 2))
    # 每个用户随机偏好 2~4 个标签
    preferred_tags = random.sample(all_tags, k=random.randint(2, 4))
    users.append({
        "userId": uid,
        "preferredCategories": "|".join(preferred_cats),
        "preferredTags": "|".join(preferred_tags),
    })

df_users = pd.DataFrame(users)
print(f"✅ 用户表: {len(df_users)} 条记录")
print(df_users.head(10))
print()

# ============================================================
# 3. 行为日志表 (Interactions)
# ============================================================
# 关键：用户的行为要"有规律"，否则模型学不到东西
# 策略：每个用户生成若干条"会话"，每条会话是 3~6 次连续点击
#       其中 70% 的点击落在用户偏好类别内，30% 随机（模拟探索行为）

all_item_ids = [item["itemId"] for item in items]
interactions = []
base_time = datetime(2025, 1, 1, 8, 0, 0)

for user in users:
    uid = user["userId"]
    pref_cats = user["preferredCategories"].split("|")

    # 该用户偏好类别下的所有商品 id
    pref_item_ids = []
    for cat in pref_cats:
        pref_item_ids.extend(category_to_items[cat])

    # 每个用户生成 5~15 条会话
    num_sessions = random.randint(5, 15)
    current_time = base_time + timedelta(hours=random.randint(0, 200))

    for _ in range(num_sessions):
        session_length = random.randint(3, 6)  # 每条会话 3~6 次点击
        for step in range(session_length):
            # 70% 概率点击偏好类别的商品，30% 随机
            if random.random() < 0.7:
                clicked_item = random.choice(pref_item_ids)
            else:
                clicked_item = random.choice(all_item_ids)

            interactions.append({
                "userId": uid,
                "itemId": clicked_item,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "action_type": "click",
            })
            current_time += timedelta(seconds=random.randint(10, 300))

        # 会话间隔 1~48 小时
        current_time += timedelta(hours=random.randint(1, 48))

df_interactions = pd.DataFrame(interactions)
print(f"✅ 行为日志表: {len(df_interactions)} 条记录")
print(df_interactions.head(10))
print()

# ============================================================
# 4. 保存为 CSV
# ============================================================
df_items.to_csv("items.csv", index=False, encoding="utf-8-sig")
df_users.to_csv("users.csv", index=False, encoding="utf-8-sig")
df_interactions.to_csv("interactions.csv", index=False, encoding="utf-8-sig")

print("=" * 50)
print("🎉 数据生成完毕！已保存到项目根目录：")
print("   📄 items.csv")
print("   📄 users.csv")
print("   📄 interactions.csv")
print(f"   商品数: {len(df_items)}")
print(f"   用户数: {len(df_users)}")
print(f"   行为记录数: {len(df_interactions)}")
