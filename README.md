# oneRec-

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)

**oneRec-** 是一个基于 **SASRec (Self-Attentive Sequential Recommendation)** 算法的序列推荐系统 Demo。该项目使用自注意力机制（Self-Attention）来捕捉用户的历史交互序列特征，进而预测用户下一个可能感兴趣的物品。

🎉 **最新更新**：本项目现已全面接入 **FastAPI**，支持通过 **Swagger UI** 进行可视化的 API 交互测试，并新增了模拟对话推荐（Chat）功能！

## 📁 目录结构

```text
oneRec-/
├── artifacts/           # 存放训练输出的模型权重（如 best_model.pt）和数据包（data_bundle.pkl）
├── data/                # 存放原始数据集和处理后的数据
├── src/                 # 核心源代码目录
│   ├── api.py           # 🚀 FastAPI 路由与接口定义（推荐、对话、用户列表）
│   ├── inference.py     # 🧠 模型推理服务封装（RecommenderService）
│   ├── generate_data.py # 样本数据生成/加载脚本
│   ├── preprocess.py    # 数据预处理模块（如序列截断、填充、构建训练/测试集）
│   ├── model.py         # SASRec 核心模型定义（基于 PyTorch 的 Transformer 架构）
│   ├── train.py         # 模型训练和验证循环
│   ├── demo.py          # 基础示例脚本，用于演示模型的前向推理
│   └── main.py          # 项目执行入口点
├── requirements.txt     # Python 依赖包列表
└── README.md            # 项目说明文档
```

## 🛠️ 环境依赖

除了基础的机器学习库，运行交互式 API 还需要安装 FastAPI 相关依赖。请确保您的环境中安装了以下包（建议使用 Python 3.8+）：

```bash
pip install -r requirements.txt
```
*主要依赖：*
- `torch` (建议安装与您机器 CUDA 版本匹配的版本)
- `numpy`, `pandas`
- `fastapi`, `uvicorn`, `pydantic` (用于 API 服务与交互式测试)

## 🚀 快速开始

### 1. 数据准备与训练 (若已有模型可跳过)
```bash
python src/generate_data.py  # 生成/准备数据
python src/preprocess.py     # 数据预处理
python src/train.py          # 训练 SASRec 模型，权重将保存至 artifacts/
```

### 2. 启动交互式 API 服务 (测试推荐与对话功能)
当 `artifacts/` 目录下存在 `data_bundle.pkl` 和 `best_model.pt` 时，您可以启动 API 服务：

```bash
# 使用 uvicorn 启动 src/api.py 中的 app 实例
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

启动成功后，控制台会输出 `RecommenderService loaded`。

### 3. 使用 Swagger UI 进行交互化测试
服务启动后，打开浏览器访问自动生成的交互式文档：
👉 **http://localhost:8000/docs**

在这里，您可以直接在网页上测试以下接口：
*   `GET /users`：快速查看当前可测试的 `user_id` 列表。
*   `POST /recommend`：输入 `user_id` 和 `topk`，一键获取该用户的个性化商品推荐及置信度得分。
*   `POST /chat`：测试最新的对话推荐功能！输入您的 `session_id` 和聊天信息，系统会返回模拟助手的推荐话术。

## 🌐 线上专属访问说明

> **提示**：目前本项目的在线 API 及 Swagger UI 交互式测试环境部署在**私人服务器网址**中，暂未对公众开放。
> 
> 如果您是受邀体验者或团队成员，请直接访问分配给您的私人链接（例如 `http://<您的私人域名或IP>:<端口>/docs`）进行在线交互测试，无需在本地配置环境。

## 🧠 关于 SASRec
**SASRec** 是一种利用 Transformer 结构来处理用户交互序列的推荐模型。它通过利用多头自注意力（Multi-Head Self-Attention）来平衡长短期偏好，并在各种序列推荐基准测试上均表现出了优异的准确性和计算效率。

## 🤝 贡献与反馈
如果你在运行代码中遇到任何问题，欢迎随时提出 [Issue](https://github.com/Catch17/oneRec-/issues) 或提交 Pull Request。
