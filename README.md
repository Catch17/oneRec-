# oneRec-

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**oneRec-** 是一个基于 **SASRec (Self-Attentive Sequential Recommendation)** 算法的序列推荐系统 Demo。该项目旨在使用自注意力机制（Self-Attention）来捕捉用户的历史交互序列特征，进而预测用户下一个可能感兴趣的物品。

## 📁 目录结构

```text
oneRec-/
├── artifacts/           # 用于存放训练输出的模型权重（如 .pth 文件）和评估日志
├── data/                # 存放原始数据集和处理后的数据
├── src/                 # 核心源代码目录
│   ├── generate_data.py # 样本数据生成/加载脚本
│   ├── preprocess.py    # 数据预处理模块（如序列截断、填充、构建训练/测试集）
│   ├── model.py         # SASRec 核心模型定义（基于 PyTorch 的 Transformer 架构）
│   ├── train.py         # 模型训练和验证循环
│   ├── demo.py          # 示例脚本，用于演示模型的前向推理和推荐过程
│   └── main.py          # 项目执行入口点
├── requirements.txt     # Python 依赖包列表
└── README.md            # 项目说明文档
```

## 🛠️ 环境依赖

要运行本项目，请确保您的环境中安装了以下基础依赖。建议使用 Python 3.8+ 版本：

```bash
pip install -r requirements.txt
```
*主要依赖：*
- `numpy`
- `pandas`
- `torch` (建议安装与您机器 CUDA 版本匹配的 PyTorch 版本)

## 🚀 快速开始

### 1. 准备数据
您可以运行 `generate_data.py` 生成模拟数据集，或者在 `data/` 目录中放入您的自定义用户行为数据：
```bash
python src/generate_data.py
```

### 2. 数据预处理
对生成的数据进行处理，转化为模型所需的定长序列：
```bash
python src/preprocess.py
```

### 3. 模型训练
运行训练脚本以训练 SASRec 模型。训练好的模型权重和日志将被保存在 `artifacts/` 文件夹下：
```bash
python src/train.py
```

### 4. 推荐演示
使用训练好的模型预测指定用户的下一个兴趣物品：
```bash
python src/demo.py
```

## 🧠 关于 SASRec
**SASRec** 是一种利用 Transformer 结构来处理用户交互序列的推荐模型。它通过利用多头自注意力（Multi-Head Self-Attention）来平衡长短期偏好，并在各种序列推荐基准测试上均表现出了优异的准确性和计算效率。

## 🤝 贡献与反馈
如果你在运行代码中遇到任何问题，欢迎随时提出 [Issue](https://github.com/Catch17/oneRec-/issues) 或提交 Pull Request。
