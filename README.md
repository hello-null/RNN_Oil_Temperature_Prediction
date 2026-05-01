# RNN 油温预测项目

基于循环神经网络（RNN）的多元时间序列预测项目，使用前10个时间步的数据预测下一时刻的目标油温（OT）。

## 项目概述

本项目实现了一个基于 PyTorch 的 RNN 时间序列预测模型，用于预测 ETTh1 数据集中的目标油温（Oil Temperature, OT）。模型使用过去10个时刻（10行数据）的7维特征（HUFL, HULL, MUFL, MULL, LUFL, LULL, OT）来预测下一时刻的油温值。

## 数据集

项目使用 **ETTh1** 数据集，这是一个电力变压器油温数据集，包含以下7个特征：

| 特征 | 含义(AI生成) |
|------|------|
| HUFL | 高电压频率负荷 |
| HULL | 高电压低频负荷 |
| MUFL | 中电压频率负荷 |
| MULL | 中电压低频负荷 |
| LUFL | 低电压频率负荷 |
| LULL | 低电压低频负荷 |
| **OT** | **目标油温（预测目标）** |

数据集位于 `ETT-small/ETTh1.csv`。
数据集来自: [ETTh1 数据集](https://github.com/zhouhaoyi/ETDataset)

## 项目结构

```
RNN_温度预测/
├── ETT-small/              # 数据集目录
│   └── ETTh1.csv           # ETTh1 油温数据集
├── img/                    # 图片目录
│   ├── 1.png               # 窗口切分示意图
│   └── 2.png               # 测试结果图
├── data_loader.py          # 数据加载器，实现ETTDataset类
├── model.py                # RNN 模型定义
├── train.py                # 训练与测试主脚本
├── last_epoch.pth          # 训练好的模型权重
└── README.md               # 项目说明文档
```

## 文件说明

| 文件 | 功能说明 |
|------|----------|
| `data_loader.py` | 实现 `ETTDataset` 类，负责从 CSV 文件读取数据、切分时间窗口、标准化处理，将时序数据转换为监督学习样本 |
| `model.py` | 定义 `RNNModel` 类，实现多层 RNN 网络结构，接受7维特征输入，输出单步油温预测值 |
| `train.py` | 主训练与测试脚本，包含 `run_train()` 和 `run_test()` 函数，协调数据加载、模型训练、评估和结果可视化 |
| `last_epoch.pth` | 训练完成后保存的模型权重文件，可直接加载用于推理 |

## 安装依赖

确保已安装 Python 3.8+，然后安装所需依赖：

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 快速开始

### 训练模型

1. 打开 `train.py`，取消注释run_train函数行：
```python
run_train(model=model, train_loader=train_loader, num_epochs=200)
```

2. 运行训练脚本：
```bash
python train.py
```

训练过程中会打印每个 epoch 的平均损失，训练完成后模型权重会保存到 `last_epoch.pth`。

### 测试模型

1. 确保已有训练好的模型 `last_epoch.pth`
2. 打开 `train.py`，注释训练代码，取消注释run_test函数行：
```python
run_test(model=model, test_loader=test_loader, scaler_y=train_dataset.scaler_y, plot=True)
```

3. 运行测试脚本：
```bash
python train.py
```

测试完成后会输出原始量纲下的 MSE 误差，并显示预测曲线与真实曲线的对比图。如下。
![测试结果](img/2.png)

## 模型架构

### RNNModel

本项目使用的 RNN 模型结构如下：

```
输入层 → RNN层（多层） → 全连接层 → 输出
```

**模型参数：**
- `input_size`: 7（7维特征输入）
- `hidden_size`: 256（隐藏层维度）
- `num_layers`: 2（RNN堆叠层数）
- `output_size`: 1（单一油温预测值）

**前向传播流程：**
1. 输入张量形状：`(batch_size, seq_len=10, input_size=7)`
2. 通过 RNN 层获取最后一个时间步的隐藏状态
3. 通过全连接层映射到输出维度，得到预测值

## 数据处理流程

```
原始数据 CSV → ETTDataset（窗口切分）→ StandardScaler 标准化 → DataLoader → RNN 模型
                                                                     ↓
                                                         反标准化 → 原始量纲预测值 → 绘图评估
```

**关键处理步骤：**
1. **窗口切分**：使用滑动窗口从时间序列中提取样本，历史窗口大小为10，预测下一时刻的油温值。
示意图如下，使用绿色背景表示历史信息，预测蓝色背景的OT值，绿色区域高度为10，表示过去10个时刻的信息，绿色区域宽度为7，表示7个特征，示意图中滑动窗口每次向下移动1个时间步，实际代码中是step_size=2。
![窗口切分示意图](img/1.png)
2. **标准化**：使用 `StandardScaler` 对特征和标签分别进行标准化，训练集拟合，测试集复用。
正确的流程必须是：
1.先划分训练集/测试集；
2.只在训练集上 fit scaler；
3.用这个scaler 分别 transform 训练集和测试集。
如果用全数据拟合 scaler，测试集的均值/方差包含了未来趋势，相当于给模型“透露”了测试集的整体数值范围。
3. **反标准化**：预测完成后将结果还原到原始量纲进行评估

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 训练集比例 | 70% | 数据按时间顺序切分 |
| 批大小 | 64 | Mini-batch 训练 |
| 学习率 | 0.001 | Adam 优化器 |
| 损失函数 | MSE | 均方误差 |
| 训练轮数 | 200 | 默认值 |
| 历史窗口 | 10 | 用过去10个时刻的数据 |

## 评估指标

模型在测试集上使用 **原始量纲下的均方误差（MSE）** 进行评估：

```python
mse_original = np.mean((pred_original - true_original) ** 2)
```

## 使用示例

### 数据加载示例

```python
from data_loader import ETTDataset
from torch.utils.data import DataLoader

# 创建训练集（开启标准化）
train_dataset = ETTDataset(
    df=train_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
    history_size=10,
    future_size=1,
    step_size=2,
    SCALER=True
)

# 创建测试集（复用训练集的 scaler）
test_dataset = ETTDataset(
    df=test_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
    history_size=10,
    future_size=1,
    step_size=2,
    SCALER=False
)
test_dataset.set_scaler(*train_dataset.get_scaler())

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

### 模型定义示例

```python
from model import RNNModel

# 创建模型
model = RNNModel(input_size=7, hidden_size=256, num_layers=2, output_size=1)

# 加载已训练模型
model = torch.load("./last_epoch.pth")
```

## 注意事项

1. **数据顺序**：特征列必须按照 `[HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]` 的顺序传入
2. **标准化一致性**：测试集必须使用训练集拟合的 scaler，否则会导致预测结果偏差
3. **测试集不打乱**：测试时 `shuffle=False`，保持时间序列的顺序性
4. **GPU 加速**：如果系统支持 CUDA，模型会自动使用 GPU 进行训练和推理

## 技术栈

- **框架**: PyTorch 1.x
- **数据处理**: Pandas, NumPy
- **标准化**: scikit-learn StandardScaler
- **可视化**: Matplotlib

## 许可证

本项目仅供学习和研究使用。

---

*项目结构清晰，代码注释详细，欢迎参考和使用！*
