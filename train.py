# train.py
"""
ETTh1 油温预测 —— 训练与评估主脚本

本脚本完成以下工作：
1. 读取原始 ETTh1.csv 数据，使用 ETTDataset 生成训练/测试样本。
2. 构建 SimpleRNN 时序预测模型（来自 model.py），并对模型进行标准化 MSE 损失下的训练。
3. 在测试集上评估已训练模型的性能，并对预测值和真实值进行反标准化，
   还原到原始油温量纲，最终绘制预测曲线。

典型使用方式：
    # 训练模型
    python train.py          # 取消 run_train 的注释，运行训练
    # 测试模型（需先训练并保存 last_epoch.pth）
    python train.py          # 注释 run_train，取消 run_test 的注释，运行测试

数据流：
    CSV → ETTDataset(标准化) → DataLoader → RNNModel → 预测值(标准化)
    → 反标准化(inverse_transform) → 原始 OT 值 → 绘图
"""

from torch import nn
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

from model import RNNModel
from data_loader import ETTDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(model, train_loader, num_epochs=1000):
    """
    模型训练函数

    参数
    ----------
    model : nn.Module
        尚未训练的 RNNModel 实例。
    train_loader : DataLoader
        训练数据加载器，每次迭代返回 (history, label)，
        history : torch.Tensor, shape (batch, history_size, 7)
        label   : torch.Tensor, shape (batch, 1, 1)   —— 标准化后的 OT 标签
    num_epochs : int, default 1000
        训练轮数。

    过程
    ----------
    - 使用 MSE 损失和 Adam 优化器。
    - 每个 epoch 打印当前平均损失。
    - 训练结束后将模型权重保存到 "./last_epoch.pth"。
    """
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(DEVICE)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (his, cl) in enumerate(train_loader):
            # his: (batch, 10, 7)  历史窗口，包含全部 7 个特征（含 OT）且已标准化
            # cl : (batch, 1, 1)   未来 1 步的 OT 标签，已标准化
            t_his = his.to(DEVICE)                     # (batch, 10, 7)
            t_cl = torch.squeeze(cl, dim=2).to(DEVICE) # (batch, 1)

            outputs = model(t_his)                     # (batch, 1)
            loss = criterion(outputs, t_cl)

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch}/{num_epochs-1}], Avg-Loss: {total_loss/len(train_loader):.4f}')

    torch.save(obj=model, f="./last_epoch.pth")
    print('模型已保存至 ./last_epoch.pth')


def run_test(model, test_loader, scaler_y, plot=True):
    """
    模型测试与评估函数

    参数
    ----------
    model : nn.Module
        已训练的 RNNModel 实例。
    test_loader : DataLoader
        测试数据加载器，shuffle=False，保持时间顺序。
    scaler_y : StandardScaler
        用于标签反标准化的 scaler（来自训练集，测试集复用）。
    plot : bool, default True
        是否绘制预测值与真实值曲线。

    返回
    ----------
    pred_original : np.ndarray, shape (N,)
        反标准化后的预测 OT 值（原始量纲）。
    true_original : np.ndarray, shape (N,)
        反标准化后的真实 OT 值（原始量纲）。
    mse_original : float
        原始量纲下的均方误差。

    说明
    ----------
    - 测试时模型处于 eval 模式，梯度不计算。
    - 所有预测和标签先在标准化空间收集，最后统一反标准化，
      以得到原始油温下的误差和曲线。
    """
    criterion = torch.nn.MSELoss(reduction='mean')
    model.to(DEVICE)
    model.eval()

    all_predictions = []  # 标准化预测值列表
    all_targets = []      # 标准化真实值列表
    total_loss = 0.0

    with torch.no_grad():
        for his, cl in test_loader:
            # his: (batch, 10, 7)  已标准化历史窗口
            # cl : (batch, 1, 1)   已标准化标签
            t_his = his.to(DEVICE)
            t_cl = torch.squeeze(cl, dim=2).to(DEVICE)  # (batch, 1)

            outputs = model(t_his)                      # (batch, 1)
            loss = criterion(outputs, t_cl)
            total_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(t_cl.cpu().numpy())

    # 拼接成完整数组
    pred = np.concatenate(all_predictions, axis=0)  # (N, 1)
    true = np.concatenate(all_targets, axis=0)      # (N, 1)

    # 反标准化：将预测值和真实值恢复到原始油温量纲
    pred_original = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
    true_original = scaler_y.inverse_transform(true.reshape(-1, 1)).flatten()

    # 计算原始尺度的 MSE
    mse_original = np.mean((pred_original - true_original) ** 2)
    print(f'Test MSE (original scale): {mse_original:.4f}')

    # 可选绘图
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(true_original, label='Ground Truth', alpha=0.5)
        plt.plot(pred_original, label='Prediction', alpha=0.5)
        plt.legend()
        plt.title(f'Test Prediction vs True (Original Scale, MSE={mse_original:.4f})')
        plt.xlabel('Sample index (time sequence)')
        plt.ylabel('OT')
        plt.show()

    return pred_original, true_original, mse_original


if __name__ == '__main__':
    # ------------------ 数据准备 ------------------
    df = pd.read_csv("./ETT-small/ETTh1.csv")

    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 训练集：开启标准化 (SCALER=True)，自动拟合 scaler
    train_dataset = ETTDataset(
        df=train_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
        history_size=10,
        future_size=1,
        step_size=2,
        SCALER=True
    )

    # 测试集：不开启标准化，复用训练集的 scaler
    test_dataset = ETTDataset(
        df=test_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
        history_size=10,
        future_size=1,
        step_size=2,
        SCALER=False
    )
    test_dataset.set_scaler(*train_dataset.get_scaler())

    # ------------------ 模型定义 ------------------
    model = RNNModel(input_size=7, hidden_size=256, num_layers=2, output_size=1)
    # 如果已有训练好的模型，取消下一行注释以加载权重
    model = torch.load("./last_epoch.pth")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试集不打乱

    # ------------------ 执行训练或测试 ------------------
    # 训练（需要时取消注释）
    # run_train(model=model, train_loader=train_loader, num_epochs=200)
    # 测试（需要时取消注释）
    run_test(model=model, test_loader=test_loader, scaler_y=train_dataset.scaler_y, plot=True)

