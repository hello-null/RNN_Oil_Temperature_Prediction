# data_loader.py
"""
ETTh1 数据集加载器

本模块实现了一个用于 ETTh1 时间序列预测任务的 PyTorch Dataset，
负责完成以下工作：
    1. 读取原始 CSV 文件（包含 7 列特征：HUFL, HULL, MUFL, MULL, LUFL, LULL, OT）。
    2. 按固定窗口大小、步长和预测长度，从连续时间序列中切分出输入‑标签样本。
    3. 可选地对输入和标签分别进行 StandardScaler 标准化（通常训练集做 fit，测试集复用其统计量）。
    4. 将处理后的数据封装为 DataLoader 可直接使用的 Tensor 格式。

使用示例（在 __main__ 中也有演示）：
    >>> import pandas as pd
    >>> df = pd.read_csv("./ETT-small/ETTh1.csv")
    >>> train_dataset = ETTDataset(df.iloc[:1000], history_size=10, future_size=1, step_size=2, SCALER=True)
    >>> test_dataset = ETTDataset(df.iloc[1000:], history_size=10, future_size=1, step_size=2, SCALER=False)
    >>> test_dataset.set_scaler(*train_dataset.get_scaler())  # 使用训练集的 scaler
    >>> loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    >>> for hist, label in loader:
    ...     print(hist.shape)   # (batch, 10, 7)
    ...     print(label.shape)  # (batch, 1, 1)
    ...     break
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ETTDataset(Dataset):
    """
    ETTh1 时序数据集类

    参数
    ----------
    df : pd.DataFrame
        包含完整 7 列特征的 DataFrame，特征顺序为 [HUFL, HULL, MUFL, MULL, LUFL, LULL, OT]。
    history_size : int, default 10
        历史窗口长度（时间步数），即用过去多少个时刻的数据作为输入。
    future_size : int, default 1
        未来预测长度（目前仅支持 1，预测下一个时刻的 OT）。
    step_size : int, default 2
        相邻样本间的滑动步长（在原序列上每次向后移动多少步）。
    SCALER : bool, default False
        是否对特征和标签进行标准化（StandardScaler）。通常在训练集置为 True，
        测试集置为 False，并通过 set_scaler 方法复用训练集的 scaler。

    属性
    ----------
    history_records : np.ndarray, shape (N, history_size, 7)
        所有输入样本的数组，N 为样本总数。最后一维 7 为全部 7 个原始特征。
    future_label : np.ndarray, shape (N, future_size, 1)
        所有标签数组，最后一维 1 为油温 OT。
    scaler_x : StandardScaler or None
        对历史 7 维特征进行标准化的 scaler（在 process 中 fit，若 SCALER=True）。
    scaler_y : StandardScaler or None
        对标签 OT 进行标准化的 scaler。
    """

    def __init__(self, df, history_size=10, future_size=1, step_size=2, SCALER=False):
        self.df = df
        self.history_size = history_size
        self.future_size = future_size
        self.step_size = step_size

        self.np_data = None          # 转换为 numpy 的原始数据
        self.SCALER = SCALER
        self.scaler_x = None
        self.scaler_y = None
        self.history_records = []    # 历史窗口列表，后续转为 np.array
        self.future_label = []       # 未来标签列表

        self.process()

    def process(self):
        """
        核心处理函数：将 DataFrame 转换为监督学习样本。
        - 遍历整个序列，按 history_size 提取历史窗口（包含所有 7 个特征）。
        - 对应每个窗口，取出 future_size 个时刻的 OT（最后一列）作为标签。
        - 步长 step_size 控制窗口移动的间隔。
        - 若 SCALER=True，则在所有样本上拟合 StandardScaler（x 和 y 分别拟合）。
        """
        self.np_data = self.df.values
        assert self.np_data.shape[1] == 7, '输入 DataFrame 必须恰好有 7 列，按顺序为 6 个特征 + OT'

        for i in range(self.history_size, self.np_data.shape[0] - self.future_size, self.step_size):
            # 历史序列：取前 history_size 行，全部 7 列   -> shape (history_size, 7)
            hist = self.np_data[i - self.history_size:i, :7]
            self.history_records.append(hist)

            # 未来标签：取当前时刻起 future_size 行，只取最后一列 OT -> shape (future_size, 1)
            la = self.np_data[i:i + self.future_size, 6:]
            self.future_label.append(la)

        # 转换为 numpy 数组以便后续处理
        self.history_records = np.array(self.history_records)   # (N, history_size, 7)
        self.future_label = np.array(self.future_label)         # (N, future_size, 1)

        # 标准化：仅当 SCALER=True 时在训练集上拟合
        if self.SCALER:
            # 对历史特征（7 维）进行标准化，reshape 为 (N*history_size, 7) 拟合
            self.scaler_x = StandardScaler()
            self.scaler_x.fit(self.history_records.reshape(-1, 7))

            # 对标签（1 维）进行标准化，reshape 为 (N*future_size, 1) 拟合
            self.scaler_y = StandardScaler()
            self.scaler_y.fit(self.future_label.reshape(-1, 1))

    def get_scaler(self):
        """获取特征和标签对应的 StandardScaler，如果尚未拟合则抛出异常。"""
        assert self.scaler_x is not None and self.scaler_y is not None, 'scaler 未初始化，请先设置 SCALER=True 或在训练集上 fit'
        return self.scaler_x, self.scaler_y

    def set_scaler(self, scax, scay):
        """
        设置特征和标签的 StandardScaler（通常在测试集上调用，复用训练集的统计量）。
        参数
        ----------
        scax : StandardScaler
            已经 fit 好的特征 scaler。
        scay : StandardScaler
            已经 fit 好的标签 scaler。
        """
        self.scaler_x = scax
        self.scaler_y = scay

    def __len__(self):
        return self.history_records.shape[0]

    def __getitem__(self, idx):
        """
        返回第 idx 个样本的标准化后张量。

        返回
        ----------
        torch.Tensor, shape (history_size, 7)
            标准化后的历史窗口数据，最后一维 7 包含全部原始特征。
        torch.Tensor, shape (future_size, 1)
            标准化后的未来标签（OT），future_size 目前固定为 1。
        """
        assert self.scaler_x is not None and self.scaler_y is not None, 'scaler 未设置，无法标准化'

        # 历史序列：从 numpy 取，用 scaler_x 标准化
        a1 = self.history_records[idx]          # (history_size, 7)
        a1_tr = self.scaler_x.transform(a1)

        # 标签：用 scaler_y 标准化
        c1 = self.future_label[idx]             # (future_size, 1)
        c1_tr = self.scaler_y.transform(c1)

        return torch.tensor(a1_tr, dtype=torch.float32), \
               torch.tensor(c1_tr, dtype=torch.float32)


if __name__ == "__main__":
    # 演示如何创建数据集并查看数据形状
    df = pd.read_csv("./ETT-small/ETTh1.csv")

    train_ratio = 0.7
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 训练集：开启标准化，自动 fit scaler
    train_dataset = ETTDataset(
        df=train_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
        history_size=10,
        future_size=1,
        step_size=2,
        SCALER=True
    )

    # 测试集：不开启标准化，之后复用训练集的 scaler
    test_dataset = ETTDataset(
        df=test_df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']],
        history_size=10,
        future_size=1,
        step_size=2,
        SCALER=False
    )
    test_dataset.set_scaler(*train_dataset.get_scaler())

    # 将测试集[0]条数据逆变换为初始值
    print(train_dataset.get_scaler()[0].inverse_transform(test_dataset[0][0].numpy()))

    # 输出训练集和测试集样本的 shape
    print("训练集样本 shape：")
    print("  历史窗口：", train_dataset[0][0].shape)   # torch.Size([10, 7])
    print("  未来标签：", train_dataset[0][1].shape)   # torch.Size([1, 1])

    print("测试集样本 shape：")
    print("  历史窗口：", test_dataset[0][0].shape)    # torch.Size([10, 7])
    print("  未来标签：", test_dataset[0][1].shape)    # torch.Size([1, 1])

