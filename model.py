# model.py
"""
循环神经网络模型定义

本模块定义了一个用于多元时间序列预测的 SimpleRNN 模型。
该模型接受一个历史窗口内的多元时间序列数据（7 维特征），
输出下一个时刻的油温 (OT) 预测值。

模型结构：
    - 一层或多层 RNN（默认 batch_first=True）
    - 取最后一个时间步的隐状态，输入到一个全连接层得到预测值

典型用法：
    model = RNNModel(input_size=7, hidden_size=256, num_layers=2, output_size=1)
    x = torch.randn(32, 10, 7)   # (batch, seq_len=10, input_size=7)
    y = model(x)                 # (batch, output_size=1)
"""

from torch import nn
import torch


class RNNModel(nn.Module):
    """
    SimpleRNN 时间序列预测模型

    参数
    ----------
    input_size : int
        输入特征的维度。当前设定为 7，表示历史窗口内的 7 个原始特征
        (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)。
    hidden_size : int
        RNN 隐状态的特征维度，例如 256。
    num_layers : int
        RNN 的堆叠层数，例如 2。
    output_size : int
        输出维度，当前任务为预测单一油温值，因此固定为 1。

    输入
    ----------
    x : torch.Tensor, shape (batch_size, seq_len, input_size)
        一批历史序列窗口，其中 seq_len 为历史时间步长，input_size 必须等于初始化时的 input_size。

    输出
    ----------
    torch.Tensor, shape (batch_size, output_size)
        对应每个样本的下一个时刻的预测值。
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        # 断言，确保仅用于当前问题设定（7个特征预测1个油温）
        assert input_size == 7 and output_size == 1, 'err'
        self.rnn = nn.RNN(
            input_size,          # 输入层特征数，即每个时间步输入向量的维度
            hidden_size,         # 隐藏层特征数，决定记忆单元的大小
            num_layers,          # 堆叠的 RNN 层数，增加模型容量
            batch_first=True     # 输入/输出张量以 batch 为第一维
        )
        self.fc = nn.Linear(hidden_size, output_size)  # 将最后一个时间步的隐状态映射到输出维度

    def forward(self, x):
        """
        前向传播

        参数
        ----------
        x : torch.Tensor, shape (batch_size, seq_len, input_size)
            输入的时间序列窗口。

        返回
        ----------
        torch.Tensor, shape (batch_size, output_size)
            预测结果，这里是单个油温值。
        """
        # 输入 x 的最后一维（特征维度）必须为 7
        assert x.shape[2] == 7, 'err'
        # 初始化 h0：全零初始隐状态
        # 形状 (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        # RNN 前向计算
        # output: (batch_size, seq_len, hidden_size)
        # h_n   : (num_layers, batch_size, hidden_size) 最后一个时间步的隐状态
        output, h_n = self.rnn(x, h0)

        # 取最后一个时间步的输出，传递给全连接层
        # output[:, -1, :] 形状 (batch_size, hidden_size)
        return self.fc(output[:, -1, :])


if __name__ == '__main__':
    # 测试代码：构造一个随机输入，打印输出形状
    model = RNNModel(input_size=7, hidden_size=256, num_layers=2, output_size=1)

    x = torch.randn(16, 10, 7)  # 16 个样本，每个样本有 10 个历史时间步，每个时间步 7 维特征
    v = model(x)

    print(v.shape)  # 期望输出 torch.Size([16, 1])