import torch
import torch.nn as nn
import torch.optim as optim


class FFLine(nn.Linear):
    """
    Forward-Forward 算法的单层实现
    继承自 nn.Linear，但增加自有的训练逻辑
    """

    def __init__(self, in_features, out_features, bias=True, threshold=10.0):
        super().__init__(in_features, out_features, bias)
        self.threshold = threshold  # 定义 Goodness 的阈值 theta
        self.opt = optim.Adam(self.parameters(), lr=0.03)  # 每一层都有自己的优化器
        self.ln = nn.LayerNorm(out_features)  # Hinton建议使用LayerNorm防止能量爆炸

    def forward(self, x):
        """
        前向传播计算：输入 x，输出经过 ReLU 激活后的神经元状态
        """
        # 计算线性加权后，进行标准化和激活
        # 注意：Hinton 在论文中强调，为了防止下一层直接利用上一层的 Goodness (即模长)，每一层传给下一层之前需要对向量进行长度归一化
        # $$x_{direction} = \frac{x}{\|x\|_2 + \epsilon}$$   归一化数学表达式
        # params(x: 当前层的输入张量（Tensor）。通常它的形状是 (batch_size, feature_dim)，即每一行代表一个样本的特征向量。
        # .norm(2, ...):计算 $L_2$ 范数（也叫欧几里得长度）。公式为 $\sqrt{\sum x_i^2}$。它代表了这个向量在多维空间里的“总长度”或“总能量”。
        # dim=1: 指定计算的方向。因为 x 的第 0 维是 Batch，第 1 维是特征，所以 dim=1 表示对每个样本分别计算它自己的模长。
        # keepdim=True: 保持维度不变。计算出的模长原本是一个一维数组，加上这个参数后，形状会从 (batch_size) 变成 (batch_size, 1)。这样做的目的是为了后续能利用 PyTorch 的广播机制（Broadcasting），让原矩阵 x 的每一行都能除以它对应的模长。
        # 1e-4: 这是一个非常小的常数（Epsilon）。它的作用是防止除以零（数值稳定性）。如果某个样本的所有特征全是 0，其模长就是 0，直接除以 0 会导致程序崩溃或产生 NaN。)
        x_direction = x / (x.norm(2, dim=1, keepdim=True) + 1e-4)

        # 数学计算表达式：$$y = \text{ReLU}(X \cdot W^T + b)$$
        #
        # 第一步：torch.mm(x_direction, self.weight.T) —— 矩阵乘法
        #   x_direction: 这是你上一条信息中提到的归一化后的输入。它的形状通常是 (batch_size, in_features)。由于它已经被归一化，它的每个向量现在只代表方向。
        #   self.weight: 这是该层的权重矩阵，形状通常是 (out_features, in_features)。它代表了这一层神经元想要寻找的模式或特征。
        #   .T (Transpose): 转置操作。self.weight 原本是 (输出维度, 输入维度)，转置后变成 (输入维度, 输出维度)。
        #   torch.mm: 执行矩阵乘法（Matrix Multiplication）。
        #   物理意义：这一步本质上是在计算输入向量与权重向量之间的点积（Dot Product）。点积越大，说明当前的输入 x_direction 与该神经元的权重 weight 越匹配（方向越一致）。
        #
        # 第二步：+ self.bias —— 加偏置
        #   self.bias: 这是一个一维向量，形状为 (out_features)。
        #   偏置项相当于给每个神经元设定了一个阈值或基础偏移。它让模型能够更灵活地拟合数据，即使输入全为 0，输出也不一定为 0。
        #
        # 第三步：torch.relu(...) —— 激活函数
        #   ReLU (Rectified Linear Unit):数学表达式为 $f(z) = \max(0, z)$。
        #   物理意义：这是仿生神经元的阈值激发。
        #   如果计算结果（$Wx+b$）大于 0，则原样输出。如果计算结果小于 0，则直接变成 0（神经元不激活/不放电）。
        #   这为网络引入了非线性，使其有能力处理复杂的模式，而不仅仅是做简单的线性投影。

        return torch.relu(torch.mm(x_direction, self.weight.T) + self.bias)

    def train_layer(self, x_pos, x_neg):
        """
        核心训练函数：利用 Goodness 函数进行局部更新
        x_pos: 正样本 (Goodness 应该 > threshold)
        x_neg: 负样本 (Goodness 应该 < threshold)
        """
        # 1. 计算正样本的 Goodness (神经元输出的平方和)
        g_pos = self.forward(x_pos).pow(2).mean(1)

        # 2. 计算负样本的 Goodness (神经元输出的平方和)
        g_neg = self.forward(x_neg).pow(2).mean(1)

        # 3. 构造损失函数：
        # 我们希望 g_pos 越大越好（远离threshold），g_neg 越小越好
        # 使用 Softplus 函数来平滑这个逻辑
        # loss = log(1 + exp(threshold - g_pos)) + log(1 + exp(g_neg - threshold))
        #
        # g_pos (Goodness Positive):正样本（真实数据，如正确标签的图片）经过该层后输出的能量。通常是该层激活值 $y$ 的平方和 $\sum y_i^2$。
        # g_neg (Goodness Negative):负样本（伪造数据，如错误标签的图片）经过该层后输出的“能量”。
        # self.threshold ($\theta$):阈值。这是一个预设的超参数（比如 2.0 或 10.0），是划分好数据和坏数据的分界线。
        # torch.log(1 + torch.exp(...)):数学上的 Softplus 函数。它的形状像是一个平滑后的 ReLU，可以将逻辑判断转化为一个可求导的损失值。当条件不满足时，损失很大；当条件满足时，损失趋向于 0。
        loss = torch.log(1 + torch.exp(self.threshold - g_pos)).mean() + \
               torch.log(1 + torch.exp(g_neg - self.threshold)).mean()

        # 4. 执行局部反向传播 (仅针对这一层更新，不涉及其他层)，清空旧梯度
        self.opt.zero_grad()
        loss.backward()  # 这里的 backward 仅计算本层权重相对于本层 loss 的梯度
        """
            loss.backward() 计算完后，每个参数（如 weight）的内部都会存有一个梯度值：weight.grad。
            调用 .step() 时，优化器会按照设定的算法（如 SGD 或 Adam）修改权重。
                SGD模式：$$w_{new} = w_{old} - \text{learning\_rate} \times \text{gradient}$$
            使权重朝着能量（Loss）变小的方向微调了一点点
        """
        self.opt.step()

        """
            传统 BP（反向传播）：通常是整个网络跑完后，调用一次 optimizer.step()，统一更新所有层。
            FF 算法：由于每一层都是独立学习的， self.opt.step() 被写在每一层的 train 函数里。
            数据进来 → 算正/负样本能量 → 算出该层自己的 Loss → 该层立刻更新权重 (step) → 把归一化后的数据传给下一层。
            这种局部更新的特性，使得 FF 算法在理论上可以像流水线一样工作，而不需要等待深层网络的反馈。
        """

        return self.forward(x_pos).detach()  # 返回计算结果供下一层使用


# --- 简单测试脚本 ---
if __name__ == "__main__":
    # 创建一层：10个输入，20个神经元
    layer = FFLine(10, 20, threshold=5.0)

    # 模拟输入数据
    # x_pos 模拟“对的特征”，x_neg 模拟“随机噪声”
    pos_data = torch.randn(32, 10) + 2.0
    neg_data = torch.randn(32, 10) - 2.0

    print("开始局部训练...")
    for i in range(100):
        layer.train_layer(pos_data, neg_data)
        if i % 20 == 0:
            # 查看当前层的 Goodness 表现
            with torch.no_grad():
                goodness_pos = layer.forward(pos_data).pow(2).mean().item()
                goodness_neg = layer.forward(neg_data).pow(2).mean().item()
                print(f"迭代 {i}: 正样本Goodness={goodness_pos:.2f}, 负样本Goodness={goodness_neg:.2f}")