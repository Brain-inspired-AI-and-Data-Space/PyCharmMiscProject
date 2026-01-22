import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, tau=0.9):
        super().__init__()
        # 1. 设置神经元的固有参数
        self.v_threshold = v_threshold  # 阈值：电位达到多少会放电
        self.v_reset = v_reset  # 重置电位：放电后电位回到多少
        self.tau = tau  # 衰减系数 (Leak)：模拟电荷漏掉的速度

        # 2. 状态变量：膜电位 (Membrane Potential)
        # 初始时膜电位为 0
        self.v = None

    def forward(self, x):
        """
        x: 当前时刻输入的电流 (来自于权重乘法的结果 W*x)
        """
        # 如果是序列的第一个时间步，初始化膜电位张量，形状与输入一致
        if self.v is None:
            self.v = torch.zeros_like(x)

        # --- 步骤 1: 漏电与整合 (Leaky & Integrate) ---
        # 当前电位 = 历史残余电位 * 衰减系数 + 这一时刻收到的新电流
        # self.v * self.tau 模拟了“漏电”过程
        self.v = self.v * self.tau + x

        # --- 步骤 2: 判定是否发放脉冲 (Fire) ---
        # 产生一个二值张量：电位大于阈值的地方为 1，否则为 0
        # .float() 将布尔值转换为 1.0 和 0.0
        spike = (self.v >= self.v_threshold).float()

        # --- 步骤 3: 重置电位 (Reset) ---
        # 如果发放了脉冲 (spike=1)，则电位需要清零
        # 使用“软重置”或“硬重置”，这里演示硬重置：
        # 如果 spike 为 1，则 (1 - 1) = 0，电位被强制归零
        # 如果 spike 为 0，则 (1 - 0) = 1，电位保持不变
        self.v = self.v * (1 - spike) + self.v_reset * spike

        # 使用软重置

        # 如果 spike 为 0，则 v = v - threshold * 0 = v
        # 如果 spike 为 1，则 v = v - threshold
        self.v = self.v - self.v_threshold * spike

        # 返回这一时刻产生的脉冲
        return spike


# --- 模拟运行示例 ---
def simulate_lif():
    # 创建 5 个神经元
    neuron = LIFNeuron(v_threshold=1.0, tau=0.8)

    # 模拟 10 个时间步 (Time Steps)
    print("时间步 | 输入电流 | 膜电位 | 是否发放脉冲")
    print("-" * 40)

    for t in range(10):
        # 模拟持续稳定的输入电流 0.4
        input_current = torch.tensor([0.4])

        # 神经元步进
        spike = neuron(input_current)

        # 打印当前状态
        print(
            f" t={t}  |  {input_current.item():.1f}   | {neuron.v.item():.3f} | {'★ 脉冲!' if spike.item() > 0 else '○'}")


if __name__ == "__main__":
    simulate_lif()