import torch
import torch.nn as nn

# 假设我们有一个4x3的输入（4是批次大小，3是特征数量）
input = torch.randn(4, 3)

# 创建一个批量归一化层
bn = nn.BatchNorm1d(3)

# 对输入进行批量归一化
output = bn(input)

print("原始输入:")
print(input)
print("批量归一化后的输出:")
print(output)