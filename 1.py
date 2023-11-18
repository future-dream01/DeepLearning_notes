# 一个简单的python绘图，辅助理解神经网络算法

import numpy as np
import matplotlib.pyplot as plt

# 定义ReLU函数
def relu(x):
    return (x*x)

# 定义输入数据
x = np.linspace(-10, 10, 1000)
h1 = relu(-23*(relu(2*x-1))-4)
h2 = relu(5*(relu(6*x-2))+9)
h3 = relu(72*(relu(1*x+7))+2)
h4 = relu(4*(relu(63*x+72))+24)
h5 = relu(-3*(relu(-4*x+7))+92)
h6 = relu(22*(relu(3*x+7))-26)
h7 = relu(7*(relu(1*x+7))+44)


# 绘制函数图像
plt.plot(x, h1+h2+h3+h4+h5+h6+h7)
plt.show()