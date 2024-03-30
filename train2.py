import matplotlib.pyplot as plt
import numpy as np

# 定义函数
def f1(x):
    return -3*x+1

def f2(x):
    return 6*x-28

def f3(x):
    return 7*x-25

def f4(x):
    return 18*x+22

def relu(x):
    return np.maximum(0, x)


# 生成x值列表
x = np.linspace(-1000, 10000, 40000)  # 从-10到10生成400个点
y1 = f1(x)  # 计算每个x点对应的y值
y2 = f2(x)
y3 = f3(x)
y4 = y1+y2+y3
y5 = relu(y4)
y6 = f1(y5)
y7 = f2(y5)
y8 = f3(y5)
y9=y6+y7+y8
y10 = relu(y9)
y11 = f1(y10)
y12 = f2(y10)
y13 = f3(y10)
y14=y11+y12+y13
y15 = relu(y14)



# 创建图像
plt.figure()
plt.plot(x, y15, label='y = x^2')  # 绘制函数图像
#plt.title('Function Plot of y = x^2')  # 添加标题
plt.xlabel('x')  # 添加x轴标签
plt.ylabel('y')  # 添加y轴标签
plt.grid(True)  # 添加网格线
plt.legend()  # 添加图例
plt.show()  # 显示图像
