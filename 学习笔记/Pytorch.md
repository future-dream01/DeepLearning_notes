# pytorch 框架
关于 pytorch：pytorch 是 torch 开放的 python 接口，两者均由 Facebook 的人工智能小组开发，是开源的深度学习框架，特点是灵活且高效
## 数据集准备
1. 总述：pytorch 数据流如下：先将训练的数据(图片或矩阵)进行打包，创建一个**Dataset 类数据集对象** >>> 利用这个数据集对象创建一个 **Dataloader类数据加载器对象** >>> 使用数据加载器类对象迭代产生训练数据 >>> 经过模型内部进行处理 >>> 模型输出，后续可进行损失计算或直接输出
2. **Dataset类**数据集对象创建器：
   - 导入：`from torch.utils.data import Dataset`
   - 使用方法：
      - 关于 pytorch 中预先定义的 Dataset 抽象类：
         - pytorch中的数据集本质上是一个处理索引到具体样本之间映射的类，内部不复杂
      - 对Datasets做自己的修改：

3. **Dataloader类**数据加载器：
   - 导入：`from torch.utils.data import Dataloader`
   - 创建一个数据加载器对象：`dataloader=DataLoader(dataset,batch_size,shuffle, num_workers) `
      - **dataset**:之前定义好的数据集对象
      - **batch_size**：批次数，每次迭代进入网络内部的图片数
      - **shuffle**：是否数据洗牌，即每个 epoch 开始时是否将数据集中的数据打乱，Ture/Falth
      - **num_workers**： 数据加载时使用的进程数
  
## 模型搭建
1. 总述：pytorch 的核心即预定义好了一系列网络核心构件，可以让使用者像搭积木一样拼接网络，所有包含在模型通路中的组件，都需要是 `Module`子类，这样才可以被参数更新
2. 有关模块
   - **神经网络模型**模块
      1. 导入：`import torch.nn`
         - 使用方法：
         - `class Name(nn.Module):`:在父类基础上继承自己的模型子类
         - `super(Name,self).__init__()`：初始化父类
         - 模块中预定义的**卷积层**：
            - `nn.Conv2d(in_channels,out_channels,kernel_size,stride)`: 二维卷积层，最常用于处理图像数据。
               - in_channels:输入通道数
               - out_channels:输出通道数
               - kernel_size：卷积核大小
               - stride：卷积步长
            - `nn.ConvTranspose2d(in_channels, out_channels, kernel_size,stride, padding, output_padding)`: 二维转置卷积层，用于图像数据的上采样,常见于生成对抗网络（GANs）
               - in_channels:
               - out_channels:
               - kernel_size:
               - stride:
               - padding:
               - output_padding:
            - `nn.DepthwiseConv2d`: 深度可分离二维卷积层，其中每个输入通道被单独卷积。
               - 2
            - `nn.GroupConv2d`: 分组卷积层，它将输入和输出通道分成多组，以减少参数数量和计算量。
         - 模块中预定义的**批量归一化层**：
            - `nn.BatchNorm2d(in_channels)`:批量归一化，对当前批次中的特征图中特征值进行归一化
               - in_channels：输入通道数
         - 模块中定义的**激活函数层**：
            - `nn.Relu()`: Relu 激活函数，直接使用，不包含可更新参数
         - 模块中定义的**池化层**：
            - `nn.MaxPool2d(kernel_size=2, stride=2)`:最大池化层，用于提取局部区域中的最大值
               - kernel_size：卷积核尺寸
               - stride：卷积核移动步长
            - `nn.AvgPool2d(kernel_size=2, stride=2)`：平均池化层，用于计算局部区域的平均值
            - `nn.AdaptiveAvgPool2d(1)`，全局平均池化层，在每个通道上取整个特征图的平均值，通常用于分类任务的最后一个池化层，以减少特征图的尺寸到1x1
         - 模块中定义的**全连接层**：
   - **functional类**：函数模块
      - 导入：`import torch.nn.functional`
      - 模块中预定义的函数：
         - 上/下采样：
            - `functional.interpolate(image, scale_factor, mode, align_corners)`
               - image:输入图像
               - scale_factor：缩放因子（0～正无穷）
               - mode：缩放模式，有：
                  - ‘bilinear’：双线性插值
                  - 
               - align_corners：是否对输入输出的角落像素进行对齐，设置为False可避免潜在锐化

## 训练、推理：
1. 定义损失函数和梯度优化器：
   - 常用损失函数：
     - `nn.CrossEntropyLoss(model_output,target)`：交叉熵损失函数，用于多分类
        - model_output：模型的全连接层输出，不需要经过 softmax
        - target：标签
     - `nn.BCELoss(model_output,target)`：二元交叉熵损失函数，用于二分类
        - model_output：模型的全连接层输出
        - target：标签
     - `nn.BCEWithLogitsLoss(model_output,target)`：带有 logits 的二元交叉熵损失函数
        - model_output：模型的全连接层输出
        - target：标签
     - `nn.MSELoss(model_output,target)`：均方误差损失函数
        - model_output：模型的全连接层输出
        - target：标签
     - `nn.L1Loss(model_output,target)`：最小绝对误差损失函数
        - model_output：模型的全连接层输出
        - target：标签 
   - 常用梯度优化器：
     - 导入：`import nn.optim as optim`
     - `optim.SGD(params, lr)`:随机梯度下降,最基本的优化器，适用于大多数情况
     - `optim.RMSprop(params, lr)`:适用于处理非平稳目标和非线性优化问题。
     - `optim.Adam(params, lr)`:自适应矩估计,结合了动量和 RMSprop 的优点，适用于大多数非凸优化问题，是目前最流行的优化方法之一。
     - `optim.AdamW(params, lr)`:对 Adam 的改进版本，添加了权重衰减，通常用于正则化和防止过拟合。
     - `optim.Adagrad(params, lr)`:适用于处理稀疏数据.
     - `optim.Adadelta(params, lr)`:是 Adagrad 的扩展，旨在减少其学习率的单调递减。
   - 损失函数和梯度优化器的使用：
     - 先创建损失函数对象criterion和梯度优化器对象optimizer
     - 更新某一模块之前，首先梯度归零：`optimizer_d.zero_grad()`
     - `criterion(model_output,target)`:使用损失函数得到损失值 loss
     -  `loss.backward()` ：对损失使用反向传播算法，获取梯度
     -  `optimizer_d.step()`：使用梯度下降算法更新参数
   -  特殊使用：
      - 在生成对抗网络中，生成器和判别器一般是先后训练，由于总的数据流是从生成器流向判别器，更新判别器参数时会默认把生成器也进行更新，所以需要在判别器的输入层将来自生成器的特征图进行信息切断，使其与生成器的模型没有关联关系：`discriminator(fake_images.detach())`
   
## 模型性能评估

## 模型权重保存、加载