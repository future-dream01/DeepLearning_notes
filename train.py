import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# 将图像的像素值标准化至0到1的范围
train_images, test_images = train_images / 255.0, test_images / 255.0

# 为图像数据增加一个维度，因为卷积层期望一个四维的输入 (batch_size, height, width, channels)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 构建模型
model = models.Sequential()
# 添加卷积层，32个滤波器，大小3x3，激活函数使用ReLU
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 添加最大池化层，池化窗口大小2x2
model.add(layers.MaxPooling2D((2, 2)))
# 再添加一个卷积层，64个滤波器
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 再添加一个最大池化层
model.add(layers.MaxPooling2D((2, 2)))
# 再添加一个卷积层，64个滤波器
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 展平层，将三维输出展开为一维
model.add(layers.Flatten())
# 添加一个有64个神经元的全连接层
model.add(layers.Dense(64, activation='relu'))
# 添加一个输出层，10个输出神经元对应10个类别，使用softmax激活函数
model.add(layers.Dense(10, activation='softmax'))

# 打印模型的结构
model.summary()

# 编译模型，使用adam优化器，交叉熵损失函数，准确率评估指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=8)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test accuracy: {test_acc}')
model.save('my_model.keras')  # creates a HDF5 file 'my_model.h5'