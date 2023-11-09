from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载模型
model = load_model('my_model.keras',compile=False)

# 加载图片并进行预处理
img = image.load_img('/Users/liuquan/Desktop/Vscodeproject/NN/NUM/4.jpg', color_mode='grayscale',target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # 为批处理添加批次维度
img_array /= 255.0  # 图像归一化

# 预测
predictions = model.predict(img_array)

# 后处理
predicted_class = np.argmax(predictions, axis=1)

# 输出预测结果
print(f"Predicted class: {predicted_class}")