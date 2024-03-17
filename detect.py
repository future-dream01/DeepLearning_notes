from openvino.inference_engine import IECore
import cv2
import numpy as np
import os

def draw_polygon(image, points, color=(0, 255, 0), thickness=2):
    if len(points) < 5:
        print("Not enough points to draw a polygon")
        return
    num_points = len(points)
    for i in range(num_points):
        cv2.line(image, points[i], points[(i+1) % num_points], color, thickness)

model_xml = '/home/nuaa/CKYF_dafu/converted_models/yolox.xml'
model_bin = '/home/nuaa/CKYF_dafu/converted_models/yolox.bin'
image_folder = '/home/nuaa/CKYF_dafu/datasets/coco/images'

ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)
exec_net = ie.load_network(network=net, device_name="CPU")

input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue
    ih, iw = image.shape[:2]

    p_image = cv2.resize(image, (640, 640))
    p_image = p_image.transpose((2, 0, 1))
    p_image = np.expand_dims(p_image, axis=0)

    res = exec_net.infer({input_blob: p_image})

    output = res[out_blob]
    print(output.shape)  # 打印输出形状以帮助调试

    # 假设第一个维度是批量大小，我们只处理一个图像
    detections = output[0, :, :]
    for detection in detections:
        # 请确保detection的长度至少是10（5个顶点的x，y坐标）
        if len(detection) < 10:
            continue
        
        # 解析顶点坐标
        points = [(int(detection[i] * iw), int(detection[i + 1] * ih)) for i in range(0, 10, 2)]
        
        # 绘制多边形
        draw_polygon(image, points)

    cv2.imshow("Detection Results", image)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join("output", image_name), image)

cv2.destroyAllWindows()
