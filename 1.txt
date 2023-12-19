import os
import random
import shutil

data_path = 'data/'
output_dir = 'car_voc/'
imgs_path = os.path.join(output_dir, 'JPEGImages/')
xmls_path = os.path.join(output_dir, 'Annotations/')
train_val_txt_path = os.path.join(output_dir, 'ImageSets/Main/')
if not os.path.exists(output_dir):
    os.makedirs(imgs_path)
    os.makedirs(xmls_path)
    os.makedirs(train_val_txt_path)
    
imgs_list = sorted([file for file in os.listdir(data_path) if file.split('.')[-1] == 'jpg'])
xmls_list = sorted([file for file in os.listdir(data_path) if file.split('.')[-1] == 'xml'])
random.shuffle(imgs_list)
train_txt = open(os.path.join(train_val_txt_path, 'train.txt'), 'w')
val_txt = open(os.path.join(train_val_txt_path, 'val.txt'), 'w')

train_test_split = 0.1
cnt = 1
for img, xml in zip(imgs_list, xmls_list):
    img_path = os.path.join(data_path, img)
    out_img_path = os.path.join(imgs_path, img)
    xml_path = os.path.join(data_path, xml)
    out_xml_path = os.path.join(xmls_path, xml)
    shutil.copyfile(img_path, out_img_path)
    shutil.copyfile(xml_path, out_xml_path)
    if cnt/len(imgs_list) > train_test_split:
        train_txt.write(img.split('.jpg')[0] + '\n')
    else:
        val_txt.write(img.split('.jpg')[0] + '\n')
    cnt+=1
train_txt.close()
val_txt.close()