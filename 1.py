classdef app1 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure         matlab.ui.Figure
        GridLayout       matlab.ui.container.GridLayout
        LeftPanel        matlab.ui.container.Panel
        nmsSpinner       matlab.ui.control.Spinner
        nmsSpinnerLabel  matlab.ui.control.Label
        confSpinner      matlab.ui.control.Spinner
        confLabel        matlab.ui.control.Label
        Button_13        matlab.ui.control.Button
        Label_24         matlab.ui.control.Label
        Button_11        matlab.ui.control.Button
        Button_10        matlab.ui.control.Button
        Label_14         matlab.ui.control.Label
        Label_13         matlab.ui.control.Label
        Button_4         matlab.ui.control.Button
        Button           matlab.ui.control.Button
        DropDown         matlab.ui.control.DropDown
        Label_7          matlab.ui.control.Label
        EditField        matlab.ui.control.EditField
        Label_6          matlab.ui.control.Label
        Label_5          matlab.ui.control.Label
        Label_2          matlab.ui.control.Label
        RightPanel       matlab.ui.container.Panel
        Button_14        matlab.ui.control.Button
        Label_22         matlab.ui.control.Label
        Image            matlab.ui.control.Image
        Button_12        matlab.ui.control.Button
        Button_8         matlab.ui.control.Button
        EditField_3      matlab.ui.control.NumericEditField
        Label_11         matlab.ui.control.Label
        EditField_2      matlab.ui.control.NumericEditField
        Label_10         matlab.ui.control.Label
        Button_5         matlab.ui.control.Button
        Label_8          matlab.ui.control.Label
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startup(app)
            app.UIFigure.Name="南京航空航天大学";
        end

        % Button pushed function: Button_10
        function Button_10Pushed(app, event)
[File,Path] = uigetfile('*.jpg;*.png;*.bmp','Select an image');

        end

        % Button pushed function: Button_8
        function Button_8Pushed(app, event)
            warndlg("确定退出吗？")
        end

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, event)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {543, 543};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {257, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 697 543];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {257, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.ForegroundColor = [0.0745 0.6235 1];
            app.LeftPanel.BackgroundColor = [0.302 0.7451 0.9333];
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create Label_2
            app.Label_2 = uilabel(app.LeftPanel);
            app.Label_2.Position = [32 294 25 22];
            app.Label_2.Text = '';

            % Create Label_5
            app.Label_5 = uilabel(app.LeftPanel);
            app.Label_5.Position = [38 171 25 22];
            app.Label_5.Text = '';

            % Create Label_6
            app.Label_6 = uilabel(app.LeftPanel);
            app.Label_6.HorizontalAlignment = 'right';
            app.Label_6.Position = [30 121 74 22];
            app.Label_6.Text = '输入尺寸      ';

            % Create EditField
            app.EditField = uieditfield(app.LeftPanel, 'text');
            app.EditField.Position = [119 121 100 22];
            app.EditField.Value = '640';

            % Create Label_7
            app.Label_7 = uilabel(app.LeftPanel);
            app.Label_7.HorizontalAlignment = 'right';
            app.Label_7.Position = [30 69 77 22];
            app.Label_7.Text = '计算设备       ';

            % Create DropDown
            app.DropDown = uidropdown(app.LeftPanel);
            app.DropDown.Items = {'GPU', 'CPU'};
            app.DropDown.Position = [122 69 100 22];
            app.DropDown.Value = 'GPU';

            % Create Button
            app.Button = uibutton(app.LeftPanel, 'push');
            app.Button.Position = [10 427 106 32];
            app.Button.Text = '训练模式';

            % Create Button_4
            app.Button_4 = uibutton(app.LeftPanel, 'push');
            app.Button_4.Position = [143 427 106 32];
            app.Button_4.Text = '分类模式';

            % Create Label_13
            app.Label_13 = uilabel(app.LeftPanel);
            app.Label_13.Position = [39 373 53 22];
            app.Label_13.Text = '图片路径';

            % Create Label_14
            app.Label_14 = uilabel(app.LeftPanel);
            app.Label_14.Position = [39 273 53 22];
            app.Label_14.Text = '权重文件';

            % Create Button_10
            app.Button_10 = uibutton(app.LeftPanel, 'push');
            app.Button_10.ButtonPushedFcn = createCallbackFcn(app, @Button_10Pushed, true);
            app.Button_10.Position = [116 370 100 25];
            app.Button_10.Text = '选择文件';

            % Create Button_11
            app.Button_11 = uibutton(app.LeftPanel, 'push');
            app.Button_11.Position = [118 272 100 25];
            app.Button_11.Text = '选择文件';

            % Create Label_24
            app.Label_24 = uilabel(app.LeftPanel);
            app.Label_24.Position = [40 326 53 22];
            app.Label_24.Text = '配置文件';

            % Create Button_13
            app.Button_13 = uibutton(app.LeftPanel, 'push');
            app.Button_13.Position = [117 325 100 25];
            app.Button_13.Text = '选择文件';

            % Create confLabel
            app.confLabel = uilabel(app.LeftPanel);
            app.confLabel.HorizontalAlignment = 'right';
            app.confLabel.Position = [29 220 75 22];
            app.confLabel.Text = 'conf置信度   ';

            % Create confSpinner
            app.confSpinner = uispinner(app.LeftPanel);
            app.confSpinner.Position = [119 220 100 22];
            app.confSpinner.Value = 0.6;

            % Create nmsSpinnerLabel
            app.nmsSpinnerLabel = uilabel(app.LeftPanel);
            app.nmsSpinnerLabel.HorizontalAlignment = 'right';
            app.nmsSpinnerLabel.Position = [31 171 76 22];
            app.nmsSpinnerLabel.Text = 'nms阈值       ';

            % Create nmsSpinner
            app.nmsSpinner = uispinner(app.LeftPanel);
            app.nmsSpinner.Position = [119 171 103 22];
            app.nmsSpinner.Value = 0.2;

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.ForegroundColor = [0.9412 0.9412 0.9412];
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create Label_8
            app.Label_8 = uilabel(app.RightPanel);
            app.Label_8.FontSize = 23;
            app.Label_8.Position = [27 485 379 32];
            app.Label_8.Text = '光子层析图像训练、分类一体化平台';

            % Create Button_5
            app.Button_5 = uibutton(app.RightPanel, 'push');
            app.Button_5.BackgroundColor = [0.3922 0.8314 0.0745];
            app.Button_5.Position = [119 394 114 65];
            app.Button_5.Text = '开始分类';

            % Create Label_10
            app.Label_10 = uilabel(app.RightPanel);
            app.Label_10.HorizontalAlignment = 'right';
            app.Label_10.Position = [28 347 53 22];
            app.Label_10.Text = '已完成数';

            % Create EditField_2
            app.EditField_2 = uieditfield(app.RightPanel, 'numeric');
            app.EditField_2.Position = [96 347 100 22];
            app.EditField_2.Value = 12;

            % Create Label_11
            app.Label_11 = uilabel(app.RightPanel);
            app.Label_11.HorizontalAlignment = 'right';
            app.Label_11.Position = [28 286 53 22];
            app.Label_11.Text = '图片总数';

            % Create EditField_3
            app.EditField_3 = uieditfield(app.RightPanel, 'numeric');
            app.EditField_3.Position = [96 286 100 22];
            app.EditField_3.Value = 416;

            % Create Button_8
            app.Button_8 = uibutton(app.RightPanel, 'push');
            app.Button_8.ButtonPushedFcn = createCallbackFcn(app, @Button_8Pushed, true);
            app.Button_8.Position = [382 21 43 51];
            app.Button_8.Text = '退出';

            % Create Button_12
            app.Button_12 = uibutton(app.RightPanel, 'push');
            app.Button_12.Position = [134 68 51 25];
            app.Button_12.Text = '上一张';

            % Create Image
            app.Image = uiimage(app.RightPanel);
            app.Image.Position = [86 117 234 131];

            % Create Label_22
            app.Label_22 = uilabel(app.RightPanel);
            app.Label_22.Position = [33 231 53 22];
            app.Label_22.Text = '效果预览';

            % Create Button_14
            app.Button_14 = uibutton(app.RightPanel, 'push');
            app.Button_14.Position = [220 68 51 25];
            app.Button_14.Text = '下一张';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = app1

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startup)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
classdef app12 < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure     matlab.ui.Figure
        Button_4     matlab.ui.control.Button
        Button_3     matlab.ui.control.Button
        Button_2     matlab.ui.control.Button
        Button       matlab.ui.control.Button
        EditField_2  matlab.ui.control.NumericEditField
        Label_3      matlab.ui.control.Label
        EditField    matlab.ui.control.NumericEditField
        Label_2      matlab.ui.control.Label
        Label        matlab.ui.control.Label
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: Button
        function ButtonPushed(app, event)
            warndlg("登陆成功！")
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create Label
            app.Label = uilabel(app.UIFigure);
            app.Label.FontSize = 22;
            app.Label.Position = [100 387 453 30];
            app.Label.Text = '光子层析图像训练、分类一体化平台用户登录';

            % Create Label_2
            app.Label_2 = uilabel(app.UIFigure);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.Position = [243 289 41 22];
            app.Label_2.Text = '用户名';

            % Create EditField
            app.EditField = uieditfield(app.UIFigure, 'numeric');
            app.EditField.Position = [299 289 100 22];
            app.EditField.Value = 1234;

            % Create Label_3
            app.Label_3 = uilabel(app.UIFigure);
            app.Label_3.HorizontalAlignment = 'right';
            app.Label_3.Position = [255 245 29 22];
            app.Label_3.Text = '密码';

            % Create EditField_2
            app.EditField_2 = uieditfield(app.UIFigure, 'numeric');
            app.EditField_2.Position = [299 245 100 22];
            app.EditField_2.Value = 1234;

            % Create Button
            app.Button = uibutton(app.UIFigure, 'push');
            app.Button.ButtonPushedFcn = createCallbackFcn(app, @ButtonPushed, true);
            app.Button.Position = [156 173 100 25];
            app.Button.Text = '登录';

            % Create Button_2
            app.Button_2 = uibutton(app.UIFigure, 'push');
            app.Button_2.Position = [412 173 100 25];
            app.Button_2.Text = '取消';

            % Create Button_3
            app.Button_3 = uibutton(app.UIFigure, 'push');
            app.Button_3.Position = [156 94 100 25];
            app.Button_3.Text = '保存用户名';

            % Create Button_4
            app.Button_4 = uibutton(app.UIFigure, 'push');
            app.Button_4.Position = [412 94 100 25];
            app.Button_4.Text = '保存密码';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = app12

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end


训练程序源码：

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import Exp, check_exp_value, get_exp
from yolox.utils import configure_module, configure_nccl, configure_omp, get_num_devices

def make_parser():
parser = argparse.ArgumentParser("YOLOX train parser")
parser.add_argument("-expn", "--experiment-name", type=str, default=None)
parser.add_argument("-n", "--name", type=str, default=None, help="model name")

# distributed
parser.add_argument(
"--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
"--dist-url",
default=None,
type=str,
help="url used to set up distributed training",
)
parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
parser.add_argument(
"-d", "--devices", default=None, type=int, help="device for training"
)
parser.add_argument(
"-f",
"--exp_file",
default=None,
type=str,
help="plz input your experiment description file",
)
parser.add_argument(
"--resume", default=False, action="store_true", help="resume training"
)
parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
parser.add_argument(
"-e",
"--start_epoch",
default=None,
type=int,
help="resume training start epoch",
)
parser.add_argument(
"--num_machines", default=1, type=int, help="num of node for training"
)
parser.add_argument(
"--machine_rank", default=0, type=int, help="node rank for multi-node training"
)
parser.add_argument(
"--fp16",
dest="fp16",
default=False,
action="store_true",
help="Adopting mix precision training.",
)
parser.add_argument(
"--cache",
type=str,
nargs="?",
const="ram",
help="Caching imgs to ram/disk for fast training.",
)
parser.add_argument(
"-o",
"--occupy",
dest="occupy",
default=False,
action="store_true",
help="occupy GPU memory first for training.",
)
parser.add_argument(
"-l",
"--logger",
type=str,
help="Logger to be used for metrics. \
Implemented loggers include `tensorboard` and `wandb`.",
default="tensorboard"
)
parser.add_argument(
"opts",
help="Modify config options using the command-line",
default=None,
nargs=argparse.REMAINDER,
)
return parser

@logger.catch
def main(exp: Exp, args):
if exp.seed is not None:
random.seed(exp.seed)
torch.manual_seed(exp.seed)
cudnn.deterministic = True
warnings.warn(
"You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
"which can slow down your training considerably! You may see unexpected behavior "
"when restarting from checkpoints."
)

# set environment variables for distributed training
configure_nccl()
configure_omp()
cudnn.benchmark = True

trainer = exp.get_trainer(args)
trainer.train()

if __name__ == "__main__":
configure_module()
args = make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)
exp.merge(args.opts)
check_exp_value(exp)

if not args.experiment_name:
args.experiment_name = exp.exp_name

num_gpu = get_num_devices() if args.devices is None else args.devices
assert num_gpu <= get_num_devices()

if args.cache is not None:
exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

dist_url = "auto" if args.dist_url is None else args.dist_url
launch(
main,
num_gpu,
args.num_machines,
args.machine_rank,
backend=args.dist_backend,
dist_url=dist_url,
args=(exp, args),
)


分类程序源码：
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.boxes import poly_postprocess,min_rect

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
parser = argparse.ArgumentParser("YOLOX Demo!")
parser.add_argument(
"demo", default="image", help="demo type, eg. image, video and webcam" # 指定数据源是图像还是视频
)
parser.add_argument("-expn", "--experiment-name", type=str, default=None) # 指定实验名称
parser.add_argument("-n", "--name", type=str, default=None, help="model name") # 指定模型名称

parser.add_argument(
"--path", default="./assets/dog.jpg", help="path to images or video" # 指定数据源路径
)
parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id") # 指定摄像头id
parser.add_argument(
"--save_result",
action="store_true",
help="whether to save the inference result of image/video",
) # 指定是否保存识别结果

# exp file
parser.add_argument(
"-f",
"--exp_file",
default=None,
type=str,
help="pls input your experiment description file",
) # 指定实验配置文件
parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval") # 指定预训练权重文件
parser.add_argument(
"--device",
default="cpu",
type=str,
help="device to run our model, can either be cpu or gpu",
) # 指定实验设备：cpu/gpu
parser.add_argument("--conf", default=0.3, type=float, help="test conf") # 指定置信度最低值
parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold") # 指定nms阈值
parser.add_argument("--tsize", default=None, type=int, help="test img size") # 指定输入图像尺寸
parser.add_argument(
"--fp16",
dest="fp16",
default=False,
action="store_true",
help="Adopting mix precision evaluating.",
) # 指定是否使用fp16加速运算
parser.add_argument(
"--legacy",
dest="legacy",
default=False,
action="store_true",
help="To be compatible with older versions",
)
parser.add_argument(
"--fuse",
dest="fuse",
default=False,
action="store_true",
help="Fuse conv and bn for testing.",
)
parser.add_argument(
"--trt",
dest="trt",
default=False,
action="store_true",
help="Using TensorRT model for testing.",
) # 是否使用tensorrt加速
return parser

def get_image_list(path):
image_names = []
for maindir, subdir, file_name_list in os.walk(path):
for filename in file_name_list:
apath = os.path.join(maindir, filename)
ext = os.path.splitext(apath)[1]
if ext in IMAGE_EXT:
image_names.append(apath)
return image_names

class Predictor(object):
def __init__(
self,
model,
exp,
cls_names=COCO_CLASSES,
trt_file=None,
decoder=None,
device="cpu",
fp16=False,
legacy=False,
):
self.model = model
self.cls_names = cls_names
self.decoder = decoder
self.num_apexes = exp.num_apexes
self.num_classes = exp.num_classes
#self.num_colors = exp.num_colors
self.confthre = exp.test_conf
self.nmsthre = exp.nmsthre
self.test_size = exp.test_size
self.device = device
self.fp16 = fp16
self.preproc = ValTransform(legacy=legacy)
if trt_file is not None:
from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(trt_file))

x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
self.model(x)
self.model = model_trt

def inference(self, img):
img_info = {"id": 0}
if isinstance(img, str):
img_info["file_name"] = os.path.basename(img)
img = cv2.imread(img)
else:
img_info["file_name"] = None

height, width = img.shape[:2]
img_info["height"] = height
img_info["width"] = width
img_info["raw_img"] = img

ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
img_info["ratio"] = ratio

img, _ = self.preproc(img, None, self.test_size)
img = torch.from_numpy(img).unsqueeze(0)
img = img.float()
if self.device == "gpu":
img = img.cuda()
if self.fp16:
img = img.half() # to FP16

with torch.no_grad():
t0 = time.time()
outputs = self.model(img)
if self.decoder is not None:
outputs = self.decoder(outputs, dtype=outputs.type())
bbox_preds = []
#Convert[reg,conf,color,classes] into [bbox,conf,color and classes]
for i in range(outputs.shape[0]):
bbox = min_rect(outputs[i,:,:self.num_apexes * 2])
bbox_preds.append(bbox)
bbox_preds = torch.stack(bbox_preds)

conf_preds = outputs[:,:,self.num_apexes * 2].unsqueeze(-1)

cls_preds = outputs[:,:,self.num_apexes * 2 + 1 :]
#Initialize colors_preds
#colors_preds = torch.clone(cls_preds)

#for i in range(self.num_colors):
#colors_preds[:,:,i * self.num_classes:(i + 1) * self.num_classes] = outputs[:,:,self.num_apexes * 2 + 1 + i:self.num_apexes * 2 + 1 + i + 1].repeat(1, 1, self.num_classes)
cls_preds_converted = cls_preds

outputs_rect = torch.cat((bbox_preds,conf_preds,cls_preds_converted),dim=2)
outputs_poly = torch.cat((outputs[:,:,:self.num_apexes * 2],conf_preds,cls_preds_converted),dim=2)
#Out Format: (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
outputs = poly_postprocess(
outputs_rect,
outputs_poly,
self.num_apexes,
self.num_classes ,
self.confthre,
self.nmsthre
)
logger.info("Infer time: {:.4f}s".format(time.time() - t0))
return outputs, img_info

def visual(self, output, img_info, cls_conf=0.35):
ratio = img_info["ratio"]
img = img_info["raw_img"]
if output is None:
return img
output = output.cpu()

bboxes = output[:, 0:self.num_apexes*2]
# preprocessing: resize
bboxes /= ratio

cls = output[:, self.num_apexes*2 + 2]
scores = output[:, self.num_apexes*2] * output[:, self.num_apexes*2 + 1]

vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
return vis_res

def image_demo(predictor, vis_folder, path, current_time, save_result):
if os.path.isdir(path):
files = get_image_list(path)
else:
files = [path]
files.sort()
for image_name in files:
outputs, img_info = predictor.inference(image_name)
result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
if save_result:
save_folder = os.path.join(
vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
)
os.makedirs(save_folder, exist_ok=True)
save_file_name = os.path.join(save_folder, os.path.basename(image_name))
logger.info("Saving detection result in {}".format(save_file_name))
cv2.imwrite(save_file_name, result_image)
ch = cv2.waitKey(0)
if ch == 27 or ch == ord("q") or ch == ord("Q"):
break

def imageflow_demo(predictor, vis_folder, current_time, args):
cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
fps = cap.get(cv2.CAP_PROP_FPS)
if args.save_result:
save_folder = os.path.join(
vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
)
os.makedirs(save_folder, exist_ok=True)
if args.demo == "video":
save_path = os.path.join(save_folder, args.path.split("/")[-1])
else:
save_path = os.path.join(save_folder, "camera.mp4")
logger.info(f"video save_path is {save_path}")
vid_writer = cv2.VideoWriter(
save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
)
while True:
ret_val, frame = cap.read()
if ret_val:
outputs, img_info = predictor.inference(frame)
result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
if args.save_result:
vid_writer.write(result_frame)
else:
cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
cv2.imshow("yolox", result_frame)
ch = cv2.waitKey(1)
if ch == 27 or ch == ord("q") or ch == ord("Q"):
break
else:
break

def main(exp, args):
if not args.experiment_name:
args.experiment_name = exp.exp_name

file_name = os.path.join(exp.output_dir, args.experiment_name)
os.makedirs(file_name, exist_ok=True)

vis_folder = None
if args.save_result:
vis_folder = os.path.join(file_name, "vis_res")
os.makedirs(vis_folder, exist_ok=True)

if args.trt:
args.device = "gpu"

logger.info("Args: {}".format(args))

if args.conf is not None:
exp.test_conf = args.conf
if args.nms is not None:
exp.nmsthre = args.nms
if args.tsize is not None:
exp.test_size = (args.tsize, args.tsize)

model = exp.get_model()
logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

if args.device == "gpu":
model.cuda()
if args.fp16:
model.half() # to FP16
model.eval()

if not args.trt:
if args.ckpt is None:
ckpt_file = os.path.join(file_name, "best_ckpt.pth")
else:
ckpt_file = args.ckpt
logger.info("loading checkpoint")
ckpt = torch.load(ckpt_file, map_location="cpu")
# load the model state dict
model.load_state_dict(ckpt["model"])
logger.info("loaded checkpoint done.")

if args.fuse:
logger.info("\tFusing model...")
model = fuse_model(model)

if args.trt:
assert not args.fuse, "TensorRT model is not support model fusing!"
trt_file = os.path.join(file_name, "model_trt.pth")
assert os.path.exists(
trt_file
), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
model.head.decode_in_inference = False
decoder = model.head.decode_outputs
logger.info("Using TensorRT to inference")
else:
trt_file = None
decoder = None

predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
current_time = time.localtime()
if args.demo == "image":
image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
elif args.demo == "video" or args.demo == "webcam":
imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
args = make_parser().parse_args()
exp = get_exp(args.exp_file, args.name)

main(exp, args)