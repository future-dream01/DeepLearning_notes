# Linux学习笔记

1. 有关环境变量：在 ~/.bashrc 中，存放了系统用来查找文件和python模块的环境变量，~/.bashrc脚本会在每个shell对话开始前自动执行，所以通过在其中加入文件或python模块的路径就可以永久性让系统成功找到。对于python模块的路径，命令行输入`python3 -m site`就可显示sys.path列表，如果其中没有所需模块的路径，就在 ~/.bashrc中加入`export PYTHONPATH=${PYTHONPATH}:/path/to/your/model`，即永久性的告诉系统所需模块的路径