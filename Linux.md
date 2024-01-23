# Linux学习笔记

1. .bashrc脚本：
   1. 有关环境变量：在.bashrc脚本 中，存放了系统用来查找文件和python模块的环境变量，.bashrc脚本会在每个shell对话开始前自动执行，所以通过在其中加入文件或python模块的路径就可以永久性让系统成功找到。对于python模块的路径，命令行输入`python3 -m site`就可显示sys.path列表，如果其中没有所需模块的路径，就在 ~/.bashrc中加入`export PYTHONPATH=${PYTHONPATH}:/path/to/your/model`，即永久性的告诉系统所需模块的路径
   2. source语句：source语句用于在当前shell会话中执行脚本，该脚本中定义的所有变量、函数和环境设置都会直接应用于当前的shell环境。而直接运行脚本则仅会在另外一个shell会话中生效。
   3. export语句：用户将当前shell变量标记为环境变量，此变量在当前shell或其子shell或程序中生效，由于.bashrc默认在每一个shell打开前自动执行，所以将export语句添加到.bashrc中意味着此变量在系统中任意位置永久生效。
   4. echo语句：一般作用是在shell语法中用于打印输出，类似于python中的print语句，其同样可以用于写入文件，所以将某语句加入.bashrc语句可以使用`echo "Some text" > ~/.bashrc`