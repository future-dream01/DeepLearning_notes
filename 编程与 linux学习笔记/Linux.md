# Linux学习笔记

1. .bashrc脚本：
   1. 有关环境变量：在.bashrc脚本 中，存放了系统用来查找文件和python模块的环境变量，.bashrc脚本会在每个shell对话开始前自动执行，所以通过在其中加入文件或python模块的路径就可以永久性让系统成功找到。对于python模块的路径，命令行输入`python3 -m site`就可显示sys.path列表，如果其中没有所需模块的路径，就在 ~/.bashrc中加入`export PYTHONPATH=${PYTHONPATH}:/path/to/your/model`，即永久性的告诉系统所需模块的路径
   2. source语句：source语句用于在当前shell会话中执行脚本，该脚本中定义的所有变量、函数和环境设置都会直接应用于当前的shell环境。而直接运行脚本则仅会在另外一个shell会话中生效。
   3. export语句：用户将当前shell变量标记为环境变量，此变量在当前shell或其子shell或程序中生效，由于.bashrc默认在每一个shell打开前自动执行，所以将export语句添加到.bashrc中意味着此变量在系统中任意位置永久生效。
   4. echo语句：一般作用是在shell语法中用于打印输出，类似于python中的print语句，其同样可以用于写入文件，所以将某语句加入.bashrc语句可以使用`echo "Some text" > ~/.bashrc`
2. ros1笔记：
   1. 安装ros：`wget http://fishros.com/install -O fishros && . fishros`
   2. ros的基本结构：
      - 工作空间： 
         1. src：源码空间
            1. 功能包
               1. src:存放c++代码
               2. include：存放c++中的头文件
               3. script:存放python代码
               4. package.xml：存放依赖信息
               5. cmakelist.txt
         2. devel：开发空间，存放开发中产生的文件
         3. install：安装空间，存放开发后产生的文件
         4. build:编译过程中产生的中间文件
   3. 如何编写节点？
      1. 创建工作空间：
         1. 创建**dev_ws/src**文件夹
         2. 在src目录下运行`catkin_init_workspace`,来初始化工作空间
         3. 回到工作空间根目录，运行`catkin_make`,产生devel、build文件夹
         4. 再次运行`catkin_make install `,产生install文件夹
      2. 创建功能包：
         1. 切换到src目录下
         2. 执行`catkin_create_pkg <功能包名> <依赖> <依赖>……` 这里的依赖可以是：roscpp、rospy、std_msgs