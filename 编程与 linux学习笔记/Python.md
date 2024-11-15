# Python 学习笔记
## 有关 python解释器：
1. 解释器的概述
   -  python 的核心单元是**python 环境**，而python 解释器是一个 python环境的**核心**，环境各种安装的包均依赖于解释器，所以我们一般使用的虚拟环境工具，如 conda 环境，本质上就是对 python 解释器和其附属的所有包进行隔离，每个环境互不影响。
   -  python 解释器的**源代码**是一个 **C 语言编写的程序**，通常成为 **CPython**，我们下载的python 解释器通常是**预编译好的可执行文件**(python解释器真正的本质)，python 程序运行时接受输入的程序，将其转换为机器码，由机器处理之后输出。由于 C 语言最为高效且最接近机器底层，顾被选用作为 python 的解释器源代码的编写语言。
   -  但是运行 python 程序也需要依赖我们自己操作系统中的 C编译器，当python 项目中包含了C 拓展模块、或者一些对性能要求高的 python 库(如 numpy、Pandas等)其核心代码往往由 C 编写，这时就需要使用 C 编译器来编译， 在各个操作系统中并不相同，Linux 中是 **GCC**，MacOS 中是 **Clang**，Windows 中则为**Microsoft Visual C++ (MSVC)**
2. 如何使用 python 解释器
   1. 在终端(tty)中：
     - 终端python 解释器的工作特性类似于IDLE,即一次只能输入一条指令，即输即解释，可以通过在终端直接输入`python`来启动，启动之后即可输入指令执行，退出则是control-D 或输入`exit()`，也可以直接在终端输入`python -c 'command'`来直接执行 command 指令
## 有关python的文件结构
1. 脚本(Script)：通常是具有完整的执行逻辑，可以单独实现完整或部分功能的程序文件
   - 特征：往往包含`if __name__==__main:`这样的语句，表示只有这个脚本被单独执行时，才会执行下面的逻辑任务。这是为了使得这个脚本既能够单独作为一个脚本执行，也可以作为一个模块将其中定义的函数、变量、类等被其他程序引用，提高脚本的可重复利用性。
2. 模块(Module)：通常是定义了很多函数、变量、类，主要用来被其他脚本或模块调用的程序文件
3. 包(Package)：
4. 库(Library)：由模块和包构成，在所针对的功能上更加具有完整性，可以分为**标准库**和**第三方库**
   1. 标准库：是 python 内置的一些模块和包，分为**内置模块**和**纯 Python 模块和包**
      - 内置模块：是最常用的一些 python 模块，由 C编写，**直接编译进了解释器**，以实现最高的使用效率，包含：sys、os、time、math等，
      - 纯 Python 模块和包：较常用的模块和包，由 python 编写，被存放在该 python环境所在文件夹的**lib/python3.x**路径下
   2. 第三方库：后期用户自行下载安装的 python 库，被存放在**lib/python3.x/site-packages**路径下，如通过pip 和 conda 下载安装的库
5. 关于脚本、模块、包、库的导入
   - `import B`：
     - 首先，B 可以是**库、包、模块**，当执行 import 指令时，程序会依次在**内置模块**、**sys.path** 列表中查找目标，其中sys.path列表形状是['','/Users/liuquan/anaconda3/envs/tf/lib/python39.zip','/Users/liuquan/anaconda3/envs/tf/lib/python3.9','/Users/liuquan/anaconda3/envs/tf/lib/python3.9/lib-dynload','/Users/liuquan/anaconda3/envs/tf/lib/python3.9/site-packages'],分别表示**当前程序所在目录**、**当前python 环境标准库的 zip 压缩文件**、**当前python 环境标准库**、**当前python 环境动态链接库**、**当前python 环境第三方库**
   

6. 在索引和切片操作中，`:`表示在对应维度上选择所有元素
7. 有关类：
   1. 类中方法名称前有`_`的表示这个是私有方法，只能在类内部使用，外部不建议使用
8. 关于深拷贝和浅拷贝：
   1. 浅拷贝：`copy.copy()`只是拷贝该对象，不拷贝该对象嵌套的子对象，而是直接建立索引，两者并不独立
   2. 深拷贝：`copy.deepcopy()`拷贝该对象的同时将该对象嵌套的子对象也同时拷贝，拷贝后的变量和被拷贝的变量完全独立
9. 关于 python 内置模块：
   1. `sys.argv`：储存了命令行中的参数