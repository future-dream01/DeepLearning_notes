# Python 学习笔记
## 有关 python 解释器：
1. 解释器的概述
   -  python 的核心单元是**python 环境**，而python 解释器是一个 python环境的**核心**，环境各种安装的包均依赖于解释器，所以我们一般使用的虚拟环境工具，如 conda 环境，本质上就是对 python 解释器和其附属的所有包进行隔离，每个环境互不影响。
   -  python 解释器的**源代码**是一个 **C 语言编写的程序**，通常成为 **CPython**，我们下载的python 解释器通常是**预编译好的可执行文件**(python解释器真正的本质)，python 程序运行时接受输入的程序，将其转换为机器码，由机器处理之后输出。由于 C 语言最为高效且最接近机器底层，顾被选用作为 python 的解释器源代码的编写语言。
   -  但是运行 python 程序也需要依赖我们自己操作系统中的 C编译器，当python 项目中包含了C 拓展模块、或者一些对性能要求高的 python 库(如 numpy、Pandas等)其核心代码往往由 C 编写，这时就需要使用 C 编译器来编译， 在各个操作系统中并不相同，Linux 中是 **GCC**，MacOS 中是 **Clang**，Windows 中则为**Microsoft Visual C++ (MSVC)**
2. 如何使用 python 解释器
   1. 唤出解释器：
     - 终端 python 解释器的工作特性类似于UNIX shell,提供了一种交互方式，一次只能输入一条指令，即输即解释，可以通过在终端直接输入`python`来启动，启动之后即可输入指令执行，退出则是 control-D 或输入`exit()`，也可以直接在终端输入`python -c 'command'`来直接执行 command 指令
     - 关于如何执行 python 文件：
    2. 交互模式
## 有关 python 环境
1. python 环境是指python 程序运行所需的一系列功能库的配置，这些库依赖于一个核心的 python 解释器而存在，与解释器捆绑在一起。可以说，python 程序的执行成功与否，一半取决于代码内部逻辑，另一半则取决于 python 环境的配置是否合适。
2. 关于如何创建与管理 python 环境
   1. **CONDA 环境**：
      1. conda 的功能：
         - conda主要被用作 **python 环境管理器**，主要作用是针对不同的项目，创建、管理不同的 python 环境，以避免多个项目使用同一环境时会出现的依赖冲突问题
      2. 安装 conda：
         - windows：
            1. 下载地址：[Anaconda](https://www.anaconda.com/products/distribution)
            2. 安装：双击下载的安装包，一路下一步即可
         - Ubuntu：
            1. `wget https://repo.anaconda.com/anaconda/Anaconda3-latest-Linux-x86_64.sh`，终端输入，获取最新的Anaconda安装包
            2. `bash XXX.sh`，XXX 替换为下载得到的安装包名，安装Anaconda
            3. `source ~/.bashrc`,更新环境变量，使其 conda 环境生效
            4. `conda --version`,验证安装是否成功
      3. conda 的使用技巧：
         1. 对python环境整体的操作：
            1. `conda env list`:显示当前所有的 python 环境，标星号*表示当前环境
            2. `conda create --name env_name python=3.8`:创建一个名为 env_name 的 python3.8 环墶
            3. `conda env remove --name env_name`:删除名为 env_name 的 python 环境
            4. `conda create --name new_env --clone old_env`：复制一个 python 环境
            5. `conda activate env_name`:激活名为 env_name 的 python 环境
            6. `conda deactivate`:退出当前 python 环境 
         2. 在当前python环境内部的操作：
            1. `conda list`:显示当前conda 环境中所有安装所有的 python 包，**包括通过 pip 安装的**，而且会在最后显示来源<pypi>
            2. `which python`:显示当前 python 环境的解释器路径
   2. **PIP 包管理器** ：
      1. pip 的功能：
         1. pip 是 **python 包管理器**，主要作用是下载、安装、卸载 python 包 ，我个人一般喜欢搭配 conda 使用，即使用 conda 管理 python 环境，使用 pip 管理 python 包。使用 conda 创建好 python 环境后，一般会自动安装 pip 包管理器
      2. pip 的使用：
         1. `pip install package_name`:安装名为 package_name 的 python 包
         2. `pip install package_name -i https://pypi.tuna.tsinghua.edu.cn/simple`:使用清华镜像源安装名为 package_name 的 python 包
         3. `pip uninstall package_name`:卸载名为 package_name 的 python 包
         4. `pip list`:显示当前 python 环境中所有的 python 包
         5. `pip freeze > requirements.txt`:将当前 python 环境中所有的 python 包导出到 requirements.txt 文件中
         6. `pip install -r requirements.txt`:根据 requirements.txt 文件中的包名，安装所有的 python 包
## 有关 python 的文件结构
1. 脚本(Script)：通常是具有完整的执行逻辑，可以单独实现完整或部分功能的程序文件
   - 特征：往往包含`if __name__==__main:`这样的语句，表示只有这个脚本被单独执行时，才会执行下面的逻辑任务。这是为了使得这个脚本既能够单独作为一个脚本执行，也可以作为一个模块将其中定义的函数、变量、类等被其他程序引用，提高脚本的可重复利用性。
2. 模块(Module)：通常是定义了很多函数、变量、类，主要用来被其他脚本或模块调用的程序文件
   - 关于模块(或脚本，本质相同)的属性：
     1. __name__:模块的名称，作为脚本直接运行时，为**__main__**,被导入时，为模块名
     2. __file__:模块在系统中的绝对路径
     3. __package__:模块的父包名
     4. __all__:模块的公共接口，当该模块被其他模块或脚本导入时，如果导入语句出现了`import *`，那么将只会将该模块中__all__属性包含的类、函数、变量导入其命名空间，其余不会导入
3. 包(Package)：包含python模块和脚本文件
   - python包的重要标志是包内含有**__init__.py**文件，此文件可以为空，也可以写上一些逻辑语句，使用最多的是直接将本包中的**某个模块中**的类、函数、变量等直接导入本包的命名空间，这样被导入的类、函数、变量在包中的层级和模块本身相当
4. 库(Library)：由模块和包构成，在所针对的功能上更加具有完整性，可以分为**标准库**和**第三方库**
   - 标准库：是 python 内置的一些模块和包，分为**内置方法**、**内置模块**和**纯 Python 模块和包**
      - 内置方法和函数：针对 python 内置类型（字符串、列表、字典、元组等）可使用的方法，被直接编译进了 python 解释器，在使用时直接使用即可，不必 import 导入，使用效率最高
      - 内置模块：是最常用的一些 python 模块，由 C编写，**直接编译进了解释器**，以实现较高的使用效率，包含：sys、os、time、math等，但是使用时仍需导入，这是为了代码的可读性与可维护性
      - 纯 Python 模块和包：较常用的模块和包，由 python 编写，被存放在该 python环境所在文件夹的**lib/python3.x**路径下，散装，使用时需要导入，运行时需要单独编译
   - 第三方库：后期用户自行下载安装的 python 库，被存放在**lib/python3.x/site-packages**路径下，如通过pip 和 conda 下载安装的库
5. 关于脚本、模块、包、库的导入
   1. 关于导入的本质：无论是`from A import B`还是`import B`，本质上都是将 B 导入到**当前模块或脚本的命名空间**中，从此在当前模块或脚本的逻辑时间内，通过访问B 这个字符就可以访问 B 原先代表的包、模块、脚本、类、函数、变量等
   2. 绝对导入: 用的更多
      - 关于搜索路径：程序会依次在**内置模块**、**sys.path** 列表中查找目标，其中sys.path列表形状是[**当前程序所在目录**，**当前python 环境标准库目录**，**当前python 环境动态链接库目录**，**当前python 环境第三方库目录**，**环境变量 PYTHONPATH包含的目录**]
   3. 相对导入: 特征为出现`.A`或`..A`等，作用是避免自己的包内的某个模块和sys.path所有路径中的已有模块发生名称冲突
      - .表示当前目录，..表示上级目录，以此类推
      - 关于搜索路径: 此时python**不会**在内置模块和sys.path中查找A,相对导入的文件查找原理是：首先获取本文件的**__file__属性**，即本文件在系统中的绝对路径，然后获取**__package__属性**，这个是本文件的父包名，通过这两条信息，再结合 **.** 、**..**、**...**等相对位置信息，python 就知道改到哪里去找后面的A，可以看出这样可以使得查找时仅关注本项目中的文件，避免和其余文件重名，提高可移植性。
      - 使用条件：
        1. 只能在被调用的模块中使用。不能在顶级脚本中使用，即不能在一个单独运行的python 程序中使用相对导入，因为此时该脚本的__package__属性为 None，除非在运行顶级脚本时使用`python -m `，才可以使得python 将该脚本作为包的一部分来执行
        2. 只能在包内使用。即父目录中一定得有__init__.py 文件，使得父目录成为一个 python 包，但是导入对象，即..A 中的 A 不一定得是 python包，不过为了项目的整体性，建议将每个文件夹都处理为 python 包。








## python 的语言
1. 数据：
   - 数值类型
      - 整数(int)：指没有小数部分的数，python 中的整数大小没有任何限制(只要内存允许)。表示方法分为十进制(正常数字如 123)、二进制(0b 开头，如 0b1010)、八进制(0o开头，如 0o12)、十六进制(0x 开头，如 0xA)
      - 浮点数(float)：有小数部分的数，python 中最多可以精确到**小数点后 15-17 位**，再下一位则进行舍入。表示方法分为十进制和科学计数法，其中科学计数法以 e 或 E 后的数字表示指数位，如 1.23e-5 表示 1.23*10^-5
      - 复数(complex):指具有实部和虚部的数值，形式为 a+bj；具有实部属性.real 和虚部属性.imag
   - 序列类型：
      - 字符串
      - 列表
      - 元组
      - 集合
      - 字典(映射)
   - 特殊类型
      - 布尔值
      - 空值
2. 语句
   1. 基本组成单元
      1. 关键字
      2. 标识符
      3. 操作符
         1. 算数运算
         2. 比较运算
         3. 逻辑运算
         4. 赋值运算
      4. 字面量
      5. 分隔符
      6. 注释
   2. 简单语句：
   3. 复合语句：
      - if
      - while
      - for
      - try
3. 类
4. 标准库
   1. 内置方法和函数
      - `type()`:查看数据类型
      - `len(iterable)`:返回可迭代对象的长度;**iterable**为可迭代对象如列表、元组、字典等
      - `sorted(iterable, key=None, reverse=False)`:对可迭代对象进行排序；**iterable**为可迭代对象如列表、元组、字典等；**key**：是一个函数选项，用于自定义排序；**reverse**为布尔值，True为降序排序，False为升序排序
      - `complex(a,b)`:创建复数;使用方法：complex(a,b),即创建 a+bj
      - `.conjugate()`：创建共轭复数；使用方法:a=b.conjugate(),其中 b 为某已知复数
   2. 内置模块
      - sys：提供python解释器与**系统交互**有关的变量
        - `.argv`：命令行参数列表
        - `.path`：模块搜索路径列表
        - `.platform`： 当前运行的操作系统平台
        - `.version`: 当前环境中Python解释器版本信息
      - os：
        - getcwd():获取当前的绝对工作路径
        - listdir(path):输入任意路径，返回该路径下所有文件和目录组成的列表
        - walk(path):
        - rmkdir(path):输入任意路径，删除指定路径下的文件夹；只能删除空文件夹
        - path子模块：
          - exists(path):输入任意文件夹或文件的绝对路径，返回该指定路径下的文件夹或文件是否存在，存在返回True，不存在返回False
          - mkdir(path):输入任意文件夹绝对路径，在其目录路径下创建指定文件名的单个单层文件夹；如果创建前已存在该文件夹则会报错
          - mkdirs(path)：输入任意文件夹绝对路径，在其目录路径下创建指定文件名的多个递归文件夹；如果创建前已存在该文件夹则会报错
          - join(path1,path2):传入两个路径，拼接起来形成新的完整路径
          - split(path):传入任意文件的绝对路径，返回将其拆分成(目录路径,文件名)的元组
          - dirname(path):传入任意文件绝对路径，只获取其目录路径
          - basename(path):传入任意文件绝对路径，只获取其文件名
          - isdir(path)：传入任意文件绝对路径，判断起是否是文件夹
          - isfile(path):传入任意文件绝对路径，判断其是否是文件
          - abspath(path):输入任意文件的相对路径，将其和当前的目录路径相整合组成绝对路径
   3. 纯 python 模块和包
5. 错误与异常