### MATLAB 学习笔记
1. matlab常识
   1. .m文件：MATLAB的脚本或函数文件格式，包含MATLAB代码
   2. .mat文件：可以存储各种类型的数据，包括变量、数组、矩阵、结构体、字符串、类对象等
   3. %：行注释
   4. `%{   %}`:段注释
   5. 
2. 变量；
   1. 变量命名原则：以字母或者下划线开头(很少用下划线打头)；后面可以跟字母、数字和下划线；变量名区分字母的大小写.
   2. ans：默认变量，无指定变量，则系统会自动将结果赋给变量 “ans”
   3. 变量的存储：`save  文件名  变量名列表` 其中文件是.mat文件
   4. 变量的读取：`load 文件名 A  x ` 
3. 语句：
   1. 语句形式：变量 = 表达式;
   2. 分号的作用：若不想在命令行窗口的屏幕上输出结果，可以在语句最后加分号
   3. 续行符：如果语句很长，可用续行符 “…”（三个点）续行； 续行符的前面最好留一个空格
   4. 输出语句：disp()
   5. **条件语句**：
      1. 纯if语句：`if 条件 执行代码  if 条件 执行代码 …… end`
         1. 每个if独立判断，上一个被执行，下一个依然会被判断。
      2. elseif语句：
         1. 只会执行从上到下第一个真，上一被执行真，下一个直接跳过
      3. 条件语句、循环语句等必须后面以end结尾，其相当于c语言中的{}、和python中的缩进符号，都表示一个代码区块
   6. **循环语句**:
      1. for循环：
         1. 默认步长为1：`for i =a:b 执行代码 end`
            1. 其中i由a到b
         2. 指定步长：`for i =a:c:b 执行代码 end`
            1. 其中i由a到b，每次步长为c
      2. while循环：`while 条件 执行代码 end`
         1. 需确保不会陷入死循环