from torch.utils.data import Dataset,DataLoader
class Animal():
    number=0                               # 类变量
    def __init__(self,name,age,color,b):   # 初始化方法
        self.name=name                     # 实例属性
        self.age=age                       # 实例属性
        self.color=color                   # 实例属性
        self.__b=b
    def introduce(self):                   # 实例方法
        print(f"Hi,我的名字是小动物{self.name},wp")
    def sit(self):                         # 实例方法
        print(f"我坐下了!")
    def eat(self):                         # 实例方法
        print(f"我正在吃饭")
    def make_sound(self):                  # 抽象实例方法
        pass
    def change_b(self,c):
        self.__b=c
    @classmethod                           # 类方法
    def change_number(cls):
        cls.number+=1
    @staticmethod                          # 静态方法
    def a():
        print(111)

class Cat(Animal):
    def __init__(self,name,age,color,secret,b):
        super().__init__(name,age,color,b)
        self.__secret=secret               # 私有实例属性
        self.change_number()
    def make_sound(self):                  # 重新定义的实例方法
        print("喵喵喵")
    def introduce(self):                   # 重新定义的实例方法
        print(f"我的名字是小猫{self.name}")
    def tell_the_secret(self):             # 新的实例方法
        print(f"告诉大家，我的秘密是{self.__secret}")
    def print_b(self):
        print(self.__b)

class Dog(Animal):
    def __init__(self,name,age,color,secret,b):
        super().__init__(name,age,color,b)
        self.change_number()               # 私有实例属性
        self.__secret=secret
    def make_sound(self):                  # 重新定义的实例方法
        print("汪汪汪")
    def introduce(self):                   # 重新定义的实例方法
        print(f"我的名字是小狗{self.name}")
    def tell_the_secret(self):             # 新的实例方法
        print(f"告诉大家，我的秘密是{self.__secret}")

cat1=Cat("小红",6,"red","我喜欢小绿",1)
cat2=Cat("小绿",7,"green","我喜欢小红",1)
cat3=Cat("小蓝",7,"blue","我是小丑",1)
dog1=Dog("小白",8,"white","我才是小丑",1)

cat1.print_b()


type
dir()