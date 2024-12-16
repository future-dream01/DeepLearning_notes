class Animal():
    number=1                             # 类变量
    def __init__(self,name,age,color):   # 初始化函数
        self.name=name
        self.age=age
        self.color=color
    def introduce(self):
        print(f"Hi,我的名字是{self.name},wp")
    def sit(self):
        print(f"我坐下了!")
    def eat(self):
        print(f"我正在吃饭")
    def make_sound(self):
        pass
    @classmethod                         # 类方法
    def change_number(cls):
        cls.number+=1
        print(cls.number)

class Cat(Animal):
    def __init__(self,name,age,color):
        super().__init__(name,age,color)
    def make_sound(self):
        print("喵喵喵")

class Dog(Animal):
    def __init__(self,name,age,color):
        super().__init__(name,age,color)
        print(Dog.number)
    def make_sound(self):
        print("汪汪汪")

cat1=Cat("小猫",6,"red")
cat2=Cat("小猫",7,"red")

cat1.change_number()
cat2.change_number()
