class A:
    pass

class B(type):
    pass

class C(metaclass=B):
    pass


print(C.__class__)