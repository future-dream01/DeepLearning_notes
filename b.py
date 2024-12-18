class A():
    pass


B=type('B',(),{})

a=A()
print(B.__class__)