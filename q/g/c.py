from sympy import symbols,Eq,nsolve
import math

lamda1=symbols('lamda1',positive=True)
equation1=Eq(320083/368342,(1-(0.4/2.4)*lamda1**2)**(1.4/0.4))
out=nsolve(equation1,lamda1,1)
print(out)