# 摩擦管摩擦系数计算

from sympy import symbols,Eq,solve,nsolve
import math

plist=[368342,320083,304292,285660,263307,243037,218087,199698,158401] # 压力总表
flist=[]
lamda=[]

lamda1=symbols('lamda1',positive=True)
equation1=Eq(plist[1]/plist[0],(1-(0.4/2.4)*lamda1**2)**(1.4/0.4))
out=nsolve(equation1,lamda1,1)
lamda.append(out)
for i in range(0,len(plist)-1):
    lamda2=symbols('lamda2',positive=True)
    equation1=Eq(plist[i+1]/plist[i],(lamda[i]/lamda2)*((2.4-0.4*lamda2**2)/(2.4-0.4*lamda[i]**2)))
    lamda2=nsolve(equation1,lamda2,1)
    f=symbols('f',positive=True)
    equation2=Eq(4*f*(0.18/0.009),(2.4/2.8)*((1/(lamda[i]**2))-(1/(lamda2**2))+math.log((lamda[i]**2)/(lamda2**2))))
    out=nsolve(equation2,f,1)
    lamda.append(lamda2)
    flist.append(out)
print("摩擦系数列表为：",flist)