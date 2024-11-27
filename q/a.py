from sympy import symbols,Eq,solve
import math

plist=[320083,304292,285660,263307,243037,218087,199698,158401]
flist=[]
lamda=[0.66470534]
for i in range(0,len(plist)-1):
    lamda2=symbols('lamda2',positive=True)
    equation1=Eq(plist[i+1]/plist[i],(lamda[i]/lamda2)*((2.4-0.4*lamda2**2)/(2.4-0.4*lamda[i]**2)))
    lamda2=solve(equation1,lamda2)[0]
    f=symbols('f',positive=True)
    equation2=Eq(4*f*(0.18/0.009),(2.4/2.8)*((1/(lamda[i]**2))-(1/(lamda2**2))+math.log((lamda[i]**2)/(lamda2**2))))
    out=solve(equation2,f)
    lamda.append(lamda2)
    flist.append(out)
print("摩擦系数列表为：",flist)