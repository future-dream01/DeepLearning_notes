import math
import numpy as np
import matplotlib.pyplot as plt
Nu=[5.895,6.257,6.992,9.987,13.498,11.220,19.410,8.188,13.1]
Gr=[15630,51000,34400,138000,194000,491000,891000,305000,4263000]
lnNu=[]
lnGr=[]
sub_GrNu=0
sub_Gr=0
sub_Nu=0
sub_Gr2=0

for i in range(0,len(Nu)):
    lnNu.append(math.log(Nu[i]))
    lnGr.append(math.log(Gr[i]))

for i in range(0,len(lnGr)):
    sub_GrNu=sub_GrNu+(lnGr[i]*lnNu[i])
    sub_Gr=sub_Gr+lnGr[i]
    sub_Gr2=sub_Gr2+(lnGr[i]**2)
for i in range(0,len(lnNu)):
    sub_Nu=sub_Nu+lnNu[i]

k=((sub_GrNu/9)-(sub_Gr/9)*(sub_Nu/9))/((sub_Gr2/9)-(sub_Gr/9)**2)
b=(sub_Nu/9)-(sub_Gr/9)*k
print(f"n={k} \n c={(math.e)**(b/k)}")
print (k,b)
x=np.linspace(8,math.log(4300000),500)
y=k*x+b
plt.plot(x,y,label="fitting_function",color="red",linestyle='-')
plt.scatter(lnGr,lnNu,color="blue",marker='o')
plt.title("lnNu-lnGr fitting map")
plt.xlabel("lnGr")
plt.ylabel("lnNu")
plt.legend()
plt.savefig('3.jpg')
plt.show()
