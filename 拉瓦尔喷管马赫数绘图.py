# 拉瓦尔喷管马赫数计算和绘图
from sympy import symbols,Eq,nsolve
import matplotlib.pyplot as pl

pointlist=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]
env_list=[11154,20845,51390,81516]      # 四个出口背压
list1=[0.9956,0.9890,0.9509,0.6937,0.3617,0.3603,0.3295,0.2781,0.2463,0.2164,0.1914,0.1702,0.1515,0.1395,0.1285,0.1175,0.1098,0.1008,0.1044,0.1123,0.1159,0.1255,0.1212,0.1124,0.1079,0.1186]# 依次为四个喷管的分布压力
list2=[0.9950,0.9905,0.9627,0.7035,0.3615,0.3614,0.3273,0.2777,0.2450,0.2175,0.1922,0.1682,0.1534,0.1489,0.1572,0.1647,0.1783,0.1690,0.1706,0.2385,0.2208,0.1982,0.2083,0.2264,0.2013,0.2090]
list3=[0.9956,0.9905,0.9528,0.7034,0.3620,0.3616,0.3260,0.2871,0.2859,0.2981,0.3364,0.3403,0.3664,0.3888,0.4216,0.4474,0.4718,0.4900,0.5130,0.5271,0.5298,0.5293,0.5269,0.5282,0.5273,0.5266]
list4=[0.9966,0.9904,0.9578,0.7414,0.5879,0.6176,0.6565,0.6871,0.7190,0.7469,0.7655,0.7814,0.7936,0.8069,0.8153,0.8242,0.8315,0.8372,0.8426,0.8471,0.8457,0.8453,0.8449,0.8463,0.8448,0.8442]
MA1=[]
MA2=[]
MA3=[]
MA4=[]
ma=symbols('ma',positive=True)
for value in list1:
    e = Eq(value, (1 + 0.2 * ma**2)**(1.4 / (-0.4)))
    out = nsolve(e, ma, 1)  
    MA1.append(out)
for value in list2:
    e = Eq(value, (1 + 0.2 * ma**2)**(1.4 / (-0.4)))
    out = nsolve(e, ma, 1)  
    MA2.append(out)
for value in list3:
    e = Eq(value, (1 + 0.2 * ma**2)**(1.4 / (-0.4)))
    out = nsolve(e, ma, 1) 
    MA3.append(out)
for value in list4:
    e = Eq(value, (1 + 0.2 * ma**2)**(1.4 / (-0.4)))
    out = nsolve(e, ma, 1) 
    MA4.append(out)

pl.plot(pointlist,MA1,label=f"{env_list[0]} Pa",color='red',linestyle='-',marker='o')
pl.plot(pointlist,MA2,label=f"{env_list[1]} Pa",color='blue',linestyle='-',marker='s')
pl.plot(pointlist,MA3,label=f"{env_list[2]} Pa",color='green',linestyle='-',marker='*')
pl.plot(pointlist,MA4,label=f"{env_list[3]} Pa",color='orange',linestyle='-',marker='v')
pl.title("Mach number distribution")
pl.xlabel("point")
pl.ylabel("MA")
pl.legend()
pl.savefig("1.jpg")

pl.clf()

pl.plot(pointlist,list1,label=f"{env_list[0]} Pa",color='red',linestyle='-',marker='o')
pl.plot(pointlist,list2,label=f"{env_list[1]} Pa",color='blue',linestyle='-',marker='s')
pl.plot(pointlist,list3,label=f"{env_list[2]} Pa",color='green',linestyle='-',marker='*')
pl.plot(pointlist,list4,label=f"{env_list[3]} Pa",color='orange',linestyle='-',marker='v')
pl.title("Relative distributed pressure")
pl.xlabel("point")
pl.ylabel("pressure")
pl.legend()
pl.savefig("2.jpg")
pl.show()

