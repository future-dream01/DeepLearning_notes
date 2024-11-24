import math,random
import matplotlib.pyplot as plt
import torch
def draw_before(data):
    sum1=sum(data)
    len1=len(data)
    av=sum1/len1
    fc=0
    for i in range(0,len1):
        fc=fc+(data[i]-av)**2
    fc=fc/len1
    y=[]
    x=[]
    for i in range(1,3001):
        x.append(i)

    for i in range(0,len1):
        y.append((data[i]-av)/math.sqrt(fc))
    plt.plot(x,y,label='1',color='r')
    plt.plot(x,data,label='归一化前分布',color='g')
    plt.title('归一化效果图')
    plt.xlabel('序号')
    plt.ylabel('数据点')
    plt.legend()
    plt.show()

if __name__=="__main__":
    draw_before([random.randint(0,100) for _ in range(3000)])