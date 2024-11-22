import matplotlib.pyplot as plt
import numpy as np
def draw(list1,list2):
    plt.subplot(1,2,1)
    plt.plot(list1,label='1',color='r',linestyle='-',marker='o')
    plt.title('111')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(list2,list1,label='1',color='g',linestyle='-',marker='o')
    plt.title('222')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    np.random.seed(42)
    x = np.random.randn(30)
    y = np.random.randn(30)
    draw(x,y)
