import numpy as np
import matplotlib.pyplot as plt

def fun1():
    x = np.array([1,2,3,4])
    y = x**2
    plt.plot(x,y)
    # plt.grid(b=True)
    plt.grid(color='magenta', ls='--', lw = 4)
    plt.title('Grafiko pavadinimas')
    plt.xlabel('greitis, [m/s]')
    plt.ylabel('galia, [W]')
    plt.show()



def fun2():
    t = np.linspace(0.,5.,100)
    y1 = np.sin(2*np.pi*t)
    y2 = 10 * np.sin(2*np.pi*t)

    plt.subplot(2,2,1), plt.plot(t,y1,'r--'), plt.grid(b=True), plt.title('Sinusas')
    plt.subplot(222), plt.plot(t,y2,'b--'), plt.grid(b=True), plt.title('Cosinusas')
    plt.subplot(223), plt.plot(t,y1,'r--'), plt.grid(b=True), plt.title('Sinusas')
    plt.subplot(2,2,4), plt.plot(t,y2,'b--'), plt.grid(b=True), plt.title('Cosinusas')
    plt.show()


def fun3():
    t = np.linspace(0.,5.,100)
    y1 = np.sin(2*np.pi*t)
    y2 = 10 * np.sin(2*np.pi*t)**3

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(t,y1,'r--',lw=2)
    ax1.plot(t,y2,'b-.',lw=2)
    ax1.set_xlabel('x asis')
    ax1.set_ylabel('Y1 asis')
    ax2.set_ylabel('Y2 asis')
    plt.show()


def fun4():
    data = np.random.randint(0,200, size=(200,4))
    x = data[:,0]
    y = data[:,1]
    colors = data[:,2]
    sizes = data[:,3]*3
    data = np.random.randint(0,200, size=(100,4))
    plt.scatter(x,y, c = colors,cmap='autumn', s=sizes, ec='k')
    plt.scatter(data[:,0],data[:,1], c = data[:,2],cmap='winter', s=data[:,3], ec='k', alpha=0.5)
    plt.show()


def fun5():
    reiksmes = [5,55,15,25]
    labels = ['A', 'Jonas' ,'B', "c"]
    plt.pie(reiksmes,labels= labels,startangle=45, explode=[0.5, 0, 0, 0,])
    plt.show()

# fun2()
# fun3()
# fun4()
fun5()









  