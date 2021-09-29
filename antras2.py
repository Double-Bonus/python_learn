from numpy.core.fromnumeric import size
from numpy.random.mtrand import random_integers
import biblioteka.antras
from biblioteka.antras import fun3 as f3
from biblioteka.antras import konstanta as kns
import numpy as np

#antras.fun2()
c, d = f3(a= 10, b = 2)

print(c, d)
print(kns)
kns += 1
print(kns)

print('--------------darbas su numpy --------------')
a = np.array([1,2,3], dtype=np.int) #1D vektoroius
b = np.array([[1,2,6], [8,6,4]], dtype=float) #2D masyvas
print(a)
print(b)
print(c)

print(np.zeros((3,3)))
print(np.ones((3,3)))

print(np.full((3,3),-100))
print(np.arange(10,100,5))
print(np.linspace(0,1,10))
print(np.random.random((3,3)))
print(np.random.randint(0,100,size=(3,3)))

a = np.random.randint(0,100,size=(3,5))

# np.save('./biblioteka/a.npy',a)
# a2 = np.load('./biblioteka/a.npy')
# print('a2',a2)
# np.savetxt('./biblioteka/a.txt',a,fmt='%.2d')
# a3 = np.loadtxt('./biblioteka/a.txt')
# print('a3', a3)

print('--------------')
rows, cols = a.shape
print(rows, cols)
# rows, cols, depht = a.shape # Jeigu trimatis
# print(rows, cols, depht)
print(a.ndim) #dimesiju kiekis
print(a.size)
print(a.dtype)
b = a.astype(np.float)
print(b.dtype)

b[1,2] = 999
print(a,'\n',b)
print(b[:,1:])

print(a+b) #paelemete sudetis
print(a-b)
print(a*b)
print(a/b)
print(np.add(a,b))
print(np.multiply(a,b))
print(np.sin(a))
print(np.cos(a))
print(np.exp(a/b))
print(np.sqrt(a))
print(a**0.5)

print('------Palginis--------')
print(a>=b)


c = np.random.randint(0,100,size=(5,5))
#kiek procentu skaiciu yra didesni nei 50 iskaiciuojat 50
print(c)
bool_arr =  (c >= 50)
print(bool_arr)
count = np.sum(bool_arr)
print(count)
prct = (100 * count) / np.size(bool_arr)
print('Didesniu nei 50 yra ', prct, ' procentu')


rows, cols = c.shape
cc = 0
for i in range(0,rows):
    for j in range(0,cols):
        if c[i,j] >=50 and c[i,j] <= 70:
            cc += 1
porct = 100*cc / c.size
print(porct)

print(np.min(c))
print(np.max(c))
print(np.min(c,axis=1)) # tarp eiluciu
print(np.min(c,axis=0)) # tarp stulpeliu
print(np.mean(c))
print(np.median(c))
print(np.std(c))
print(np.corrcoef(c))
            
        