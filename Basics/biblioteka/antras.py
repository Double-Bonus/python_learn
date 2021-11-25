
def kazkokie_veiksmai():
    print('Hello!')

    a = list() #tuscias!

    for i in range(0,10):
        a.append(i)

    print(a)
    a.remove(5) #pagal reiksme
    print(a)
    a.pop(2) # pagal indeksa pasalina

    for i in range(0, len(a)):
        print(a[i])

    print('-------------------')
    print(a[0]) # pirmasis narys
    print(a[-1]) # paskutins narys
    print(a[:]) # visi nariai
    print(a[2:5]) 
    print(a[2:])
    print(a[:5])
    print(a[0:-1:2]) #zingsneliai kas du

    a[-1] = 125
    print('-------------------')
    for i in range(0, len(a)):
        val = a[i]
        if not (val > 5 and val < 15):
            print('ok: {} skaicius antras: {}'.format(val, val*val))



    print('----------while-------')
    ttt = 0
    while ttt < 10:
        ttt += 1


    while True:
        ttt += 1
        if ttt > 25:
            print('nu viskas')
            break
    print(ttt)
    print('-------end while-------')

    str = 'Kazkos ilgas bacis tekstas'

    a_cnt = 0
    for i in range(0, len(str)):
        #print(str[i])
        if 'a' == str[i]:
            print(str[i])
            a_cnt += 1


    print('a sudaro: {} procentu '.format(100 * a_cnt/len(str)))
    # kiek procestu teksto sudaro a raides

def fun1():
    print('Labas!')

def fun2(a = -1 ,b =-20):
    print(a*b)

def fun3(a=10, b=20):
    c=a*b**(b/5)
    d = c/1000
    return c, d

konstanta = 3.14
    

if 0: #uzkomentavome 
    fun1()
    fun2(a=10, b=5)
    fun2()

    c, g = fun3()
    print(c, g)