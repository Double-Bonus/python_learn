import numpy as np
import matplotlib.pyplot as plt
# import keras as kr
import keras
#from tensorflow import keras
from keras.layers import Dense
from numpy.core.defchararray import mod
from tensorflow.keras.models import Sequential, load_model
from keras import optimizers

def getModel():
    model = Sequential()
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    # 'binary_crossentropy' - viena klase, 'sigmoid', 'tangent'
    #
    # 'mse' - reiksmes prognozavimui
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=trainData,y=trainY,epochs=100,batch_size=32)

    # model.build(trainData.shape)
    return model



# from keras.models import Sequential
print('Labas!')

data1 = np.random.randn(100,2)
rows1, cols1 = data1.shape
y1 = np.zeros((rows1, 0), dtype=np.float)
data2 = np.random.randn(100,2)
rows2, cols2 = data2.shape
y2 = np.ones((rows2, 0), dtype=np.float)


data1[:,0] = data1[:,0] + 5
data1[:,1] = data1[:,1] + 3

data2[:,0] = data2[:,0] + 8
data2[:,1] = data2[:,1] + 1.5

trainData = np.vstack((data1,data2))
trainY = np.vstack((y1,y2))
print(trainData.shape)

train = True
if train == True:
    model = getModel()
    model.fit(x=trainData,y=trainY,epochs=100)
    model.summary()
    model.save('modelis.h5')
else:
    model = load_model('modelis.h5')

    minX = np.min(trainData[:,0])
    maxX = np.max(trainData[:,0])
    minY = np.min(trainData[:,1])
    maxY = np.max(trainData[:,1])

    stepsX = np.linspace(minX,maxX,100)
    stepsY = np.linspace(minY,maxY,100)


    for x in range(0,len(stepsX)):
        for y in range(0, len(stepsY)):
            xTest = np.array([[stepsX[x], stepsY[y]]], dtype=np.float)
            prediction = model.predict(xTest)
            if prediction[0][0] > 0.5:
                color = 'b'
            else:
                color = 'r'
            plt.plot(xTest[0,0], xTest[0,1], 'gs', ms=20,)
        

    xTest = np.array([[5,5]],dtype=np.float)
    prediction = model.predict(xTest)
    if prediction[0][0] > 0.5:
        color = 'b'
    else:
        color = 'r' 
        
    print('prediction: {}')








plt.plot(data1[:,0],data1[:,1], 'ro')
plt.plot(data2[:,0],data2[:,1], 'bo')

if train == False:
    plt.plot(xTest[0][0],xTest[0][1], 'g*')


plt.xlabel('X')
plt.ylabel('Y')
plt.grid(b=True)
plt.show()
