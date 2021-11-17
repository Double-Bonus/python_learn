from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential, load_model
 
# from tensorflow import keras
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential, load_model
 
def getModel():
    model = Sequential()
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=4,activation='relu'))
    model.add(Dense(units=1,activation='sigmoid'))
    #'binary_crossentropy' - viena klase, 'sigmoid' 'tangent'
    # 'categorical_crossentropy' - daugiaklasis klasifikavimas, 'softmax'
    # 'mse' - reiksmes prognozavimui, 'sigmoid', 'pureline' 
    model.compile(optimizer=keras.optimizers.Adam(),loss='binary_crossentropy',metrics=['accuracy']) 
    return model
 
data1 = np.random.randn(100,2)
rows1, cols1 = data1.shape
y1 = np.zeros((rows1,1),dtype=np.float)
data2 = np.random.randn(100,2)
rows2, cols2 = data2.shape
y2 = np.ones((rows2,1), dtype=np.float)
 
data1[:,0] = data1[:,0] + 5
data1[:,1] = data1[:,1] + 3
data2[:,0] = data2[:,0] + 8 
data2[:,1] = data2[:,1] + 1.5
 
trainData = np.vstack((data1,data2))
trainY = np.vstack((y1,y2))
 
print(trainData.shape)
 
train = False
if train == True:
    model = getModel()
    model.fit(x=trainData,y=trainY,epochs=100,batch_size=32)
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
            xTest = np.array([[stepsX[x],stepsY[y]]], dtype=np.float)
            prediction = model.predict(xTest)
            if prediction[0][0] > 0.5:
                color = 'm'
            else:
                color = 'y'
            #plt.scatter(xTest[0,0],xTest[0,1],'gs',ms=20,markerfacecolor=color,markeredgecolor=color)
            plt.scatter(xTest[0,0],xTest[0,1],marker='s',color=color,s=10,alpha=0.5)
            print('x:{},y:{}'.format(x,y))
 
    print('prediction: {}'.format(prediction))
 
plt.plot(data1[:,0],data1[:,1],'ro')
plt.plot(data2[:,0],data2[:,1],'bo')
plt.grid(b=True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()