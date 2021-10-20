from tensorflow import keras
#from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential, load_model
from numpy.core.numeric import True_
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
 
# from tensorflow import keras
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import Sequential, load_model
 
def getModel(numberOffClasses = 3):
    model = Sequential()
    model.add(Dense(units=16,activation='relu'))
    model.add(Dense(units=8,activation='relu'))
    model.add(Dense(units=4,activation='relu'))
    model.add(Dense(units=numberOffClasses, activation='softmax'))
    #'binary_crossentropy' - viena klase, 'sigmoid' 'tangent'
    # 'categorical_crossentropy' - daugiaklasis klasifikavimas, 'softmax'
    # 'mse' - reiksmes prognozavimui, 'sigmoid', 'pureline' 
    #model.compile(optimizer=keras.optimizers.Adam(),loss='binary_crossentropy',metrics=['accuracy']) 
    #model.compile(optimizer=keras.optimizers.rmsprop(),loss='categorical_crossentropy',metrics=['accuracy']) 
    model.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy']) 
    #model.compile(optimizer=keras.optimizers.rmsprop(),loss=keras.losses.categorical_crossentropy,metrics=['accuracy']) 
    return model
 
number_of_samples = 100 
data1 = np.random.randn(number_of_samples,2)
rows1, cols1 = data1.shape
y1 = np.zeros((rows1,1),dtype=np.float)
data2 = np.random.randn(number_of_samples,2)
rows2, cols2 = data2.shape
y2 = np.ones((rows2,1), dtype=np.float)
data3 = np.random.randn(number_of_samples,2)
rows3, cols3 = data3.shape
y3 = np.ones((rows3,1), dtype=np.float)*2

data1[:,0] = data1[:,0] + 5
data1[:,1] = data1[:,1] + 3

data2[:,0] = data2[:,0] + 8 
data2[:,1] = data2[:,1] + 1.5

data3[:,0] = data3[:,0] + 10 
data3[:,1] = data3[:,1] + 5
 
trainData = np.vstack((data1,data2,data3))
trainY = np.vstack((y1,y2,y3))
trainY = to_categorical(trainY,num_classes=3) 
print(trainData.shape)
 
train = False
if train == True:
    model = getModel()
    model.fit(x=trainData, y=trainY, epochs=200, batch_size=32)
    model.summary()
    model.save('modelis_multi.h5')
else:
    model = load_model('modelis_multi.h5')
 
    minX = np.min(trainData[:,0])
    maxX = np.max(trainData[:,0])
    minY = np.min(trainData[:,1])
    maxY = np.max(trainData[:,1])
 
    stepsX = np.linspace(minX, maxX, 100)
    stepsY = np.linspace(minY, maxY, 100)

    xx, yy = np.meshgrid(stepsX, stepsY)

    predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
    predictions = np.where(predictions>0.5, predictions, 0)
    predictions = np.where(predictions<0.5, predictions, 0)
    z = predictions.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap='Paired',alpha = 0.5)
    
 
 
    print('prediction: {}'.format(predictions))
 
plt.plot(data1[:,0],data1[:,1],'ro')
plt.plot(data2[:,0],data2[:,1],'bo')
plt.plot(data3[:,0],data3[:,1],'go')
plt.grid(b=True)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()