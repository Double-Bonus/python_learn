# Classify images to 10 classes using MLP and CNN, and compare results
#
# With MLP managed to achieve ~60 validation accuracy
# With CNN *only* ~40 validation accuracy

import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D,  Flatten
from keras.models import Sequential
from keras.layers.core import Dropout

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

print(x_train.shape)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

print(x_train.shape)
print(np.min(x_train), np.max(x_train))
print(y_train)

# This is a dataset of 50,000 32x32 color training images and 10,000 test images,
number_of_samples = 1123
img = x_train[number_of_samples]

number_of_outputs = 10  # labeled over 10 categories
train_labels = keras.utils.to_categorical(y_train, num_classes=number_of_outputs)
test_labels = keras.utils.to_categorical(y_test, num_classes=number_of_outputs)


# print(train_labels)
# plt.imshow(img)
# plt.show()


# 1 - MLP training
# 2 - CNN training
Mode = 1

    
if Mode == 2:
    x_train_c = x_train.reshape(50000, 32, 32, 3)
    x_test_c  =  x_test.reshape(10000, 32, 32, 3)
        
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', input_shape = (32,32,3)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu' ))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu' ))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=number_of_outputs, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    # mokymas
    hist = model.fit(x_train, train_labels, batch_size=128, epochs=40, verbose=1, validation_data=(x_test, test_labels))
    
    model.save('uzd_cnn.h5')
    
    plt.subplot(121)
    plt.plot(hist.history['accuracy'], 'r')
    plt.plot(hist.history['val_accuracy'], 'b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(b=True)
    # plt.show()
    
    
    plt.subplot(122)
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(b=True)
    plt.show()
    
elif Mode == 1:
    
    x_train = x_train.reshape(50000, 3072)
    x_test  =  x_test.reshape(10000, 3072)
    
    model = Sequential()
    model.add(Dense(units=1024, activation='relu', input_shape=(3072,)))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(number_of_outputs, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # mokymas
    hist = model.fit(x_train, train_labels, batch_size=64, epochs=50, verbose=1, validation_data=(x_test, test_labels))
    
    model.save('uzd_ann.h5')
    
    plt.subplot(121)
    plt.plot(hist.history['accuracy'], 'r')
    plt.plot(hist.history['val_accuracy'], 'b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(b=True)
    # plt.show()

    plt.subplot(122)
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(b=True)
    plt.show()
        

print("Done")