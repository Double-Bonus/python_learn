import numpy as np
import mnist
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D,  Flatten
from keras.models import Sequential



x_train, y_train, x_test, y_test = mnist.load()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.0
x_test /= 255.0

print(x_train.shape)
print(np.min(x_train), np.max(x_train))
print(y_train)

number_of_samples = 15500
img = np.reshape(x_train[number_of_samples, :], (28,28))

number_of_outputs = 10
train_labels = keras.utils.to_categorical(y_train, num_classes=number_of_outputs)
test_labels = keras.utils.to_categorical(y_test, num_classes=number_of_outputs)


print(train_labels)


plt.imshow(img, cmap='gray')
plt.show()


# 1 - MLP training
# 0 - MLP testing
# 2 - CNN training
Mode = 1

if Mode == 1:
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=(784,)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(number_of_outputs, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # mokymas
    hist = model.fit(x_train, train_labels, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, test_labels))
    
    model.save('ann_mnist.h5')
    
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
elif Mode == 0:
    #testavimas
    model = keras.models.load_model('ann_mnist.h5')
    model.summary()
    
    number_of_samples = 1500
    img = np.reshape(x_test[number_of_samples, :], (28,28))
    
    xnew = np.array([x_test[number_of_samples, :]])
    # predicted_class = model.predict_classes(xnew)
    
    # predicted_class=np.argmax(xnew, axis=1)
    
    scores = model.predict(xnew)
    
    print("Predicted class: {}".format(np.argmax(scores)), 'scores', np.max(scores))
    
    # y_test

    total_miss = 0

    for i in range(y_test.shape[0]):
        xnew = np.array([x_test[i, :]])
        scores = model.predict(xnew)
        if(np.argmax(scores) != y_test[i]):
            total_miss += 1

            
    
    print((y_test.shape[0] - total_miss) / y_test.shape[0])
        
        

    
    plt.imshow(img, cmap='gray')
    plt.show()
elif Mode == 2:
    
    
    x_train_c = x_train.reshape(60000, 28, 28, 1)
    x_test_c = x_test.reshape(10000, 28, 28, 1)
    
    
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=3, activation='relu', input_shape = (28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=number_of_outputs, activation='relu'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # mokymas
    hist = model.fit(x_train_c, train_labels, batch_size=32, epochs=10, verbose=1, validation_data=(x_test_c, test_labels))
    
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
    
## todo
elif Mode == 3: 
        #testavimas
    model = keras.models.load_model('cnn_mnist.h5')
    model.summary()
    
    number_of_samples = 1500
    img = np.reshape(x_test[number_of_samples, :], (28,28))
    # xnew
    
    xnew = np.array([x_test[number_of_samples, :]])
    # predicted_class = model.predict_classes(xnew)
    
    # predicted_class=np.argmax(xnew, axis=1)
    
    scores = model.predict(xnew)
    
    print("Predicted class: {}".format(np.argmax(scores)), 'scores', np.max(scores))
    
    # y_test

    total_miss = 0

    for i in range(y_test.shape[0]):
        xnew = np.array([x_test[i, :]])
        scores = model.predict(xnew)
        if(np.argmax(scores) != y_test[i]):
            total_miss += 1

            
    
    print((y_test.shape[0] - total_miss) / y_test.shape[0])
        
        

    
    plt.imshow(img, cmap='gray')
    plt.show()    
    
    
print("Done")