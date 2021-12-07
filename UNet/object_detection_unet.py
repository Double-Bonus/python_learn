import os, fnmatch
from glob import glob
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from tensorflow import keras # ???


from keras.models import Model, load_model
from keras.layers import Input
from keras.metrics import Metric
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
from keras.preprocessing import image

# tf.config.run_functions_eagerly(True)


#-------------Function that computes intersection unit ------------------
@tf.function
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, 2)
        
        print("Labas     + : ")
        print(tf.inside_function())

        sess = tf.compat.v1.keras.backend.get_session()
        # sess.run(tf.local_variables_initializer())

        
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score) # reikia tensor cia!!!!
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[Lambda(mean_iou)])
    model.summary()
    return model


trainingflag = True

BATCH_SIZE = 10 # the higher the better
IMG_WIDTH = 128 # for faster computing 
IMG_HEIGHT = 128 # for faster computing
IMG_CHANNELS = 3
TRAIN_PATH = 'data\\stage1_train\\*'
TEST_PATH = 'data\\stage1_test\\*'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42

train_ids = glob(TRAIN_PATH)
test_ids = glob(TEST_PATH)
np.random.seed(10)

if trainingflag == True:
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    # for i in range(0,len(train_ids)):
    for i in range(0, 1): # test
        img_name = os.path.basename(os.path.normpath(train_ids[i]))
        path = train_ids[i] + '\\images\\' + img_name + '.png'
        img = imread(path)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[i] = img

        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        listofmask = fnmatch.filter(os.listdir(train_ids[i] + '\\masks\\'), '*.png')
        for j in range(0,len(listofmask)):
            mask_ = imread(train_ids[i] + '\\masks\\' + listofmask[j])
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[i] = mask
        print("i:{}/{} training data".format(i,len(train_ids)))

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for i in range(0,len(test_ids)):
    img_name = os.path.basename(os.path.normpath(train_ids[i]))
    path = train_ids[i] + '\\images\\' + img_name + '.png'
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[i] = img
    print("i:{}/{} testing data".format(i,len(train_ids)))

print('Done!')

if trainingflag == True:

    #--------------------- Data augmentation ----------------------------
    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')
    mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together

    image_datagen.fit(X_train[:int(X_train.shape[0]*0.9)], augment=True, seed=seed)
    mask_datagen.fit(Y_train[:int(Y_train.shape[0]*0.9)], augment=True, seed=seed)

    x=image_datagen.flow(X_train[:int(X_train.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
    y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*0.9)],batch_size=BATCH_SIZE,shuffle=True, seed=seed)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(X_train[int(X_train.shape[0]*0.9):], augment=True, seed=seed)
    mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*0.9):], augment=True, seed=seed)

    x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
    y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*0.9):],batch_size=BATCH_SIZE,shuffle=True, seed=seed)
    #--------------------- Data augmentation ----------------------------

    #--------------------- Create generator -----------------------------
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)
    #--------------------- Create generator -----------------------------

    #----------- Create Unet model ------------------------
    model = get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    #----------- Create Unet model ------------------------

    earlystopper = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint('..\\models\\model-nucleus.h5', verbose=1, save_best_only=True)
    results = model.fit(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=250,
                                epochs=3, callbacks=[earlystopper, checkpointer], verbose=1)
    
    
else:

    model = load_model('./models/model-nucleus.h5', custom_objects={'mean_iou': mean_iou})
    preds_test = model.predict(X_test, verbose=1)
    # Threshold predictions
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    print(preds_test.shape)

    plt.imshow(preds_test[1],cmap='gray')
    plt.show()

    # Create list of upsampled test masks
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))


print('ok')