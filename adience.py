import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from sklearn.utils import shuffle
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def create_img_dataset(X):
    height = X[0][0].shape[0]
    width = X[0][0].shape[1]
    num = X.shape[0]
    res = np.ones((num, height, width))
    for i in range(num):
        res[i,:,:] = X[i,0]
    
    return res[:, :, :, np.newaxis]

def create_label_dataset(y):
    num_classes = len(np.unique(y))
    y_encoded = np.zeros((y.shape[0],num_classes))
    for idx, l in enumerate(y):
        y_encoded[idx,l] = 1
    return y_encoded


if __name__ == "__main__":
    bs = 128
    epochs = 100
    num_classes = 8
    lr = 0.01
    SHUFFLE_BUFFER_SIZE = 100


    DATA_DIR = './data/adience_dataset_preprocessed'
    MODEL_DIR = './model/adience'
    LOG_DIR = './log/adience'

    TRAIN_DIR = DATA_DIR + '/train'
    VAL_DIR = DATA_DIR + '/val'
    TEST_DIR = DATA_DIR + '/test'

    IMG_TRAIN_DIR = TRAIN_DIR + '/img_array_train.pkl'
    AGE_TRAIN_DIR = TRAIN_DIR + '/age_groups_train.pkl'
    GENDER_TRAIN_DIR = TRAIN_DIR + '/gender_train.pkl'

    IMG_VAL_DIR = VAL_DIR + '/img_array_val.pkl'
    AGE_VAL_DIR = VAL_DIR + '/age_groups_val.pkl'
    GENDER_VAL_DIR = VAL_DIR + '/gender_val.pkl'

    IMG_TEST_DIR = TEST_DIR + '/img_array_test.pkl'
    AGE_TEST_DIR = TEST_DIR + '/age_groups_test.pkl'
    GENDER_TEST_DIR = TEST_DIR + '/gender_test.pkl'


    img_train = pd.read_pickle(IMG_TRAIN_DIR).to_numpy()
    age_train = pd.read_pickle(AGE_TRAIN_DIR).to_numpy()
    gender_train = pd.read_pickle(GENDER_TRAIN_DIR).to_numpy()

    img_val = pd.read_pickle(IMG_VAL_DIR).to_numpy()
    age_val = pd.read_pickle(AGE_VAL_DIR).to_numpy()
    gender_val = pd.read_pickle(GENDER_VAL_DIR).to_numpy()

    img_test = pd.read_pickle(IMG_TEST_DIR).to_numpy()
    age_test = pd.read_pickle(AGE_TEST_DIR).to_numpy()
    gender_test = pd.read_pickle(GENDER_TEST_DIR).to_numpy()

    img_train = create_img_dataset(img_train)
    img_val = create_img_dataset(img_val)
    img_test = create_img_dataset(img_test)

    age_train = create_label_dataset(age_train)
    age_val = create_label_dataset(age_val)
    age_test = create_label_dataset(age_test)

    with tf.device('GPU:0'):
        # datagen = ImageDataGenerator(
        # rescale=1 / 255.0,
        # rotation_range = 10,
        # zoom_range = 0.1,
        # horizontal_flip = True
        # )

        # train_generator = datagen.flow(img_train, age_train, batch_size = bs, subset = 'training', shuffle=True, seed = 43)
        # # test_datagen = ImageDataGenerator(rescale=1 / 255.0)

        # valid_generator = datagen.flow(img_val, age_val, batch_size = bs, subset = 'Validation', shuffle=True, seed=43)

        train_dataset = tf.data.Dataset.from_tensor_slices((img_train, age_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((img_val, age_val))

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(bs)
        val_dataset = val_dataset.batch(bs)
        # for x, y in train_dataset:
        #     print(x.shape)
        #     print(y.shape)
        #     break


        model = Sequential([
        layers.Conv2D(256, kernel_size=[3,3] ,padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, kernel_size=3 ,padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size = (2,2)),
        layers.Dropout(0.4),

        layers.Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size = (2,2)),
        layers.Dropout(0.4),


        layers.Conv2D(192, padding='same', kernel_size = 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size = (2,2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation = 'softmax', activity_regularizer=tf.keras.regularizers.L2(0.01))
        ])


        # model.summary()
        training_weights='./weights'  #这里是保存每次训练权重的  如果需要自己取消注释
        checkpoint_period = ModelCheckpoint(training_weights + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1) #学习率衰减
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1) # val_loss 不下降时 停止训练 防止过拟合
        tensorboard = TensorBoard(log_dir=LOG_DIR)  #训练日志
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics='acc',optimizer=optimizer)
        history = model.fit(train_dataset,validation_data=val_dataset, epochs=epochs,
                            callbacks=[tensorboard, early_stopping,checkpoint_period,reduce_lr])
        model.evaluate(val_dataset,verbose=1)
        model.save(MODEL_DIR+'base_model.h5')
    print(history.history.keys())
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    lr = history.history['lr']
    print("Learning Rate History: ", lr)
    epochs_range = [i for i in range(1,len(lr)+1)]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
