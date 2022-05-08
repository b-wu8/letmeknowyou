from multiprocessing.dummy import active_children
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

if __name__ == "__main__":
    batch_size = 128
    img_height = 48
    img_width = 48
    epochs = 720
    lr = 1e-3
    num_classes = 7

    # print(tf.__version__)
    # print(tf.config.list_physical_devices())

    # Loading data from data folder /data/sentiment
    DATA_DIR = "data/sentiment/"
    # LABEL_DIR = "data/sentiment/fer2013.csv"

    TRAIN_DIR = DATA_DIR + "train/"
    VAL_DIR = DATA_DIR + "val/"
    MODEL_DIR = "models/"
    LOG_DIR = 'log/'

    # tf.debugging.set_log_device_placement(True)

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #             print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)
 
    # )
    # test_gen = test_datagen.flow_from_directory(
    #     directory=test_dataset_path,
    #     target_size=(48, 48),
    #     color_mode="grayscale",
    #     class_mode="categorical",
    #     batch_size=batch_size,
    #     shuffle=True,
    #     seed=42
    # )


    with tf.device('GPU:0'):
        train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range = 10,
        zoom_range = 0.1,
        horizontal_flip = True
        )

        train_generator = train_datagen.flow_from_directory(
            directory=TRAIN_DIR,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        test_datagen = ImageDataGenerator(
            rescale=1 / 255.0)
            
        valid_generator = test_datagen.flow_from_directory(
            directory=VAL_DIR,
            target_size=(48, 48),
            color_mode="grayscale",
            class_mode="categorical",
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )

        # train_ds = tf.keras.utils.image_dataset_from_directory(
        # TRAIN_DIR,
        # validation_split=0.2,
        # subset="training",
        # seed=123,
        # image_size=(img_height, img_width),
        # batch_size=batch_size)

        # val_ds = tf.keras.utils.image_dataset_from_directory(
        # VAL_DIR,
        # validation_split=0.2,
        # subset="validation",
        # seed=123,
        # image_size=(img_height, img_width),
        # batch_size=batch_size)

        # for image_batch, labels_batch in train_ds:
        #     print(image_batch.shape)
        #     print(labels_batch.shape)
        #     break

        # AUTOTUNE = tf.data.AUTOTUNE

        # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


        model = Sequential([
        layers.Conv2D(256, kernel_size=3 ,padding='same', activation='relu'),
        layers.Conv2D(512, kernel_size=3 ,padding='same', activation='relu'),
        layers.BatchNormalization(),
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

        # model = base_model.build()

        # model.summary()
        training_weights='./weights'  #这里是保存每次训练权重的  如果需要自己取消注释
        checkpoint_period = ModelCheckpoint(training_weights + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                            monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)
        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1) #学习率衰减
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1) # val_loss 不下降时 停止训练 防止过拟合
        tensorboard = TensorBoard(log_dir=LOG_DIR)  #训练日志
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss=tf.keras.losses.categorical_crossentropy, metrics='acc',optimizer=optimizer)
        history = model.fit(train_generator,validation_data=valid_generator,
                        epochs=epochs,callbacks=[tensorboard, early_stopping,checkpoint_period])
        # model.evaluate(test_gen,verbose=1)
        model.save(MODEL_DIR+'base_model.h5')
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        #         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        #         metrics=['accuracy'])

        # print(model.summary())

        # history = model.fit(
        # train_ds,
        # validation_data=val_ds,
        # epochs=epochs
        # )



        # tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     min_delta=0,
        #     patience=0,
        #     verbose=0,
        #     mode='auto',
        #     baseline=None,
        #     restore_best_weights=False
        # )



    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

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