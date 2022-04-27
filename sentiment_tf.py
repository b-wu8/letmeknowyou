import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

if __name__ == "__main__":
    batch_size = 512
    img_height = 48
    img_width = 48
    epochs = 10

    print(tf.__version__)
    print(tf.config.list_physical_devices())

    # Loading data from data folder /data/sentiment
    DATA_DIR = "data/sentiment/"
    # LABEL_DIR = "data/sentiment/fer2013.csv"

    TRAIN_DIR = DATA_DIR + "train/"
    VAL_DIR = DATA_DIR + "val/"
    MODEL_DIR = "models/"

    tf.debugging.set_log_device_placement(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 9

    model = Sequential([
    layers.Conv2D(64, kernel_size=5 ,padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
    layers.AveragePooling2D(),
    layers.Conv2D(256, padding='same', kernel_size = 2, activation='relu'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(7, activation='linear'),
    layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # print(model.summary())

    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )
