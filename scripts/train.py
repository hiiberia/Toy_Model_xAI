import os
import tensorflow as tf
import datetime
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import cv2
import numpy as np

for device in tf.config.list_physical_devices():
    print(f"{device.name}")
    
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, BatchNormalization,  GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard

import warnings
warnings.filterwarnings("ignore")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

train_path = "../eo-toy-model-poc/data/train"
valid_path = "../eo-toy-model-poc/data/valid"
test_path = "../eo-toy-model-poc/data/test"

im_size = 224
image_resize = (im_size, im_size, 3) 
batch_size_training = 100
batch_size_validation = 100
batch_size_test = 100
num_classes = 2
class_names_valid = ['no wildfire', 'wildfire']

data_generator = ImageDataGenerator(dtype='float32', rescale= 1./255.)
train_generator = data_generator.flow_from_directory(train_path,
                                                   batch_size = batch_size_training,
                                                   target_size = (im_size, im_size),
                                                   class_mode = 'categorical')

valid_generator = data_generator.flow_from_directory(valid_path,
                                                   batch_size = batch_size_validation,
                                                   target_size = (im_size, im_size),
                                                   class_mode = 'categorical')

class_mapping = train_generator.class_indices

class_names = list(train_generator.class_indices.keys())
print("Class names :", class_names_valid)

labels_train = train_generator.classes
unique_labels_train, label_counts_train = np.unique(labels_train, return_counts=True)

print("Number of unique labels in train data:", len(unique_labels_train))
for label, count in zip(unique_labels_train, label_counts_train):
    print("Label:", class_names[label], "- Count:", count)
    
labels_valid = valid_generator.classes
unique_labels_valid, label_counts_valid = np.unique(labels_valid, return_counts=True)

print("Number of unique labels in valid data:", len(unique_labels_valid))
for label, count in zip(unique_labels_valid, label_counts_valid):
    print("Label:", class_names_valid[label], "- Count:", count)
    
def base_model(input_shape, repetitions): 
  
  input_ = tf.keras.layers.Input(shape=input_shape, name='input')
  x = input_
  
  for i in range(repetitions):
    n_filters = 2**(4 + i)
    x = Conv2D(n_filters, 3, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)

  return x, input_

def final_model(input_shape, repetitions):
    
    x, input_ = base_model(input_shape, repetitions)

    x = Conv2D(64, 3, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    class_out = Dense(num_classes, activation='softmax', name='class_out')(x)

    model = Model(inputs=input_, outputs=class_out)

    print(model.summary())
    return model

model = final_model(image_resize, 4)
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
# checkpoint = tf.keras.callbacks.ModelCheckpoint('../eo-toy-model-poc/saved_models/custom_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [
    # checkpoint,
    tensorboard_callback
    ]

num_epochs = 2
steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(valid_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=valid_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
    callbacks=[callbacks_list],
)

# model.save('../eo-toy-model-poc/saved_models')
