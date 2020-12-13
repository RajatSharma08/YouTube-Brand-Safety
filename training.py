import keras
# from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL import ImageFile
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from shutil import copyfile
from model import VGG16Model,VGG16ModelDropout


print('Classes:',os.listdir('data_adult/train'))
print('No of Images per classes in training')
print(len(os.listdir('data_adult/train/Safe/')),len(os.listdir('data_adult/train/Unsafe/')))

print('No of Images per classes in validation')
print(len(os.listdir('data_adult/valid/Safe/')),len(os.listdir('data_adult/valid/Unsafe/')))
train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data_adult/train', target_size=(256, 256), 
                                      color_mode='rgb', classes=None, 
                                      class_mode='binary', batch_size=64, 
                                      shuffle=True,  follow_links=False, 
                                      subset=None, interpolation='nearest')

val_generator = val_datagen.flow_from_directory(
        'data_adult/valid',
        target_size=(256, 256),
        batch_size=64,
        class_mode='binary')


train_filenames=train_generator.filenames
val_filenames=val_generator.filenames
print('Train Size : {}\nVal Size : {}'.format(len(train_filenames),len(val_filenames)))




model = VGG16ModelDropout()


# # ---------1st Step of Training-----------------
# for layer in model.layers[:-4]:
#     layer.trainable = False

# optimiz=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(optimizer=optimiz, loss='binary_crossentropy',
#               metrics=['accuracy'
# #                        ,precision, recall,f1_score
#                       ])



# print(model.summary())
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# print('-----Training-------')
# model.fit_generator(train_generator,
#                     steps_per_epoch=235,
#                     validation_data=val_generator,
#                     validation_steps=50, 
#                     epochs=5)
# print("Saving Weights")
# model.save_weights('model_weights/vgg_stage-1.hdf5')



# # ---------2nd Step of Training-----------------

# for layer in model.layers:
#     layer.trainable = True
# for layer in model.layers[:-8]:
#     layer.trainable = False

# optimiz=keras.optimizers.Adam(lr=0.00025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# def as_keras_metric(method):
#     import functools
#     from keras import backend as K
#     import tensorflow as tf
#     @functools.wraps(method)
#     def wrapper(self, args, **kwargs):
#         """ Wrapper for turning tensorflow metrics into keras metrics """
#         value, update_op = method(self, args, **kwargs)
#         K.get_session().run(tf.local_variables_initializer())
#         with tf.control_dependencies([update_op]):
#             value = tf.identity(value)
#         return value
#     return wrapper

# precision = as_keras_metric(tf.metrics.precision)
# recall = as_keras_metric(tf.metrics.recall)
# f1_score=as_keras_metric(tf.contrib.metrics.f1_score)

# model.compile(optimizer=optimiz, loss='binary_crossentropy',
#               metrics=['accuracy'
#                        ,precision, recall,f1_score
#                       ])


# print(model.summary())
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# model.load_weights('model_weights/vgg_stage-1.hdf5')
# print('-----Training-------')
# model.fit_generator(train_generator,
#                     steps_per_epoch=235,
#                     validation_data=val_generator,
#                     validation_steps=50, 
#                     epochs=3)
# print("-------Saving Weights---------")
# model.save_weights('model_weights/vgg_stage-2.hdf5')

# print("---------Done--------------")



# # ---------3rd Step of Training-----------


# for layer in model.layers:
#     layer.trainable = True

# optimiz=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# model.compile(optimizer=optimiz, loss='binary_crossentropy',
#               metrics=['accuracy'])


# print(model.summary())
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# model.load_weights('model_weights/vgg_stage-2.hdf5')
# print('-----Training-------')
# model.fit_generator(train_generator,
#                     steps_per_epoch=235,
#                     validation_data=val_generator,
#                     validation_steps=50, 
#                     epochs=10)
# print("-------Saving Weights---------")
# model.save_weights('model_weights/vgg_stage-3.hdf5')

# print("---------Done--------------")


# # # ---------4th Step of Training-----------


# for layer in model.layers:
#     layer.trainable = True

# optimiz=keras.optimizers.Adam(lr=0.000025, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# model.compile(optimizer=optimiz, loss='binary_crossentropy',
#               metrics=['accuracy'])


# print(model.summary())
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# model.load_weights('model_weights/vgg_stage-3.hdf5')
# print('-----Training-------')
# model.fit_generator(train_generator,
#                     steps_per_epoch=235,
#                     validation_data=val_generator,
#                     validation_steps=50, 
#                     epochs=10)
# print("-------Saving Weights---------")
# model.save_weights('model_weights/vgg_stage-4.hdf5')
# # ---------5th Step of Training Adding Dropout-----------


for layer in model.layers:
    layer.trainable = True

optimiz=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


model.compile(optimizer=optimiz, loss='binary_crossentropy',
              metrics=['accuracy'])


print(model.summary())
ImageFile.LOAD_TRUNCATED_IMAGES = True

model.load_weights('model_weights/vgg_stage-2.hdf5')
print('-----Training-------')
model.fit_generator(train_generator,
                    steps_per_epoch=442,
                    validation_data=val_generator,
                    validation_steps=99, 
                    epochs=7)
print("-------Saving Weights---------")
model.save_weights('model_weights/vgg_stage-2-dropout.hdf5')

print("---------Done--------------")

