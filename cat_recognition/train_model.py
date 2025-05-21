from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


from PIL import Image
import numpy as np
import os
from dotenv import dotenv_values

vars = dotenv_values(".env")
total = int(vars.get("train_total", 0))
print(total)

# input resolution of training images
img_width, img_height = 150, 150

# rescale pixel values of training images from [0, 255] to [0, 1] to Normal Data
trainRescale = ImageDataGenerator(rescale=1. / 255)

# flow from directory will load images and label them based on folder
trainData = trainRescale.flow_from_directory(
    './train/SmallPetImages/',
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# define model layers
model = Sequential()

# conv2d 32 filters, size 3x3. low level features (edges/curves)
# ReLU Activation function that replaces negatives with 0
# Max Pooling 2D, reduce image by tanking max value from 2x2 region
model.add(Conv2D(32, (3,3), input_shape = (img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# repeat

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# extract more features 64 filter 3x3

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten output into 1-D vector
model.add(Flatten())

# pass to dense layer 64 neurons
# ReLU action function
# Droput Layer with 50% Drop Rate To Prevent Overfitting/Complexity
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Dense Layer 1 Neuron & sigmoid activation to produce probability value between [0, 1]
model.add(Dense(1))
model.add(Activation('sigmoid')) # sigmoid produce probability value

# compile the mode (rmsprop is generally okay for image classifier)
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics =['accuracy']
)

# train the model, steps per epoch is how many weights
# are updated per epoch
train_samples = len(trainData.filenames)
steps_per_epoch = train_samples // trainData.batch_size

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6)
]


model.fit(
    trainData,
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    callbacks=callbacks
)

# save the model


model.save_weights(f'./models/{total}_model.weights.h5')
model.save(f'./models/{total}_model_keras.h5')
