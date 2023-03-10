import os

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Set the paths to the weights and the data
weights_path = r'/path/to/vgg/notop/weights'
data_path = r'/path/to/the/dataset'

# Set the image size, number of channels, batch size, and number of epochs
img_size = (224, 224)
channels = 3
batch_size = 33
epochs = 10

# Create a dictionary that maps each character to the number of images
char_dict = {}
for char in os.listdir(data_path):
    char_dict[char] = len(os.listdir(os.path.join(data_path, char)))

# Create an ImageDataGenerator for data augmentation and set its parameters
train_datagen = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)

# Create a generator for the training data
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

# Create a generator for the validation data
validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

# Get the number of classes from the training generator
num_classes = train_generator.num_classes

# Define a function to create the VGG16 model with additional layers
def prepare_model(num_classes):
    model = Sequential()
    model.add(VGG16(include_top=False, weights=weights_path, input_shape=(img_size[0], img_size[1], channels)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

# Create the model
model = prepare_model(num_classes)

# Train the model on the training and validation data
model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=epochs)

# Save the model
model.save('model.h5')