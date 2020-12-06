#all imports here
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


#dimension of the image in the dataset
img_size = 48
#batches of image
batch_size = 64

#Generate batches of tensor image data with real-time data augmentation.
#using horizontal flip to create random data from the small dataset given
data_generator = ImageDataGenerator(horizontal_flip=True)
#read the data from the directory
#reference:https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
train_data = data_generator.flow_from_directory('data/train/',
                                                target_size=(img_size, img_size),
                                                color_mode='grayscale',
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True)

test_data = data_generator.flow_from_directory('data/test/',
                                               target_size=(img_size, img_size),
                                               color_mode='grayscale',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True)


#make a linear model
model = Sequential()

#use 2d kernel on the input matrix
model.add(layers.Conv2D(64,(3,3), input_shape=(48, 48,1)))
#using normalization on data to scale the data such that mean is 0 and variance is one
model.add(layers.BatchNormalization())
#applying activation function relu to it
model.add(layers.Activation('relu'))
#use dropout to avoid overfitting
model.add(layers.Dropout(0.25))
#using max pooling to down sample the data
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#CNN layer 2
model.add(layers.Conv2D(128,(5,5)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#CNN layer 3
model.add(layers.Conv2D(512,(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#CNN layer 4
model.add(layers.Conv2D(512,(3,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

#flatten to reduce the dimension of the data to fit the dense layer
model.add(layers.Flatten())

#dense layers with 256 neuron
model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.Activation('relu'))

#dense layers with 512 neurons
model.add(layers.Dense(512))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.25))
model.add(layers.Activation('relu'))

#finally a dense layer to get the softmax of 7 emotions
model.add(layers.Dense(7, activation='softmax'))

#using adaptive gradient descent as optimizer
optimizer = Adam(learning_rate=0.01)
#using cotegorical loss as classification problem
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#number of epochs
epochs = 35

#number of steps
steps_per_epoch = train_data.n//train_data.batch_size
validation_steps = test_data.n//test_data.batch_size

#reduce the learning rate 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=2, min_lr=0.00001, mode='auto')

#save the best weights
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy',
                             save_weights_only=True, mode='max', verbose=1)

callbacks = [reduce_lr, checkpoint]

#traing the model
history = model.fit(x=train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs,
                    validation_data=test_data,
                    validation_steps=validation_steps,
                    callbacks=callbacks)

#store the model in json file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
