## model implementation

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization


##creating the sequential model
model = Sequential()

#1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(32, 32, 3),kernel_size=(11, 11), strides=(4, 4), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#1st Maxpool Layer
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



#2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#2nd Maxpool Layer
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



#3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))



#4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))



#5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
#3rd Maxpool Layer
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))



#Passing it to a Fully Connected layer
model.add(Flatten())

# 1st Fully Connected Layer
model.add(Dense(4096, input_shape=(32, 32, 3,)))
model.add(BatchNormalization())
model.add(Activation('relu'))


# Add Dropout to prevent overfitting
model.add(Dropout(0.4))

#2nd Fully Connected Layer
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))
#Add Dropout
model.add(Dropout(0.4))


#3rd Fully Connected Layer
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Activation('relu'))
#Add Dropout
model.add(Dropout(0.4))

#Output Layer
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

#Model Summary
model.summary()
