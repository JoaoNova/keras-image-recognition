"""
# Used to force tensorflow to use CPU instead of GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import cifar10  # Importing our images dataset from CIFAR10
import matplotlib.pyplot as plt  # Just used to show images

# Our images are 32x32 pixels with RGB, so 32x32x3
# 50000 Images for training & 10000 for testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Creating training and testing groups. X as input (img) and Y as output (label)

# Number - Label - One-hot encoding
# 0 -  airplane  - [1 0 0 0 0 0 0 0 0 0]
# 1 - automobile - [0 1 0 0 0 0 0 0 0 0]
# 2 -    bird    - [0 0 1 0 0 0 0 0 0 0]
# 3 -    cat     - [0 0 0 1 0 0 0 0 0 0]
# 4 -    deer    - [0 0 0 0 1 0 0 0 0 0]
# 5 -    dog     - [0 0 0 0 0 1 0 0 0 0]
# 6 -    frog    - [0 0 0 0 0 0 1 0 0 0]
# 7 -    horse   - [0 0 0 0 0 0 0 1 0 0]
# 8 -    ship    - [0 0 0 0 0 0 0 0 1 0]
# 9 -    truck   - [0 0 0 0 0 0 0 0 0 1]

# Converting categorical values as One Hot Encoded binary arrays
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

# Converting the RGB values of images from 0~255 to 0~1 range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# Creating our empty sequential model, where we will add layers at a time
model = Sequential()

# We will use ReLU activation for all our layers, except for the last which is a Softmax activation.
# 1st & 2nd Layers: Convolutional Layers [Filter = 3x3, Stride = 1, Depth = 32]
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))  # Default stride is 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # After first layer, we dont need to specify input_shape
# 3rd Layer: Max Pool Layer [Filter = 2x2, Stride = 2]
model.add(MaxPooling2D(pool_size=(2, 2)))  # Default stride for a max pooling layer is the pool size, in this case 2
# 4th Layer: Dropout Layer [Probability = 0.25]
model.add(Dropout(0.25))
# 5th & 6th Layers: Convolutional Layers [Filter = 3x3, Stride = 1, Depth = 64]
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# 7th Layer: Max Pool Layer [Filter = 2x2, Stride = 2]
model.add(MaxPooling2D(pool_size=(2, 2)))
# 8th Layer: Dropout Layer [Probability = 0.25]
model.add(Dropout(0.25))
# 9th Layer: Flatten Layer
model.add(Flatten())  # This Layer is used to flatten the cubic spacial format that our neurons have thus far, to a row format
# 10th Layer: Fully Connected Layer [Neurons = 512]
model.add(Dense(512, activation='relu'))
# 11th Layer: Dropout [Probability = 0.5]
model.add(Dropout(0.5))
# 12th Layer: Fully Connected Layer [Neurons = 10] with Softmax activation (Output Layer)
model.add(Dense(10, activation='softmax'))
# Shows the summary of the full architecture
model.summary()

# Loss function used is Categorical Cross Entropy, that is used for a classification problem of many classes
# Our Optimizer will be Adam, that is a Stochastic Gradient Descent algorithm with modifications to train better
# And we are tracking the Accuracy of our model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# We are using a batch size of 32 and 20 epochs (How many times all images go through and back in our CNN)
# And we split 20% of our train set to be our validation set
hist = model.fit(x_train, y_train_one_hot,
                 batch_size=32, epochs=20,
                 validation_split=0.2)

# Visualization of model training and validation loss over the number of epochs
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Visualization of model training and validation accuracy over the number of epochs
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Now, using our model to evaluate the test set
model.evaluate(x_test, y_test_one_hot)

# Saving our model
model.save('model2.h5')
