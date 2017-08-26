import numpy as np

import keras
# Feed forward
from keras.models import Sequential
# Core layers
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers
from keras.layers import Conv2D, MaxPooling2D
# Utils
from keras.utils import np_utils
# Data
from keras.datasets import mnist

from keras import backend as K

from matplotlib import pyplot as plt

# Ignore tensorflow CPU warnings as we are using the GPU build anyways
# See https://github.com/tensorflow/tensorflow/issues/7778
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# Workaround for crash
# See https://github.com/tensorflow/tensorflow/issues/6698
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


if __name__ == '__main__':
    np.random.seed(123) 

    # Load pre-shuffled MNIST data into train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print('Initial shape:')
    print(X_train.shape)

    # Plot first sample
    try:
        plt.imshow(X_train[0])
    except Exception:
        print('Failed to display plot')

    # Reshape to a depth of 1, as MNIST is not RGB
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    print('Shape after setting depth to one:')
    print(X_train.shape)


    # Convert underlying data types
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize values
    X_train /= 255
    X_test /= 255

    print('Shape of class labels as array:')
    print(y_train.shape)

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    
    print('Shape of class labels after conversion to 10-dimensional class matrix:')
    print(y_train.shape)


    # Define the model 	
    model = Sequential()
    img_rows = img_cols = 28
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)
    model.add(Conv2D(32, 3, activation='relu', input_shape=input_shape))
    print("Shape of model's output:")
    print(model.output_shape)
	
    # Add some more conv layers
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # Add a FCNN at the end
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Declare how the training should be evaluated and compile model
    print('Compiling model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Done.')

    # Train!
    print('Starting training')
    model.fit(X_train, Y_train, 
              batch_size=32, epochs=10, verbose=1)
    # Evaluate
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Finished training with score ' + str(score))

    # Save the model
    MODEL_FILENAME = 'mnist.h5'
    print ('Saving trained model as ' + MODEL_FILENAME)
    model.save(MODEL_FILENAME)
    