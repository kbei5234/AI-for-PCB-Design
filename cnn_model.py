import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # suppress info and warning messages
import tensorflow.keras as keras
import math
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


X_data = np.load('X.npy')
y_data = np.load('y.npy')


# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=123)


print("X_train shape:", X_train.shape)
print("y_train.shape:", y_train.shape)
print("X_test.shape:", X_test.shape)
print("y_test.shape:", y_test.shape)



X_train[0].shape


#Commenting out reshape to avoid error
#X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))


def train_model(size,strides):
   # 1. Create CNN model object
  
  
   cnn_model = keras.Sequential()
  
  
   # 2. Create the input layer and add it to the model object:
   input_shape=(X_train.shape[1:])
   input_layer = keras.layers.InputLayer(input_shape = input_shape)
   cnn_model.add(input_layer)                                     
   
   
   # 3. Create the first convolutional layer and add it to the model object:
   conv_1 = keras.layers.Conv2D(filters = 16, kernel_size = size, strides=strides)
   batchNorm_1 = keras.layers.BatchNormalization()
   ReLU_1 = keras.layers.ReLU()
   cnn_model.add(conv_1)
   cnn_model.add(batchNorm_1)
   cnn_model.add(ReLU_1)
   
   
   # 4. Create the second convolutional layer and add it to the model object:
   conv_2 = keras.layers.Conv2D(filters = 32, kernel_size = size, strides=strides)
   batchNorm_2 = keras.layers.BatchNormalization()
   ReLU_2 = keras.layers.ReLU()
   cnn_model.add(conv_2)
   cnn_model.add(batchNorm_2)
   cnn_model.add(ReLU_2)
   
   
   # 5. Create the third convolutional layer and add it to the model object:
   conv_3 = keras.layers.Conv2D(filters = 64, kernel_size = size, strides=strides)
   batchNorm_3 = keras.layers.BatchNormalization()
   ReLU_3 = keras.layers.ReLU()
   cnn_model.add(conv_3)
   cnn_model.add(batchNorm_3)
   cnn_model.add(ReLU_3)
   
   
   # 6. Create the fourth convolutional layer and add it to the model object:
   conv_4 = keras.layers.Conv2D(filters = 128, kernel_size = size, strides=strides)
   batchNorm_4 = keras.layers.BatchNormalization()
   ReLU_4 = keras.layers.ReLU()
   cnn_model.add(conv_4)
   cnn_model.add(batchNorm_4)
   cnn_model.add(ReLU_4)
   
   
   # 7. Create the pooling layer and add it to the model object:
   pooling_layer = keras.layers.GlobalAveragePooling2D()
   cnn_model.add(pooling_layer)
  
  
   # 8. Create the output layer and add it to the model object:
   output_layer = keras.layers.Dense(units = 1) #changed units to 1 bc regression
   cnn_model.add(output_layer)
  
  
   cnn_model.summary()
   
   return cnn_model
