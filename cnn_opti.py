#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:04:27 2024

@author: yzy
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # suppress info and warning messages
import tensorflow.keras as keras
import math
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from cnn_model import train_model, X_train, y_train, X_test, y_test
# Function to visualize the data
def plot_imgs(images, labels=None):
   subplots_x = int(math.ceil(len(images) / 5))
   plt.figure(figsize=(10,2*subplots_x))
   for i in range(min(len(images), subplots_x*5)):
       plt.subplot(subplots_x,5,i+1)
       plt.xticks([])
       plt.yticks([])
       plt.grid(False)
       plt.imshow(images[i], cmap=plt.cm.binary)
       if labels is not None:
           plt.xlabel(labels[i])
   plt.show()
 
def opti(cnn_model):
   #STEP 2
   sgd_optimizer = keras.optimizers.SGD(learning_rate = 0.1)
   
   
   #STEP 3
   loss_fn = keras.losses.MeanSquaredError()
  
  
   #STEP 4
   cnn_model.compile(optimizer = sgd_optimizer, loss = loss_fn)
  
  
   #STEP 5
   num_epochs = 1 # Number of epochs
   t0 = time.time() # start time
  
   history = cnn_model.fit(X_train, y_train, epochs = num_epochs, validation_split=0.2)
   
   t1 = time.time() # stop time
   
   print('Elapsed time: %.2fs' % (t1-t0))
  
  
  
  
   #STEP 6
   MSE = cnn_model.evaluate(X_test, y_test)
   RMSE = np.sqrt(MSE)
   
   print('Loss: MSE = ', str(MSE), 'RMSE = ', str(RMSE))
   
   # STEP 7
   prediction = cnn_model.predict(X_test)
   r2 = r2_score(y_test, prediction)
   print('R²: ', str(r2))
   
   
   
   
   # Plot training and validation loss
   #plt.plot(range(1, num_epochs + 1), np.sqrt(history.history['loss']), label='Training RMSE')
   #plt.plot(range(1, num_epochs + 1), np.sqrt(history.history['val_loss']), label='Validation RMSE')

   #plt.xlabel('Epoch')
   #plt.ylabel('RMSE')
   #plt.legend()
   #plt.show()
   

#display the test sets
def display():
   # Make predictions on the test set
   logits = cnn_model.predict(X_test)
   #predictions = logits.argmax(axis = 1) #good for classification    
   ## Plot individual predictions
   plot_imgs(X_test, logits)


# change kernel sizes
kernel_sizes = [3, 5]
for size in kernel_sizes:
   print(f'kernel size = {size}, strides = (1,1) (default)')
   cnn_model = train_model(size,strides=(1,1)) #default strides
   opti(cnn_model)
   display()


#change strides
strides_values = [(1,2), (2,2)]
for strides in strides_values:
   print(f'kernel size = 3, strides = {strides}')
   cnn_model = train_model(3,strides=strides) #kernel size fixed to 3
   opti(cnn_model)
   display()


  

  
  
  
  
   