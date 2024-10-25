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
   loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = False)
  
  
  
  
   #STEP 4
   cnn_model.compile(optimizer = sgd_optimizer, loss = loss_fn, metrics = ['accuracy'])
  
  
  
  
   #STEP 5
   num_epochs = 5 # Number of epochs
   t0 = time.time() # start time
  
  
  
  
   history = cnn_model.fit(X_train, y_train, epochs = num_epochs)
   
   
   
   t1 = time.time() # stop time
   
   print('Elapsed time: %.2fs' % (t1-t0))
  
  
  
  
   #STEP 6
   loss, accuracy = cnn_model.evaluate(X_test, y_test)
  
   print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))
  
   # Plot training loss and accuracy
   #plt.plot(range(1, num_epochs + 1), history.history['loss'], label='Training Loss')
  
   #plt.xlabel('Epoch')
   #plt.ylabel('Loss')
   #plt.legend()
   #plt.show()
  
   # Plot training accuracy
   #plt.plot(range(1, num_epochs + 1), history.history['accuracy'], label='Training Accuracy')
  
   #plt.xlabel('Epoch')
   #plt.ylabel('Accuracy')
   #plt.legend()
   #plt.show()


#display the test sets
def display():
   # Make predictions on the test set
   logits = cnn_model.predict(X_test)
   predictions = logits.argmax(axis = 1)   
   ## Plot individual predictions
   plot_imgs(X_test[:25], predictions[:25])


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



