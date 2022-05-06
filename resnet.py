#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import os
import shutil
import glob
import matplotlib.pyplot as plt


# In[5]:


dir = r"tmp\train_data"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir, labels = "inferred", label_mode = "int", class_names = ['COVID','Viral Pneumonia'],
    color_mode = "rgb", batch_size = 32, image_size = (224, 224), 
    shuffle = True, seed = 42, validation_split = 0.1, subset = "training", interpolation = "bicubic")

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dir, labels = "inferred", label_mode = "int", class_names = ['COVID','Viral Pneumonia'],
    color_mode = "rgb", batch_size = 32, image_size = (224, 224), 
    shuffle = True, seed = 42, validation_split = 0.1, subset = "validation", interpolation = "bicubic")


# In[7]:


import tensorflow as tf
import tensorflow
from tensorflow import keras
from keras.applications.resnet import ResNet50
from tensorflow.keras.layers import Flatten, Dense, MaxPool2D, Dropout


# In[8]:


res  = ResNet50( input_shape=(224,224,3), include_top=False)


# In[9]:


# We won't train all parameters again
for layer in res.layers:
  layer.trainable = False


# In[10]:


x = Flatten()(res.output)
out = Dense(units=2, activation='sigmoid', name='predictions')(x)


# Creating our model
model = Model(inputs=res.input, outputs=out)


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


ACCURACY_THRESHOLD = 0.95
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):   
          print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
          self.model.stop_training = True

es  = myCallback()


# In[ ]:


hist = model.fit(train_ds, steps_per_epoch=10, epochs=30, validation_data=val_ds, validation_steps=16, callbacks=[es])


# In[ ]:


model.save("covid-resnet50")


# In[ ]:


# checking out the accuracy of our model 

acc = model.evaluate_generator(generator= test)[1] 

print(f"The accuracy of your model is = {acc * 100} %")


# In[ ]:





# In[ ]:





# In[ ]:




