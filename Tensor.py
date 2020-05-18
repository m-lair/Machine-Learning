#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[6]:


train_images.shape


# In[20]:


train_labels.shape


# In[14]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']


# In[15]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)


# In[16]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[18]:


plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])


# In[21]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[22]:


model.compile(optimizer=tf.train.AdamOptimizer(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[23]:


model.fit(train_images, train_labels, epochs = 5)


# In[24]:


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


# In[ ]:




