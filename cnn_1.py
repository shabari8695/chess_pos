#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pydot
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D


# In[22]:


train_size = 10000
test_size = 3000

train = glob.glob("train/*.jpeg")
test = glob.glob("test/*.jpeg")

shuffle(train)
shuffle(test)

train = train[:train_size]
test = test[:test_size]

piece_symbols = 'prbnkqPRBNKQ'


# In[10]:


train[:5]


# In[11]:


unique = set(train)
unique = list(unique)
len(unique)


# In[12]:


def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


# In[13]:


def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img)
    img_read = transform.resize(
      img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)


# In[15]:


def onehot_from_fen(fen):
    eye = np.eye(13)
    output = np.empty((0, 13))
    fen = re.sub('[-]', '', fen)

    for char in fen:
        if(char in '12345678'):
            output = np.append(
              output, np.tile(eye[12], (int(char), 1)), axis=0)
        else:
            idx = piece_symbols.index(char)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)

    return output


# In[16]:


def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output


# In[17]:


def train_gen(features, labels, batch_size):
    for i, img in enumerate(features):
        y = onehot_from_fen(fen_from_filename(img))
        x = process_image(img)
        yield x, y


# In[18]:


def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)


# In[19]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(25, 25, 3), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(13, kernel_initializer='he_normal'))
model.add(Activation('softmax'))


# In[20]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[23]:


model.fit_generator(train_gen(train, None, 64), steps_per_epoch=train_size)


# In[31]:


res = (
  model.predict_generator(pred_gen(test, 64), steps=test_size)
  .argmax(axis=1)
  .reshape(-1, 8, 8)
)


# In[32]:


pred_fens = np.array([fen_from_onehot(one_hot) for one_hot in res])
test_fens = np.array([fen_from_filename(fn) for fn in test])

final_accuracy = (pred_fens == test_fens).astype(float).mean()

print("Final Accuracy: {:1.5f}%".format(final_accuracy))


# In[33]:


model_json = model.to_json()
with open("models/model_cnn_1.json", "w") as json_file:
    json_file.write(model_json)


# In[34]:


model.save_weights("weights/model_cnn_1.h5")
print("Saved model to disk")


# In[ ]:




