#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D


# In[20]:


train_size = 10000
test_size = 3000

train = glob.glob("train/*.jpeg")
test = glob.glob("test/*.jpeg")

shuffle(train)
shuffle(test)

train = train[:train_size]
test = test[:test_size]

piece_symbols = 'prbnkqPRBNKQ'


# In[3]:


len(train)


# In[4]:


def fen_from_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


# In[5]:


print(fen_from_filename(train[0]))
print(fen_from_filename(train[1]))
print(fen_from_filename(train[2]))


# In[6]:


f, axarr = plt.subplots(1,3, figsize=(120, 120))

for i in range(0,3):
    axarr[i].set_title(fen_from_filename(train[i]))
    axarr[i].imshow(mpimg.imread(train[i]))
    axarr[i].axis('off')


# In[7]:


def process_image(img):
    downsample_size = 200
    square_size = int(downsample_size/8)
    img_read = io.imread(img)
    img_read = transform.resize(
      img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)


# In[8]:


tiles = process_image(train[0])


# In[9]:


tiles.shape


# In[10]:


def plot_features(features):
        #features = np.reshape(features,(features.shape[1],features.shape[2],features.shape[3]))
        n_features = features.shape[0]
        n_cols = 8
        n_rows = (n_features// n_cols) + 1
        fig = plt.figure(figsize=(n_cols,n_rows))
        for i in range(features.shape[0]):
            ax1 = fig.add_subplot(n_rows,n_cols,i+1)
            feature = features[i]
            ax1.imshow(feature)
            f = plt.imshow(feature)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
        
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()


# In[11]:


plot_features(tiles)


# In[12]:


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


# In[13]:


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


# In[14]:


def train_gen(features, labels, batch_size):
    for i, img in enumerate(features):
        y = onehot_from_fen(fen_from_filename(img))
        x = process_image(img)
        yield x, y


# In[15]:


def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)


# In[17]:


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(25, 25, 3), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(13, kernel_initializer='he_normal'))
model.add(Activation('softmax'))


# In[18]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit_generator(train_gen(train, None, 64), steps_per_epoch=train_size)


# In[ ]:




