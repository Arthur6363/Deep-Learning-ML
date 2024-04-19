# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:31:48 2024

@author: necip
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import VGG16
from PIL import Image
import matplotlib.pyplot as plt 
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri yolu
veri_yolu = "C:/Users/necip/Downloads/airplane-dataset-asoc/airplane-dataset-trans"

# Veri artırma ve hazırlama
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Eğitim verisi için veri setini yükleme
train_set = datagen.flow_from_directory(
    veri_yolu,
    target_size=(224, 224),  # Resim boyutları
    batch_size=32,
    class_mode='categorical',  # Sınıflandırma etiketlerini belirleme
    subset='training')  # Eğitim verisi için

# Test verisi için veri setini yükleme
test_set = datagen.flow_from_directory(
    veri_yolu,
    target_size=(224, 224), 
    batch_size=32,
    class_mode='categorical',  
    subset='validation')  # Test verisi için


from tensorflow.keras.preprocessing.image import load_img

resim_yolu = veri_yolu + "/type-16(A-26)/16-1.jpg"  # test_set ile dosya yolunu birleştirme
resim = load_img(resim_yolu)

# Resmi göster
#plt.imshow(resim)
#plt.axis('off')
#plt.show()

# Test setindeki sınıf isimlerini gösterme
#print("Test Seti Sınıf İsimleri:", test_set.class_indices)

vgg = VGG16()

vgg_layers= vgg.layers

kullanilankatmanlar = len(vgg.layers)-1


model = Sequential( )


for i in range(kullanilankatmanlar):
    model.add(vgg_layers[i])

for layers in model.layers:
    layers.trainable=False
    
model.add(Dense(36, activation="softmax"))
#model.summary()

model.compile(loss= "categorical_crossentropy", optimizer = "adam", metrics=["accuracy"])

batch_size = 4

model.fit_generator(train_set,
                    steps_per_epoch=100//batch_size,
                    epochs=4,
                    validation_data=test_set,
                    validation_steps=200//batch_size)
























