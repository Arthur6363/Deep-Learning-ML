# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:15:58 2024

@author: necip
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2  # Install opencv-python
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image

model = VGG16(weights="imagenet")

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    x = cv2.resize(frame,(224,224))
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    
    predictions = model.predict(x)
    label = decode_predictions(predictions, top=1)[0][0]
        
    label_name, label_confidence = label[1], label[2]
    cv2.putText(frame, f"{label_name} ({label_confidence*100}:.2f%)", (10,30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)
    cv2.imshow("vgg", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
        
cap.release()
cv2.destroyAllWindows()