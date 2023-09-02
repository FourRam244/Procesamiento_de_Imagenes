# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:00:25 2022

@author: cesar
"""


import cv2
import numpy as np


imagen = cv2.imread("Lenna.png",0) # gris

#matriz del mismo tamaño de la imagen
mt = np.zeros((imagen.shape), np.uint8)
mt[:,:] = imagen[:,:] #asignamos a la matriz la imagen
x = 0

for i in range(0, imagen.shape[0], 30):
    mt[i:i+10] = 255
    

    
cv2.imshow("Horiontal", mt)
cv2.waitKey(0)
cv2.destroyAllWindows()