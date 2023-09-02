# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:00:25 2022

@author: cesar
"""


import cv2
import random
import numpy as np

#leemos la imagen a color
imagen = cv2.imread('paisaje.jpg',0)
ancho, alto = imagen.shape
resultado = np.ones((427,640),np.uint8)
img = np.ones((427,640),np.uint8)

img[:,:] = 0
img[170:270, 250:410] = 255
imagen2= img
resultado2=cv2.subtract(imagen,imagen2)
ancho, alto = imagen2.shape
cv2.imshow('Cuadrado ',img)
cv2.waitKey()

for i in range(ancho):
    for j in range(alto):
        resultado[i,j]=(imagen[i,j].astype(np.uint8)) - (imagen2[i,j].astype(np.uint8))
        if (resultado[i, j] >= 0):
            resultado[i, j] = resultado[i, j]

        else:
            resultado[i, j] = 255

        resultado[i, j] = resultado[i, j]
           
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Resta', resultado2)
cv2.waitKey()
