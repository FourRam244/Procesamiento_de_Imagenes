

import cv2
import random
import numpy as np

#leemos la imagen a color
imagen = cv2.imread('cielo.jpg',0)
ancho, alto = imagen.shape
print(ancho,alto)
resultado = np.ones((510, 680), np.uint8)
img = np.ones((510,680), np.uint8)

img[:,:] = 255
img[0:200, 0:200] = 0
imagen2= img
ancho, alto = imagen2.shape
cv2.imshow('Cuadrado ', imagen2)
cv2.waitKey()


for i in range(ancho):
    for j in range(alto):
        resultado[i,j]=imagen[i,j].astype(np.uint16) * imagen2[i,j].astype(np.uint16)
        if(resultado[i,j] >0):
            resultado[i,j] = resultado[i,j]

        else:
            resultado[i,j] = 0
           

            resultado[i,j] = resultado[i,j]
           
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Multiplicacion', resultado)
cv2.waitKey()