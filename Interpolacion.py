import cv2
import numpy as np
import math

image = cv2.imread('barca.png')

ancho, alto, color = image.shape

#rotar
aux = np.zeros((int(alto),int(ancho),3), dtype=np.uint8)
angulo=20
for x in range(alto):
    for y in range(ancho):
        xr = abs(int(y * math.cos(math.pi/angulo) - x * math.sin(math.pi/angulo)))
        yr = abs(int(x * math.cos(math.pi/angulo) + y * math.sin(math.pi/angulo)))
        if xr > 0 and yr > 0 and xr < ancho and yr < alto:
            aux[xr,yr,:] = image[y,x,:]
cv2.imshow('Imagen original',image)
cv2.imshow('Imagen de rotada',aux)

#escalar
escala=0.12
aux2 = np.zeros((int(alto),int(ancho),3), dtype=np.uint8)
for y in range(alto):
    for x in range(ancho):
        aux2[int(y*escala),int(x*escala),:] = image[y,x,:]

cv2.imshow('Imagen de escalada',aux2)
cv2.waitKey(0)
cv2.destroyAllWindows()





  
                  
