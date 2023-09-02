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




for i in range(3,imagen.shape[1]):
    pot = 2**i
    
    for j in range(0, imagen.shape[1],pot):
        mt[:,j] = 255
'''
mat[:,0:8] = 255
mat[:,32:64] = 255
mat[:,64:128] = 255
mat[:,256:512] = 255
'''
cv2.imshow("Verticales", mt)
cv2.waitKey(0)
cv2.destroyAllWindows()