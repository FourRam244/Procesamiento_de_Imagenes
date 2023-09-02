# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:53:35 2022

@author: cesar
"""

import cv2
import numpy as np

imagen = cv2.imread("Lenna.png",0) # gris
fila, columna = imagen.shape #ayudar a saber cuanto mide


mat = np.ones((517, 517), np.uint8) # matriz de unos con dimensiones mas grandes para los 5 pixeles
mat[5:517,5:517] = imagen[:,:] #ponemos la imagen dejando 5 pixeles
#de acuerdo a las medidas ocupamos 5 pixeles para pintarlo
mat[0:517, 0:5] = 255
mat[0:5, 0:517] = 255
mat[512:517, 0:517] = 255
mat[0:517, 512:517] = 255

#muestra la nueva imagen
cv2.imshow("Borde", mat)
cv2.waitKey(0)
cv2.destroyAllWindows()