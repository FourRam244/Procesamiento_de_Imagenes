# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:56:43 2022

@author: cesar
"""
import cv2 
import numpy as np 
  
  
img = cv2.imread('m2.png', 0)
  
m, n = img.shape
   
img_new = np.zeros([m, n])
  
for i in range(1, m-1): 
    for j in range(1, n-1): 
        temp = [img[i-1, j-1],
                img[i-1, j],
                img[i-1, j + 1],
                img[i, j-1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j-1],
                img[i + 1, j],
                img[i + 1, j + 1]]
          
        temp = sorted(temp) 
        img_new[i, j]= temp[4]
  
img_new = img_new.astype(np.uint8)
cv2.imshow("Imagen Original", img)
cv2.imshow("Imagen Sin sal y pimienta", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()