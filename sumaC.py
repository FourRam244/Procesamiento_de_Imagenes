# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 19:00:25 2022

@author: cesar
"""


import cv2
import random
import numpy as np

#leemos la imagen a color
imagen = cv2.imread('paisaje.jpg')

image = np.ones((427,640,3), np.uint8)
resultado = np.ones((427,640,3), np.uint8)
img = np.ones((427,640,3), np.uint8)

image[:,:,:] = imagen[:,:,:]

img[:,:,0] = random.randint(0,255)
img[:,:,1] = random.randint(0,255)
img[:,:,2] = random.randint(0,255)
img[170:270, 250:410, :] = 130

cv2.imshow('Cuadrado Centro', img)
cv2.waitKey()

cB = np.ones((427,640), np.uint8)
cB[:,:] = image[:,:,0]
cG = np.ones((427,640), np.uint8)
cG[:,:] = image[:,:,1]
cR = np.ones((427,640), np.uint8)
cR[:,:] = image[:,:,2]

cB2 = np.ones((427,640), np.uint8)
cB2[:,:] = img[:,:,0]
cG2 = np.ones((427,640), np.uint8)
cG2[:,:] = img[:,:,1]
cR2 = np.ones((427,640), np.uint8)
cR2[:,:] = img[:,:,2]


for i in range(427):
    for j in range(640):
        p1 = int(cB[i,j])
        p2 = int(cG[i,j])
        p3 = int(cR[i,j])
        p4 = int(cB2[i,j])
        p5 = int(cG2[i,j])
        p6 = int(cR2[i,j])
        s1 = p1 + p4
        s2 = p2 + p5
        s3 = p3 + p6
        if(s1 >=255):
            s1 = s1
        if(s2 >=255):
            s2 = s2
        if(s3 >=255):
            s3 = s3
       

            resultado[i,j,0] = s1
            resultado[i,j,1] = s2
            resultado[i,j,2] = s3

cv2.imshow('Imagen Original', imagen)
cv2.imshow('Suma', resultado)
cv2.waitKey()