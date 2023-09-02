

import cv2
import numpy as np

#leemos la imagen
imagen = cv2.imread("star.jpg")
imgOri= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", imgOri)
cv2.waitKey(0)
cv2.destroyAllWindows()

#realizamos erosion
matriz = np.ones((3,3), np.uint8)
erosion = cv2.erode(imgOri,matriz,iterations = 1)

cv2.imshow("erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

#creamos una matriz de ceros con el tamaño de la imagen original
#esto para guardar el nuevo resultado
perimetro = np.zeros((imgOri.shape), dtype=np.uint8)
#hacemos dos for para recorrer la imagen original
for i in range(imgOri.shape[0]):
    for j in range(imgOri.shape[1]):
        #ahora con el algoritmo del perimetro 
        #que es restar la imagen original menos la de erosion
        #con esto nos ayuda a sacar los bordes
        perimetro[i,j] = imgOri[i,j] - erosion[i,j]
        if perimetro[i,j]<0:
            perimetro = 0
            

#mostramos la imagen          
cv2.imshow("resultado", perimetro)
cv2.waitKey(0)
cv2.destroyAllWindows()





        
            
            

