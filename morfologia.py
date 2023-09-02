import cv2
import numpy as np
import matplotlib.pyplot as plt


#BINARIZACION DE IMAGEN 
#leemos la imagen
im = cv2.imread("star.jpg")
imgOri=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imshow("Color", im)
cv2.waitKey(0)
cv2.imshow("Gris", imgOri)
cv2.waitKey(0)

ancho, alto=imgOri.shape
#convertimos la imagen (matriz) a un vector
array_im=np.array(imgOri)
vector=array_im.ravel()
#histograma
plt.hist(vector,256, [0,256])
plt.xticks([101],["Umbral ideal"])
#plt.plot(cdf_normalized, color = 'blue')
plt.show()

#volvemos a convertir la imagen a matriz
array_im=np.array(imgOri)
umbral=147
 
#binarizamos la imagen 
for i in range(ancho):
    for j in range(alto):
        if umbral <=imgOri[i,j]:
            
            imgOri[i,j]=255
        else:
            imgOri[i,j]=0
            


cv2. imshow("imagen binarizada", imgOri )
cv2. waitKey()


#proceso de dilatacionA
#creamos la mascara en forma de cruz
dila = np.array(((0,1,0),(1,0,1),(0,1,0)),dtype=np.uint8)
#matriz para almacenar el resultado o donde se veran los cambios

imgDila = np.zeros((imgOri.shape), dtype=np.uint8)
imgDila = imgOri
for k in range(2):
    mat = np.pad(imgOri,(1,1), 'edge')#trabajamos con matrices
    for i in range(1,imgOri.shape[0]):
        for j in range(1,imgOri.shape[1]):
            #con los for hacemos el recorrido de la imagen
            #si la suma de donde va pasando la mascar excede los 255
            #entonces toma el valor de 255
            if np.sum(dila * mat[i-1:i+2,j-1:j+2]) >=255:
                imgDila[i,j] = 255

cv2.imshow("Dilatacion", imgDila)
cv2.waitKey(0)



#proceso de erosion
#matriz para almacenar el nuevo resuktado
ero = np.zeros((imgOri.shape), dtype=np.uint8)
ero = imgOri
#creamos la mascara
matEro = np.array(((0,1,0),(1,0,1),(0,1,0)),dtype=np.uint8)
for k in range(2):
    mat1 = np.pad(imgOri,(1,1), 'edge')#trabajamos con matrices
    for i in range(1,imgOri.shape[0]):
        for j in range(1,imgOri.shape[1]):
            #hacemos el recorrido de la imagen
            #condicion de los nuevos parametros donde ira pasando la 
            #mascara
            #si es menor que 255*4 entonces toma el valor de 0
            if np.sum(matEro * mat1[i-1:i+2,j-1:j+2] ) <255*4:
                ero[i,j] = 0
#mostramos la imagen                
cv2.imshow("Erosion", ero)
cv2.waitKey(0)



#proceso de apertura
#tomamos la imagen que ya esta en el proceso de erosion
#Y = X * S = (X EROSION S) DILATACION S

apertura = np.zeros((imgOri.shape), dtype = np.uint8)
apertura = ero
#luego tenemos que hacer dilatacion

#creamos nuestra mascara
matApe = np.array(((0,1,0),(1,0,1),(0,1,0)),dtype=np.uint8)
for k in range(2):
    mat2 = np.pad(imgOri,(1,1), 'edge')#trabajamos con matrices
    for i in range(1,imgOri.shape[0]):
        for j in range(1,imgOri.shape[1]):
            #con los for hacemos el recorrido de la imagen
            #si la suma de donde va pasando la mascar excede los 255
            #entonces toma el valor de 255
            if np.sum(matApe * mat2[i-1:i+2,j-1:j+2]) >=255:
                apertura[i,j] = 255
cv2.imshow("Apertura", apertura)
cv2.waitKey(0)



#proceso de cierre
#tomamos la imagen que ya tiene el proceso de dilatacion
# Y = X * S = (X DILATACION S) EROSION S

cierre = np.zeros((imgOri.shape), dtype=np.uint8)
cierre = imgDila
#una vez que ya tebemos la imagen dilatara hacemos la erosion

#mascara 
matCierre = np.array(((0,1,0),(1,0,1),(0,1,0)),dtype=np.uint8)

for k in range(2):
    mat3 = np.pad(imgOri,(1,1), 'edge')#trabajamos con matrices
    for i in range(1,imgOri.shape[0]):
        for j in range(1,imgOri.shape[1]):
            #hacemos el recorrido de la imagen
            #condicion de los nuevos parametros donde ira pasando la 
            #mascara
            #si es menor que 255*4 entonces toma el valor de 0
            if np.sum(matCierre * mat3[i-1:i+2,j-1:j+2] ) <255*4:
                cierre[i,j] = 0
cv2.imshow("Cierre", cierre)
cv2.waitKey(0)
cv2.destroyAllWindows()

                
    