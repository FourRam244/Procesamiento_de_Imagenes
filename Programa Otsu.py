
import cv2
import numpy as np
from matplotlib import pyplot as plt
#Cargas imagen
imagen = cv2.imread('tanjiro.jpg')
#Convertir imagen gris
gris= cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#Mostrar Imagenes
cv2.imshow("Imagen Original", imagen)
cv2.waitKey(0)
cv2.imshow("Imagen Gris", gris)
cv2.waitKey(0)
cv2.destroyAllWindows()  

#histograma 1
hist = cv2.calcHist([gris],[0], None, [256], [0,256])
plt.figure("Histograma 1")
plt.subplot(121)
plt.plot(hist)

within=[]
#Proceso de otsu
for i in range(len(hist)):
    x,y=np.split(hist,[i])
    x1=np.sum(x)/(imagen.shape[0]*imagen.shape[1])
    y1=np.sum(y)/(imagen.shape[0]*imagen.shape[1])
    
    x2= np.sum([j*t for j, t in enumerate(x)])/np.sum(x)
    y2= np.sum([j*t for j, t in enumerate(y)])/np.sum(y)
    x3 = np.sum([(j-x2)**2*t for j,t in enumerate(x)])/np.sum(x)
    x3=np.nan_to_num(x3)
    print(x1)
    print(y2)
    print(x3)
    y3 = np.sum([(j-y2)**2*t for j,t in enumerate(y)])/np.sum(y)
    
    within.append(x1*x3 + y1*y3)
  
    #m=umbral
z=np.argmin(within)
print("Umbral Deseado: ")
print(z)
#Histograma de gris
hist = cv2.calcHist([gris],[0], None, [256], [0,256])
plt.figure("Histograma Umbral Deseado")
plt.subplot(122)
plt.plot(hist) 
plt.xticks([z],["Umbral Deseado"])

#umbral binarizado
hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
plt.plot(hist, color="gray")
plt.show()
#Umbral ideal optenido
umbral=102
for i in range (gris.shape[0]):
    for j in range (gris.shape[1]):
        if gris[i, j] < umbral:
            gris[i, j]=0
        else:
            gris[i, j]=255
#Mostrar imagen final del umbral binario
cv2.imshow("umbral binario", gris)
cv2.waitKey(0)
cv2.destroyAllWindows()
