import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Lenna.png")
gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ancho, alto = gris.shape

#cv2.imshow("imagen", gris)
x = np.linspace(0,255, num = 256, dtype = np.uint8)
y = np.zeros(256)
plt.figure(1)
plt.subplot(121)
plt.title("Histograma")

#hist = cv2.calcHist([gris],[0], None, [256], [0,256])
#plt.plot(hist, color='gray')

for i in range(ancho):
    for j in range(alto):
        v = gris[i,j]
        y[v] = y[v]+1
        
         
plt.plot(x,y)
plt.xlim(0,256)
plt.subplot(122)
plt.title("Imagen")
plt.imshow(gris)
plt.show()
        

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.bar(x,y)