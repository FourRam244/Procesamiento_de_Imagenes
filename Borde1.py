import cv2
import numpy as np
pixeles = cv2.imread('eri.jpg')
gris= cv2.cvtColor(pixeles, cv2.COLOR_BGR2GRAY)
w, h = gris.shape
imBor= np.zeros([w, h])
g = ([1,1])
#Bordes 1
for i in range(0,w-2):
    for j in range(0,h-2):
        resX = sum(sum(g * gris[i:i+1, j:j+1])) #vertical
        resY = sum(sum(g * gris[i:i + 1, j:j + 1]))  # Horizontal
        imBor[i + 1, j + 1] = np.sqrt((resX ** 2) + (resY ** 2))
for p in range(0, w):
    for q in range(0, h):
        if imBor[p,q] < 255:
            imBor[p, q] = 0

#Bordes 2
imBor2= np.zeros([w, h])
gx = ([-1,0,1])
gy = ([-1,0,1])

for i in range(0,w-3):
    for j in range(0,h-3):
        resX = sum(sum(gx * gris[i:i+3, j:j+3])) #vertical
        resY = sum(sum(gy * gris[i:i + 3, j:j + 3]))  # Horizontal
        imBor2[i + 1, j + 1] = np.sqrt((resX ** 2) + (resY ** 2))
for p in range(0, w):
    for q in range(0, h):
        if imBor2[p,q] < 255:
            imBor2[p, q] = 0

cv2.imshow("Original", pixeles)
cv2.imshow("Bordes1", imBor)
cv2.imshow("Bordes2", imBor2)
cv2.waitKey(0)
cv2.destroyAllWindows()