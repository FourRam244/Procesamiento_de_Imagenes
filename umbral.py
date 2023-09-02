import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread("lenna.png")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color="gray")
plt.show()
umbral=140
for i in range (gris.shape[0]):
    for j in range (gris.shape[1]):
        if gris[i, j] < umbral:
            gris[i, j]=0
        else:
            gris[i, j]=255

cv2.imshow("umbral binario", gris)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (gris.shape[0]):
    for j in range (gris.shape[1]):
        if gris[i, j] > umbral:
            gris[i, j]=0
        else:
            gris[i, j]=255

cv2.imshow("umbral binario inverso", gris)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range (gris.shape[0]):
    for j in range (gris.shape[1]):
        if gris[i, j] > umbral:
            gris[i, j]=0
        else:
            gris[i, j]=255

cv2.imshow("umbral binario inverso", gris)
cv2.waitKey(0)
cv2.destroyAllWindows()
