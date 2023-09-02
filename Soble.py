
import cv2
import numpy as np
#Sobel
pixeles = cv2.imread('tanjiro.jpg')
gris= cv2.cvtColor(pixeles, cv2.COLOR_BGR2GRAY)
w, h = gris.shape
imSB= np.zeros([w, h])
gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
for i in range(0,w-2):
    for j in range(0,h-2):
        resX = sum(sum(gx * gris[i:i+3, j:j+3])) #vertical
        resY = sum(sum(gy * gris[i:i + 3, j:j + 3]))  # Horizontal
        imSB[i + 1, j + 1] = np.sqrt((resX ** 2) + (resY ** 2))
for p in range(0, w):
    for q in range(0, h):
        if imSB[p,q] < 255:
            imSB[p, q] = 0

cv2.imshow("Imagen Original", pixeles)
#cv2.imshow("Sobel", imSB)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Prewit
imPr= np.zeros([w, h])
gx = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
gy = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
for i in range(0,w-2):
    for j in range(0,h-2):
        resX = sum(sum(gx * gris[i:i+3, j:j+3])) #vertical
        resY = sum(sum(gy * gris[i:i + 3, j:j + 3]))  # Horizontal
        imPr[i + 1, j + 1] = np.sqrt((resX ** 2) + (resY ** 2))
for p in range(0, w):
    for q in range(0, h):
        if imPr[p,q] < 255:
            imPr[p, q] = 0

cv2.imshow("Sobel", imSB)
cv2.imshow("Prewit", imPr)
cv2.waitKey(0)
cv2.destroyAllWindows()
