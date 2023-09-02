import numpy as np
import cv2
import matplotlib.pyplot as plt

r2 = cv2.imread("eva01.jpg")
imagenl= cv2.resize(r2, None, fx=3/4, fy=3/4, interpolation=cv2.INTER_AREA)
imagen = cv2.cvtColor(imagenl, cv2.COLOR_BGR2GRAY)
cv2.imshow("imagen ", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

# primero realiazamos una erosion
matriz = np.ones((23, 23), np.uint8)
erosion = cv2.erode(imagen, matriz, iterations=1)
# mostramos la imagen de erosin
cv2.imshow("Erosion", erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# una vez realizada la erosion
# hacemos la dilatacion
# con esa operacion realizamos la apertura
apertura = cv2.dilate(erosion, matriz, iterations=1)

cv2.imshow("apertura", apertura)
cv2.waitKey(0)
cv2.destroyAllWindows()

# matriz para guardar resultados de la resta
resta = np.zeros((imagen.shape), dtype=np.uint8)
# vamos a restar la imagen original menos la apertura
for i in range(imagen.shape[0]):
    for j in range(imagen.shape[1]):
        resta[i, j] = imagen[i, j] - apertura[i, j]
        if resta[i, j] < 0:
            resta[i, j] = 0

# mostramos la resta
cv2.imshow("Operacion resta", resta)
cv2.waitKey(0)
cv2.destroyAllWindows()

#umbral

hist = cv2.calcHist([resta], [0], None, [256], [0, 256])
plt.plot(hist, color="gray")
plt.show()
umbral=90
for i in range (resta.shape[0]):
    for j in range (resta.shape[1]):
        if resta[i, j] < umbral:
            resta[i, j]=0
        else:
            resta[i, j]=255

cv2.imwrite("umbral.png", resta)
cv2.imshow("final", resta)
cv2.waitKey(0)
cv2.destroyAllWindows()