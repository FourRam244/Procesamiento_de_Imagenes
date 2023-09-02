
import cv2
import numpy as np

#Leemos imagen a escala de grises
img = cv2.imread('paloma.png',0)

# Threshold the image
ret,img = cv2.threshold(img, 127, 255, 0)

# Paso 1: Crea un esqueleto vacío
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Obtenga un kernel en forma de cruz
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repita los pasos 2-4
while True:
    #Step 2: Abrimos imagen
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    # Paso 3: Restamos de la imagen original
    temp = cv2.subtract(img, open)
    # Paso 4: erosiona la imagen original y refina el esqueleto
    eroded = cv2.erode(img, element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    # Paso 5: si no quedan píxeles blancos, es decir, la imagen se ha erosionado por completo, salga del ciclo
    if cv2.countNonZero(img)==0:
        break

# Mostramos esqueleto
cv2.imshow("Skeleton",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()