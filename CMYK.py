import numpy as np
import matplotlib.pyplot as plt
import cv2

img = plt.imread("deku.jpg")   #Leer la imagen
rgb = cv2.cvtColor( img, cv2.COLOR_BGR2RGB) #Convertir a RGB
cv2.imshow("Original", rgb)
cv2.waitKey()
bgr = img.astype(float)/255.  #Cambiar a flotante y dividir entre 255

# Extraemos canales
with np.errstate(invalid='ignore', divide='ignore'):
	K = 1 - np.max(bgr, axis=2)
	C = (1-bgr[...,2] - K)/(1-K)
	M = (1-bgr[...,1] - K)/(1-K)
	Y = (1-bgr[...,0] - K)/(1-K)

# Convierta la imagen BGR de entrada al espacio de color CMYK
CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

# Dividir canales CMYK
Y, M, C, K = cv2.split(CMYK)

np.isfinite(C).all()
np.isfinite(M).all()
np.isfinite(K).all()
np.isfinite(Y).all()

cv2.imshow("Color C", C )
cv2.waitKey()
cv2.imshow("Color M", M )
cv2.waitKey()
cv2.imshow("Color Y", Y )
cv2.waitKey()
cv2.imshow("Color K", K )
cv2.waitKey()
cv2.destroyAllWindows()

