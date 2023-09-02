import cv2
import numpy as np
import matplotlib.pyplot as plt
#Leer imagen
image = cv2.imread("p2.jpg")
cv2.imshow("Imagen Original", image)
#Convertir a RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# remodelar la imagen a una matriz 2D de píxeles y 3 valores de color(RGB)
pixel_values = image.reshape((-1, 3))
# convertir a float
pixel_values = np.float32(pixel_values)
# imprime salida de valores de los pixeles
print(pixel_values.shape)
# definir criterios de parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# número de grupos (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# convertir de nuevo a valores de 8 bits
centers = np.uint8(centers)
# aplanar la matriz de etiquetas
labels = labels.flatten()
# convertir todos los píxeles al color de los centroides
segmented_image = centers[labels.flatten()]
# remodelar de nuevo a la dimensión de la imagen original
segmented_image = segmented_image.reshape(image.shape)
#Mostrar la imagen

plt.imshow(segmented_image)
plt.show()

# deshabilite solo el grupo número 2 (convierta el píxel en negro)
masked_image = np.copy(image)
# convertir a la forma de un vector de valores de píxeles
masked_image = masked_image.reshape((-1, 3))
# color (es decir, grupo) para deshabilitar
cluster = 2
masked_image[labels == cluster] = [0, 0, 0]
# volver a convertir a la forma original
masked_image = masked_image.reshape(image.shape)
# Mostrar la imagen
plt.imshow(masked_image)
plt.show()