import cv2
import math
import numpy as np
img = cv2.imread("eri.jpg")
# Procesamiento en escala de grises
h=img.shape[0]
w=img.shape[1]
img1=np.zeros((h,w),np.uint8)
for i in range(h):
    for j in range(w):
        img1[i,j]=0.144*img[i,j,0]+0.587*img[i,j,1]+0.299*img[i,j,1]
# Calcular el núcleo de convolución gaussiano
sigma=1.0
gausskernel=np.zeros((5,5),np.float32)
for i in range (5):
    for j in range (5):
        norm=math.pow(i-1,2)+pow(j-1,2)
        gausskernel [i, j] = math.exp (-norm / (2 * math.pow (sigma, 2))) # Encuentra convolución gaussiana
    sum = np.sum (gausskernel) # sum
    kernel = gausskernel / sum # normalización
print("kernel",kernel)
 # Filtro gaussiano
img2=np.zeros((h,w),np.uint8)
kernel = gausskernel**(3) # Calcular kernel de convolución gaussiana
for i in range (1,h-1):
    for j in range (1,w-1):
        sum=0
        for k in range(-1,2):
            for l in range(-1,2):
                sum+=img1[i + k, j + l] * kernel[k + 1, l + 1] # filtro gaussiano
            img2[i,j]=sum
#Soble
w2,h2 = img2.shape
imSB= np.zeros([w2, h2])
gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
for i in range(0,w2-2):
    for j in range(0,h2-2):
        resX = np.sum(np.sum(gx * img2[i:i+3, j:j+3])) #vertical
        resY = np.sum(np.sum(gy * img2[i:i + 3, j:j + 3]))  # Horizontal
        imSB[i + 1, j + 1] = np.sqrt((resX ** 2) + (resY ** 2))
for p in range(0, w2):
    for q in range(0, h2):
        if imSB[p,q] < 255:
            imSB[p, q] = 0
cv2.imshow("Original",img)
cv2.imshow("gris",img1)
cv2.imshow("Filtro Gaussiano",img2)
cv2.imshow("Canny", imSB)
cv2.waitKey(0)
cv2.destroyAllWindows()



