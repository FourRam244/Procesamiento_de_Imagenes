import cv2
import numpy as np

imgBGR=cv2.imread("snk.jpg")
cv2.imshow("BGR", imgBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()

R = imgBGR[:,:,2]
G = imgBGR[:,:,1]
B = imgBGR[:,:,0]

matrizRGB = np.zeros((imgBGR.shape), dtype=np.uint8)

matrizRGB[:,:,0] = R
matrizRGB[:,:,1] = G
matrizRGB[:,:,2] = B

cv2.imshow("RGB", matrizRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("RGB")
print(matrizRGB)

#CONVERTIR HSV
HSV = cv2.cvtColor(matrizRGB, cv2.COLOR_RGB2HSV)
cv2.imshow("HSV libreria ", HSV)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
matrizHSV = np.ones((imgBGR.shape), dtype=np.uint8)
def convertirHSV(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0

    cmax = max(r.all(), g.all(), b.all())
    cmin = min(r.all(), g.all(), b.all())

    dif =np.logical_xor(cmax, cmin)

    if cmin==cmax:
        h=0
    elif cmax == r.all():
        h = (60*((g - b) / dif)+360) % 360
    elif cmax == g.all():
        h = (60 * ((b - r) / dif) + 120) % 360
    elif cmax == b.all():
        h = (60 * ((r - g) / dif) + 240) % 360

    if cmax ==0:
        s=0
    else:
        s=(dif/cmax)*100

    v = cmax *100

    matrizHSV[:, :, 0] = h
    matrizHSV[:, :, 1] = s
    matrizHSV[:, :, 2] = v

    cv2.imshow("HSV", matrizHSV)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return matrizHSV

print(convertirHSV(R, G, B))
'''
matrizHSV = np.ones((imgBGR.shape), dtype=np.uint8)
#Convetir a HSV
r = R/255
g = G/255
b = B/255
minimo = min(r.all(), g.all(), b.all())
maximo = max(r.all(), g.all(), b.all())
delta = np.logical_xor(maximo, minimo)
V = maximo
if (delta == 0.0):
    H=0.0
    S=0.0
else:
    S = delta/maximo
    R2 = (((maximo-r)/6.0)+(delta/2.0))/delta
    G2 = (((maximo - r) / 6.0) + (delta / 2.0)) / delta
    B2 = (((maximo - r) / 6.0) + (delta / 2.0)) / delta
if(r.all()==maximo):
    H=B - G
if(g.all()==maximo):
    H=(1.0/3.0)+R - B
if(b.all()==maximo):
    H=(2.0/3.0)+G - R
if(H.all()<0):
    H+=1
if(H.all()>1):
    H-=1


matrizHSV[:, :, 0] = H
matrizHSV[:, :, 1] = S
matrizHSV[:, :, 2] = V

cv2.imshow("HSV2", matrizHSV)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("HSV")
print(matrizHSV)


matrizCMYK = np.ones((imgBGR.shape), dtype=np.uint8)
#CONVERTIR CMYK
r = r / 255
g = g / 255
b = b / 255

vmax = max(r.all(), g.all(), b.all())
k = 1-vmax

C = (1 - r - k) / (1 - k)
M = (1 - g - k) / (1 - k)
Y = (1 - b - k) / (1 - k)

matrizCMYK[:, :, 0] = C
matrizCMYK[:, :, 1] = M
matrizCMYK[:, :, 2] = Y
cv2.imshow("CMYK", matrizCMYK)
cv2.waitKey(0)
cv2.destroyAllWindows()


