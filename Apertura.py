import cv2
import numpy as np
print("Elija el tamaño de la matriz")
print("x:")
x = int(input())
print("y:")
y = int(input())
filt = np.ones((x, y), np.uint8)

#Erosion
pixeles = cv2.imread('eri.jpg') #Leemos imagen
gris= cv2.cvtColor(pixeles, cv2.COLOR_BGR2GRAY) #Convertimos a gris
(thresh,bina) = cv2.threshold(gris,127,255,cv2.THRESH_BINARY) #Imagen a binario
cv2.imshow("Binario",bina) #Mostramos imagen binaria

#filt=np.array([(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1),(1,1,1,1,1,1,1)]) #Filtro
S = bina.shape #tamaño del binario
F= filt.shape #Tamaño fiiltro
bina= bina/255
R= S[0]+F[0]-1
C= S[1]+F[1]-1
N= np.zeros((R,C)) #Matriz de zeros

for i in range(S[0]):
    for j in range(S[1]):
        N[i+1,j+1]=bina[i,j]

for i in range(S[0]):
    for j in range(S[1]):
        k=N[i:i+F[0],j:j+F[1]]
        rs= (k==filt)
        final = np.all(rs==True) #Tomamos "all"
        if final:
            bina[i,j]=1
        else:
            bina[i,j]=0

cv2.imshow("Erosion",bina) #Mostramos imagen Erosionada


#Dilatacion
for i in range(S[0]):
    for j in range(S[1]):
        N[i+1,j+1]=bina[i,j]

for i in range(S[0]):
    for j in range(S[1]):
        k=N[i:i+F[0],j:j+F[1]]
        rs= (k==filt)
        final = np.any(rs==True) #Tomamos "any"
        if final:
            bina[i,j]=1
        else:
            bina[i,j]=0

cv2.imshow("Apertura",bina) #Mostramos imagen Dilatada
cv2.imwrite("apertura.png",bina)
cv2.waitKey(0)


