

import numpy as np
import cv2
ImgRGB = cv2.imread("eva01.jpg")   #Leer imagen
cv2.imshow("RGB", ImgRGB)   #Convetir a RGB
cv2.waitKey(0)
cv2.destroyAllWindows()

def RGB2HSV(ImgRGB):
    RGB_normalized = ImgRGB / 255.0 # Normalizar valores de 0.0 - 1.0 (float64)
    R = RGB_normalized[:, :, 0]    # Separa canales
    G = RGB_normalized[:, :, 1]
    B = RGB_normalized[:, :, 2]
    
    v_max = np.max(RGB_normalized, axis=2) # Calcular maximo
    v_min = np.min(RGB_normalized, axis=2) # Calcular minimo
    C = v_max - v_min                      # Calcular chroma
    
    hue_defined = C > 0
    
    r_is_max = np.logical_and(R == v_max, hue_defined) # Los calculos depende del maximo
    g_is_max = np.logical_and(G == v_max, hue_defined)
    b_is_max = np.logical_and(B == v_max, hue_defined)
    
    H = np.zeros_like(v_max)   # Calcular matiz
    H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
    H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
    H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4
    
    H[r_is_max] = H_r
    H[g_is_max] = H_g
    H[b_is_max] = H_b
    H *= 60
    
    V = v_max   # Calcular el valor
    
    sat_defined = V > 0
    
    S = np.zeros_like(v_max)  # Calcular la saturacion
    S[sat_defined] = C[sat_defined] / V[sat_defined]
    
    return np.dstack((H, S, V))

HSV = RGB2HSV(ImgRGB)   # Convertir de RGB a HSV

HUE = HSV[:, :, 0]   # Separar los atributos
SAT = HSV[:, :, 1]
VAL = HSV[:, :, 2]

cv2.imshow("Atributo H", HUE)
cv2.waitKey(0)
cv2.imshow("Atributo S", SAT)
cv2.waitKey(0)
cv2.imshow("Atributo V", VAL)
cv2.waitKey(0)
cv2.destroyAllWindows()

