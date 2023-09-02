import math
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.image as mpimg
import skimage.exposure as exposure

img = cv2.imread("deku.jpg")
img1 = cv2.imread("m2.png")

opcion = int(input("Seleccione una opción: \n1) Filtro gaussiano \n2) Filtro gradiente \n3) Filtro paso altos y paso bajos"
                   "\n4) Filtro Laplaciano \n5) Filtro Robert \n6) Filtro Sobel \n0) Salir \n"))
while opcion != 0:
    if opcion == 1:
        print("=" * 100)
        print("Seleccionaste Filtro gaussiano")


        def gausskernel(size, k, sigma):
            gausskernel = np.zeros((size, size), np.float32)
            for i in range(size):
                for j in range(size):
                    norm = math.pow(i - k, 2) + pow(j - k, 2)
                    gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2))) / 2 * math.pi * pow(sigma, 2)
            sum = np.sum(gausskernel)
            kernel = gausskernel / sum
            return kernel


        def mygaussFilter(img_gray, kernel):
            h, w = img_gray.shape
            k_h, k_w = kernel.shape
            for i in range(int(k_h / 2), h - int(k_h / 2)):
                for j in range(int(k_h / 2), w - int(k_h / 2)):
                    sum = 0
                    for k in range(0, k_h):
                        for l in range(0, k_h):
                            sum += img_gray[i - int(k_h / 2) + k, j - int(k_h / 2) + l] * kernel[k, l]
                    img_gray[i, j] = sum
            return img_gray


        if __name__ == '__main__':
            img = cv2.imread("deku.jpg")
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_g = img_gray.copy()
            kernel = np.array([[1, 1, 2, 2, 2, 1, 1],
                               [1, 2, 2, 4, 2, 2, 1],
                               [2, 2, 4, 8, 4, 2, 2],
                               [2, 4, 8, 16, 8, 4, 2],
                               [2, 2, 4, 8, 4, 2, 2],
                               [1, 2, 2, 4, 2, 2, 1],
                               [1, 1, 2, 2, 2, 1, 1]])
            print(kernel)
            k = 3
            size = 2 * k + 1
            kernel = gausskernel(size, k, 1.5)

            img_B, img_G, img_R = cv2.split(img)
            img_gauss_B = mygaussFilter(img_B, kernel)
            img_gauss_G = mygaussFilter(img_G, kernel)
            img_gauss_R = mygaussFilter(img_R, kernel)
            img_gauss = cv2.merge([img_gauss_B, img_gauss_G, img_gauss_R])
            cv2.imshow("Original", img)
            cv2.imshow("FiltroG", img_gauss)
            cv2.waitKey(0)


        print("=" * 100)
    elif opcion == 2:
        print("=" * 100)
        print("Seleccionaste Filtro gradiente")

        img = cv2.imread('deku.jpg')
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Prewitt operator
        kernelN = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
        kernelO = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)

        kernelE = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
        kernelS = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)

        kernelNO = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=int)
        kernelSE = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=int)

        kernelSO = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=int)
        kernelNOR = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=int)

        N = cv2.filter2D(grayImage, cv2.CV_16S, kernelN)
        O = cv2.filter2D(grayImage, cv2.CV_16S, kernelO)

        E = cv2.filter2D(grayImage, cv2.CV_16S, kernelE)
        S = cv2.filter2D(grayImage, cv2.CV_16S, kernelS)

        NO = cv2.filter2D(grayImage, cv2.CV_16S, kernelNO)
        SE = cv2.filter2D(grayImage, cv2.CV_16S, kernelSE)

        SO = cv2.filter2D(grayImage, cv2.CV_16S, kernelSO)
        NOR = cv2.filter2D(grayImage, cv2.CV_16S, kernelNOR)

        # Pasar a uint8
        absN = cv2.convertScaleAbs(N)
        absO = cv2.convertScaleAbs(O)

        absE = cv2.convertScaleAbs(E)
        absS = cv2.convertScaleAbs(S)

        absNO = cv2.convertScaleAbs(NO)
        absSE = cv2.convertScaleAbs(SE)

        absSO = cv2.convertScaleAbs(SO)
        absNOR = cv2.convertScaleAbs(NOR)
        # Combinar dos imagenes
        # Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5,0)

        cv2.imshow("Prewitt Norte", absN)
        cv2.imshow("Prewitt Oeste", absO)

        cv2.imshow("Prewitt Este", absE)
        cv2.imshow("Prewitt Sur", absS)

        cv2.imshow("Prewitt Noroeste", absNO)
        cv2.imshow("Prewitt Sureste", absSE)

        cv2.imshow("Prewitt Suroeste", absSO)
        cv2.imshow("Prewitt Noroeste1", absNOR)

        cv2.imshow("Original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("=" * 100)

    elif opcion == 3:
        print("=" * 100)
        print("Seleccionaste Filtro paso altos y bajos")

        img = cv2.imread("deku.jpg", cv2.IMREAD_GRAYSCALE)

        kernel_3x3 = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])

        kernel_5x5 = np.array([[0, -1, -1, -1, 0],
                               [-1, 2, -4, 2, -1],
                               [-1, -4, 13, -4, -1],
                               [-1, 2, -4, 2, -1],
                               [0, -1, -1, -1, 0]])

        lowPassFilter = np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]])

        lowPassFilter1 = np.array([[1,1,1,1,1],
                               [1,4,4,4,1],
                               [1,4,12,4,1],
                               [1,4,4,4,1],
                               [1,1,1,1,1]])

        k3 = ndimage.convolve(img, kernel_3x3)
        k5 = ndimage.convolve(img, kernel_5x5)

        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        g_hpf = img - blurred

        cv2.imshow("3x3", k3)
        cv2.imshow("5x5", k5)
        cv2.imshow("Paso bajo", g_hpf)
        cv2.imshow("Original", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


        print("=" * 100)

    elif opcion == 4:
        print("=" * 100)
        print("Seleccionaste Filtro Laplaciano")
        img = cv2.imread("deku.jpg", 0)

        lap_fil = np.array([[1, 1, 6],
                            [0, 8, 1],
                            [7, 1, 0]])

        dst1 = cv2.filter2D(img, ddepth=cv2.CV_16S, kernel=lap_fil)

        cv2.imshow('Original', img)
        cv2.imshow('Filtro Laplacinao', dst1 + 500)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("=" * 100)

    elif opcion == 5:
        print("=" * 100)
        print("Seleccionaste Filtro Robert")
        roberts = np.array([[0, -1],
                                    [-1, 0]])

        img = cv2.imread("deku.jpg", 0).astype('float64')
        img /= 255.0

        FiltroRobert = ndimage.convolve(img, roberts)

        cv2.imshow('Original', img)
        cv2.imshow("Filtro Robert", FiltroRobert)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("=" * 100)

    elif opcion == 6:
        print("=" * 100)
        print("Seleccionaste Filtro Sobel")

        img = cv2.imread('deku.jpg').astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.3, 1.3)

        # definir los kernels Sobel X e Y (correlación)
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

        Ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])

        # aplicar correlaciones y normalizar por suma de valores absolutos de elementos
        sobelx = ndimage.correlate(blur, Kx)
        sobely = ndimage.correlate(blur, Ky)
        # normalice opcionalmente al rango de 0 a 255 para una visualización adecuada y guardar como datos de 8 bits.
        sobelx_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        sobely_norm = exposure.rescale_intensity(sobelx, in_range='image', out_range=(0, 255)).clip(0, 255).astype(np.uint8)
        # sumar y sacar raiz cuadrada
        sobel_magnitud = np.sqrt(np.square(sobelx) + np.square(sobely))
        # normalizar en un rango de 0 a 255 y recortar negativos
        sobel_magnitud = exposure.rescale_intensity(sobel_magnitud, in_range='image', out_range=(0, 255)).clip(0,255).astype(np.uint8)

        cv2.imshow('sobelx_norm', sobelx_norm)
        cv2.imshow('sobely_norm', sobely_norm)
        cv2.imshow('sobel_magnitude', sobel_magnitud)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("=" * 100)

    else:
        print("Opcion inválida prro")

    opcion = int(input("Seleccione una opción: \n1) Filtro gaussiano \n2) Filtro gradiente \n3) Filtro paso altos \n4) Filtro paso bajos"
                   "\n5) Filtro Laplaciano \n6) Filtro Robert \n7) Filtro Sobel \n0) Salir \n"))