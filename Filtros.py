import cv2
import numpy as np
import scipy.ndimage


opcion = int(input("Seleccione una opción: \n1) Filtro de mediana aritmetica \n2) Filtro mediana geométrica \n3) Filtro media harmonica"
                   "\n4) Filtro mediana \n5) Filtro máxima \n6) Filtro mínima \n7) Filtro Punto medio \n8) Filtro Alfa recortado"
                   "\n9) Filtro local adoptivo \n10) Filtro mediana adativa \n11) Filtro Gaussiano"
                   "\n12) Transformación Gamma \n13) Estiramiento del contraste \n14) Destacar niveles de gris \n15) Planos de bits \n"))

img = cv2.imread("ruidoR.jpg",0)
'-------------------------------------------------------------------'
"Imagen de Mediana geometrica"
imagen = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE).astype(float)
rows, cols = imagen.shape[:2]
ksize = 5
'-------------------------------------------------------------------'
while opcion != 0:
    if opcion == 1:
        print("="*100)
        print("Seleccionaste Filtro de media aritmetica")
        mediaA = cv2.blur(img, (3, 3))
        cv2.imshow("Media aritmetica", mediaA)
        cv2.imshow("Img original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 2:
        print("Seleccionaste Filtro de mediana geometrica")
        print("="*100)

        geomean2 = np.uint8(np.exp(cv2.boxFilter(np.log(imagen), -1, (ksize, ksize))))
        cv2.imshow('Mediana geometrica', geomean2)
        cv2.imshow("Img original", img)
        cv2.waitKey()

        print("="*100)
    elif opcion == 3:
        print('='*100)
        print('Seleccionaste filtro de media harmonica')

        kernel = np.ones([3, 3])
        epsilon = 1e-8
        img_h = img.shape[0]
        img_w = img.shape[1]
        m, n = kernel.shape[:2]
        order = kernel.size
        padding_h = int((m - 1) / 2)
        padding_w = int((n - 1) / 2)
        # Este método de llenado, sí, los núcleos pares o impares se pueden llenar correctamente
        image_pad = np.pad(img.copy(), ((padding_h, m - 1 - padding_h), (padding_w, n - 1 - padding_w)), mode="edge")
        image_mean = img.copy()
        # Aquí debe especificar el tipo de datos, pero la designación es uint64 quizás float64,
        # pero el resultado no es correcto, en su lugar, multiplique 1.0, también es float64,
        # pero hace que el resultado sea rig
        # Añadir épsilon
        for i in range(padding_h, img_h + padding_h):
            for j in range(padding_w, img_w + padding_w):
                temp = np.sum(
                    1 / (image_pad[i - padding_h:i + padding_h + 1, j - padding_w:j + padding_w + 1] * 1.0 + epsilon))
                image_mean[i - padding_h][j - padding_w] = order / temp

        cv2.imshow("Img media harmonica", image_mean)
        cv2.imshow("Img original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('='*100)

    elif opcion == 4:
        print("="*100)
        print("Seleccionaste Filtro de mediana")
        img11 = cv2.imread("noise.jpg")
        mediana = cv2.medianBlur(img11,3)
        cv2.imshow("Imagen mediana",mediana)
        cv2.imshow("Img original", img11)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 5:
        print("="*100)
        print("Seleccionaste Filtro de Máximo")
        img = cv2.imread('ibai.jpg')
        n = 5
        # Kernel
        size = (n, n)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)
        # Aplica el filtro mínimo con kernel NxN
        imgResult = cv2.dilate(img, kernel)

        cv2.imshow('Original', img)
        cv2.imshow('Img maximo' + str(n), imgResult)
        print(imgResult)
        cv2.waitKey(0)
        print("="*100)

    elif opcion == 6:
        print("="*100)
        print("Seleccionaste Filtro Minimo")

        img = cv2.imread('ibai.jpg')
        n = 7
        # Kernel
        size = (n, n)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)

        # Aplica el filtro mínimo con kernel NxN
        imgResult = cv2.erode(img, kernel)
        cv2.imshow('Original', img)
        cv2.imshow('Img minima', imgResult)
        cv2.waitKey(0)

        print("="*100)

    elif opcion == 7:
        print("="*100)
        print("Seleccionaste Filtro Punto medio")

        a = cv2.imread("ibai.jpg")
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        k = np.ones((5, 5)) / 25
        b = scipy.ndimage.convolve(a, k)
        cv2.imshow("Originsl",a)
        cv2.imshow("FiltroPuntoM", b)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("="*100)

    elif opcion == 11:
        print("="*100)
        print("Seleccionaste Filtro Gaussiano")
        img = cv2.imread("ImgRuidoP.jpg")
        FGauss = cv2.GaussianBlur(img,(3,3),0)
        cv2.imshow("Imagen gaussiana", FGauss)
        cv2.imshow("Img original", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 12:
        print("="*100)
        print("Seleccionaste Filtro transformación Gamma")

        img = cv2.imread('ibai.jpg')
        porcentaje = 2.9
        for gamma in [porcentaje]:
            gamma = np.array(255 * (img / 255) ** gamma, dtype='uint8')
            cv2.imshow('Transformación Gamma', gamma)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 13:
        print("="*100)
        print("Seleccionaste Filtro Estiramiento del contraste")

        img = cv2.imread('ibai.jpg')
        original = img.copy()

        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img = cv2.LUT(img, table)
        cv2.imshow("original", original)
        cv2.imshow("Output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 14:
        print("="*100)
        print("Seleccionaste Filtro Destacar niveles de gris")

        img = cv2.imread("ibai.jpg", 0)
        l = 156
        rows, columns = img.shape
        img2 = np.zeros((rows, columns), dtype=np.uint8)
        for x in range(rows):
            for y in range(columns):
                img2[x, y] = (l - 1) - img[x, y]
        cv2.imshow('Original', img)
        cv2.imshow('Salida', img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("="*100)

    elif opcion == 15:
        print("="*100)
        print("Seleccionaste Filtro Plano de bits")
        img = cv2.imread('ibai.jpg', 0)

        # Itera sobre cada píxel y cambia el valor del píxel a binario usando np.binary_repr() y guárdalo en una lista.
        lst = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                lst.append(np.binary_repr(img[i][j], width=8))  # width = no. of bits

        # Tenemos una lista de cadenas donde cada cadena representa un valor de píxel binario. Para extraer planos de bits
        # necesitamos iterar sobre las cadenas y almacenar los caracteres correspondientes a los planos de bits en listas.
        # Multiplique por 2^(n-1) y remodele para reconstruir la imagen de bits.
        eight_bit_img = (np.array([int(i[0]) for i in lst], dtype=np.uint8) * 128).reshape(img.shape[0], img.shape[1])
        cv2.imshow('Original', img)
        cv2.imshow('Salida', eight_bit_img)
        cv2.waitKey(0)
        print("="*100)

    else:
        print("Por favor digita una opcion correcta")

    opcion = int(input(
        "Seleccione una opción: \n1) Filtro de mediana aritmetica \n2) Filtro mediana geométrica \n3) Filtro media harmonica"
        "\n4) Filtro mediana \n5) Filtro máxima \n6) Filtro mínima \n7) Filtro Punto medio \n8) Filtro Alfa recortado"
        "\n9) Filtro local adoptivo \n10) Filtro mediana adativa \n11) Filtro Gaussiano"
        "\n12) Transformación Gamma \n13) Estiramiento del contraste \n14) Destacar niveles de gris \n15) Planos de bits \n"))