import cv2
import numpy as np
def grayscale_image(image):
    # convertir a gris
    gray_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # gray_image[i,j] = np.clip( (image[i,j,0] + image[i,j,1] + image[i,j,2] )/3, 0, 255) # using average method
            gray_image[i, j] = np.clip(0.07 * image[i, j, 0] + 0.72 * image[i, j, 1] + 0.21 * image[i, j, 2], 0,
                                       255)  # using luminosity method

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # mostrar imagen
    cv2.imshow('image', gray_image)
    cv2.waitKey(0)

    return gray_image


def initialize_x_filter(size):
    x_filter = np.zeros(size)
    h, w = size
    x_filter[h // 2][0] = -1
    x_filter[h // 2][-1] = 1

    return x_filter


def initialize_y_filter(size):
    y_filter = np.zeros(size)
    h, w = size
    y_filter[0][h // 2] = -1
    y_filter[-1][h // 2] = 1

    return y_filter


def padding(image):
    padded_image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))

    return padded_image


def conv2d(image, ftr):
    s = ftr.shape + tuple(np.subtract(image.shape, ftr.shape) + 1)
    sub_image = np.lib.stride_tricks.as_strided(image, shape=s, strides=image.strides * 2)
    return np.einsum('ij,ijkl->kl', ftr, sub_image)


if __name__ == "__main__":
    # lee imagen
    image = cv2.imread('star.jpg')

    # convertir escala de grises
    gray_image = grayscale_image(image)

    x_filter = initialize_x_filter((3, 3))
    y_filter = initialize_y_filter((3, 3))

    # convulucionar imagen con el filtro x
    I_x = conv2d(padding(gray_image), x_filter)

    # convulucionar imagen con el filtro y
    I_y = conv2d(padding(gray_image), y_filter)

    # calcular magnitud de gradiente
    G = np.sqrt(np.power(I_x, 2) + np.power(I_y, 2))

    # aplicar en los diferentes imagenes
    G = np.where(G > 150, G, 0)
    cv2.imwrite("Gradiente.jpg", G)
    # display the result
    cv2.imshow('Gradiente', G)
    cv2.waitKey(0)