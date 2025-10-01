import cv2
import numpy as np


def sobel(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    # Carregue a imagem e converta para escala de cinza
    cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicando o filtro de Sobel nos eixos X e Y
    sobelx = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)  # Derivada X
    sobely = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)  # Derivada Y

    # Calculando a magnitude do gradiente
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = cv2.convertScaleAbs(magnitude)

    return image
