import cv2
import numpy as np
# Carregar a imagem
image = cv2.imread('airport.png')

# Converter a imagem para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar tecnica threshold com intervalo de histograma de 185 a 255
_, binary_image = cv2.threshold(gray_image, 185, 255, cv2.THRESH_BINARY)

# Encontrar os contornos na imagem binarizada invertida
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Desenhar uma bounding box em volta do maior contorno
if len(contours) > 0:
    # Encontrar o maior contorno baseado em sua área
    max_contour = max(contours, key=cv2.contourArea)
    # Calcular a bounding box para o maior contorno
    x, y, w, h = cv2.boundingRect(max_contour)
    # Desenhar a bounding box na imagem original
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow('Silo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
