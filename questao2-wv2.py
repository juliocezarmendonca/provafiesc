
import cv2
import numpy as np
import tifffile

# Carregar a imagem original do satélite WorldView-2 (8 canais)
image = tifffile.imread('8band_AOI_1_RIO_img46.tif')

# Separar as bandas espectrais
#coastalBlue, b, g, y, r, redEdge,nir1,nir2 = cv2.split(image)
band1, band2, band3, band4, band5, band6, band7, band8 = cv2.split(image)

# Normalizar os valores para o intervalo [0, 255]
band1 = cv2.normalize(band1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band2 = cv2.normalize(band2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band3 = cv2.normalize(band3, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band4 = cv2.normalize(band4, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band5 = cv2.normalize(band5, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band6 = cv2.normalize(band6, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band7 = cv2.normalize(band7, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
band8 = cv2.normalize(band8, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)




# Calcular NDVIs
ndvi1 = (band7 - band5) /  (band7 + band5 + 1e-9)
ndvi2 = (band8 - band6) /  (band8 + band6 + 1e-9)
ndvi3 = (band8 - band4) /  (band8 + band4 + 1e-9)
ndvi4 = (band6 - band1) /  (band6 + band1 + 1e-9)
ndvi5 = (band6 - band5) /  (band6 + band5 + 1e-9)

# Segmentar a vegetação
threshold_value = 0.2  # Limiar NDVI


# Aplicar a máscara de vegetação na imagem original
vegetation_mask1 = np.where(ndvi1 > threshold_value, 1, 0).astype(np.uint8)*255
vegetation_mask2 = np.where(ndvi2 > threshold_value, 1, 0).astype(np.uint8)*255
vegetation_mask3 = np.where(ndvi3 > threshold_value, 1, 0).astype(np.uint8)*255
vegetation_mask4 = np.where(ndvi4 > threshold_value, 1, 0).astype(np.uint8)*255
vegetation_mask5 = np.where(ndvi5 > threshold_value, 1, 0).astype(np.uint8)*255

# Definir os pixels da vegetação como verde
vegetation_mask1 = cv2.merge([np.zeros_like(vegetation_mask1), vegetation_mask1, np.zeros_like(vegetation_mask1)])
vegetation_mask2 = cv2.merge([np.zeros_like(vegetation_mask2), vegetation_mask2, np.zeros_like(vegetation_mask2)])
vegetation_mask3 = cv2.merge([np.zeros_like(vegetation_mask3), vegetation_mask3, np.zeros_like(vegetation_mask3)])
vegetation_mask4 = cv2.merge([np.zeros_like(vegetation_mask4), vegetation_mask4, np.zeros_like(vegetation_mask4)])
vegetation_mask5 = cv2.merge([np.zeros_like(vegetation_mask5), vegetation_mask5, np.zeros_like(vegetation_mask5)])

# Redimensionar as imagens para 256x256
resized_original = cv2.resize(cv2.merge([band5,band3,band2]), (1024, 1024))
resized_segmented1 = cv2.resize(vegetation_mask1, (1024, 1024))
resized_segmented2 = cv2.resize(vegetation_mask2, (1024, 1024))
resized_segmented3 = cv2.resize(vegetation_mask3, (1024, 1024))
resized_segmented4 = cv2.resize(vegetation_mask4, (1024, 1024))
resized_segmented5 = cv2.resize(vegetation_mask5, (1024, 1024))

# Converter a imagem RGB para o espaço de cor HSV
hsv_image = cv2.cvtColor(resized_original, cv2.COLOR_BGR2HSV)

# Definir os limites para a cor verde em HSV
lower_green = np.array([36, 25, 25])  # Limite inferior para o verde
upper_green = np.array([86, 255, 255])  # Limite superior para o verde

# Aplicar a máscara para segmentar a vegetação
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Inverter a máscara para ter a vegetação como 255 e o fundo como 0
mask = cv2.bitwise_not(mask)

# Redimensionar a máscara para 256x256
resized_mask = cv2.resize(mask, (256, 256))

# Exibir a máscara resultante
cv2.imshow('Segmentação da Vegetação', resized_mask)




# Exibir as imagens resultantes
cv2.imshow('Imagem RGB', resized_original)
#cv2.imshow('Imagem NDVI 1', resized_segmented1)
#cv2.imshow('Imagem NDVI 2', resized_segmented2)
#cv2.imshow('Imagem NDVI 3', resized_segmented3)
#cv2.imshow('Imagem NDVI 4', resized_segmented4)
#cv2.imshow('Imagem NDVI 5', resized_segmented5)
cv2.waitKey(0)
cv2.destroyAllWindows()
