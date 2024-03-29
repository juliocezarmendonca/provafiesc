import cv2

# Carregar a imagem com ruído
image_with_noise = cv2.imread('airport_gray_noisy.png', cv2.IMREAD_GRAYSCALE)

# Aplicar filtro de mediana para remoção de ruído
denoised_image = cv2.medianBlur(image_with_noise, 5)  # Tamanho do kernel: 5x5

# Exibir a imagem original e a imagem denoised
cv2.imshow('Imagem com Ruido', image_with_noise)
cv2.imshow('Imagem Sem Ruido', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
