import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('/media/williancaddd/CODES/fiotec/AETrampa/aedes_eggs_data/images/173-4.jpeg')

# Converter para escala de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar filtro gaussiano para suavizar a imagem
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detectar bordas utilizando o algoritmo Canny
edges = cv2.Canny(blurred, 50, 150)

# Encontrar contornos na imagem
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Criar uma máscara com o mesmo tamanho da imagem original
mask = np.zeros_like(gray)

# Preencher a máscara com os contornos do objeto
cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

# Aplicar a máscara à imagem original
segmented_image = cv2.bitwise_and(image, image, mask=mask)

# Usar matplotlib para mostrar as imagens
plt.figure(figsize=(10,5))

# Mostrar a imagem com contornos detectados
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2), cv2.COLOR_BGR2RGB))
plt.title('Contornos Detectados')

# Mostrar a imagem segmentada
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Objeto Segmentado')

plt.show()
