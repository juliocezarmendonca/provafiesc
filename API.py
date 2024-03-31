from flask import Flask, request, jsonify
import base64
from PIL import Image
import io
import cv2
import numpy as np
import tifffile

app = Flask(__name__)


@app.route('/airport', methods=['POST'])
def airport():
    try:
        # Decodificar a imagem Base64
        data = request.json
        image_base64 = data['imagem']
        image_data = base64.b64decode(image_base64)
        #image = Image.open(io.BytesIO(image_data)
        image = cv2.imread(io.BytesIO(image_data))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 185, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Encontrar o maior contorno baseado em sua área
            max_contour = max(contours, key=cv2.contourArea)
            # Calcular a bounding box para o maior contorno
            x, y, w, h = cv2.boundingRect(max_contour)
            # Desenhar a bounding box na imagem original
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Codificar a imagem processada de volta para Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Retornar a imagem processada como Base64
        return jsonify({'Silo': processed_image_base64})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

# Rotas 2 e 3 serão semelhantes, basta implementar a lógica desejada para cada uma

@app.route('/noise', methods=['POST'])
def noise():
    # Implemente a lógica para o método da rota 2 aqui
    try:
        data = request.json
        image_base64 = data['imagem']
        image_data = base64.b64decode(image_base64)
        #image = Image.open(io.BytesIO(image_data)
        image = cv2.imread(io.BytesIO(image_data))
        
        denoised_image = cv2.medianBlur(image_with_noise, 5)

        
        # Codificar a imagem processada de volta para Base64
        buffered = io.BytesIO()
        denoised_image.save(buffered, format="JPEG")
        processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Retornar a imagem processada como Base64
        return jsonify({'Denoised': processed_image_base64})
        


    except Exception as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/satelite', methods=['POST'])
def satelite():
        try:
            data = request.json
            image_base64 = data['imagem']
            image_data = base64.b64decode(image_base64)
            #image = Image.open(io.BytesIO(image_data)
            image = tifffile.imread(io.BytesIO(image_data))
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

            # Redimensionar as imagens para 256x256 IMAGEM RGB
            resized_original = cv2.resize(cv2.merge([band5,band3,band2]), (256, 256))
            # Converter a imagem RGB para o espaço de cor HSV
            hsv_image = cv2.cvtColor(resized_original, cv2.COLOR_BGR2HSV)

            # Definir os limites para a cor verde em HSV
            lower_green = np.array([60, 108, 0])  # Limite inferior para o verde
            upper_green = np.array([123, 173, 98])  # Limite superior para o verde
             # Aplicar a máscara para segmentar a vegetação
            mask = cv2.inRange(hsv_image, lower_green, upper_green)
            # Inverter a máscara para ter a vegetação como 255 e o fundo como 0
            mask = cv2.bitwise_not(mask)
            # Redimensionar a máscara para 256x256
            resized_mask = cv2.resize(mask, (256, 256))
            buffered = io.BytesIO()
            resized_mask.save(buffered, format="JPEG")
            processed_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Retornar a imagem processada como Base64
            return jsonify({'VegetacaoSegmentada': processed_image_base64})
        

            
            
        except Exception as e:
            return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
