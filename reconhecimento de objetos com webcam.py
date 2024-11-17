import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model = load_model("modelo_reconhecimento_imagem.h5")

# Mapeamento de IDs para nomes das classes
class_names = {
    0: "Energia Eólica",
    1: "Energia Hidroelétrica",
    2: "Energia Solar",
    # Adicione outros IDs e nomes conforme necessário
}

# Função para prever a classe
def predict_class(frame, model):
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (128, 128))  # Ajuste o tamanho se necessário
    normalized_frame = resized_frame / 255.0  # Normalizar
    input_data = np.expand_dims(normalized_frame, axis=0)  # Adicionar dimensão batch
    predictions = model.predict(input_data)
    class_id = np.argmax(predictions)  # Pega o índice com maior probabilidade
    class_name = class_names.get(class_id, "Desconhecido")  # Converter ID para nome
    confidence = predictions[0][class_id]  # Pega a confiança
    return class_name, confidence

# Iniciar a webcam
cap = cv2.VideoCapture(0)  # 0 para webcam padrão

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam.")
        break

    # Realizar a previsão
    class_name, confidence = predict_class(frame, model)

    # Converter a imagem para tons de cinza para detecção de bordas
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar o retângulo ao redor do maior contorno
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if cv2.contourArea(largest_contour) > 500:  # Tamanho mínimo para evitar ruído
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{class_name}: {confidence * 100:.2f}%"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Exibir o vídeo
    cv2.imshow("Reconhecimento por Webcam", frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar os recursos
cap.release()
cv2.destroyAllWindows()
