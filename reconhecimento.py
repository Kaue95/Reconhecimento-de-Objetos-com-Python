import cv2
import numpy as np
import tensorflow as tf

class_names = {
    0: "Energia Eolica",
    1: "Energia Hidroeletrica",
    2: "Energia Solar",
    # Adicione outros IDs e nomes conforme necessário
}

# Carregar o modelo treinado
model = tf.keras.models.load_model('modelo_reconhecimento_imagem.h5')  # Substitua pelo caminho do modelo salvo

# Função para pré-processar a imagem
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)  # Carregar a imagem com OpenCV
    img_resized = cv2.resize(img, target_size)  # Redimensionar para o tamanho esperado pelo modelo
    img_normalized = img_resized / 255.0  # Normalizar os valores dos pixels
    img_array = np.expand_dims(img_normalized, axis=0)  # Adicionar uma dimensão para lote
    return img, img_array


# Função para realizar a inferência e exibir a imagem com o reconhecimento
def run_inference(image_path, target_size=(128, 128)):
    # Carregar e pré-processar a imagem
    original_img, preprocessed_img = preprocess_image(image_path, target_size)

    # Fazer a predição
    prediction = model.predict(preprocessed_img)
    predicted_class_id = prediction.argmax(axis=1)[0]  # ID da classe com maior probabilidade
    confidence = prediction[0][predicted_class_id]     # Confiança da predição

    # Obter o nome da classe
    predicted_class_name = class_names.get(predicted_class_id, "Desconhecido")

    # Desenhar quadrado e rótulo na imagem original
    label = f"{predicted_class_name} ({confidence:.2f})"
    height, width, _ = original_img.shape
    square_size = min(height, width) // 4
    top_left_x = (width - square_size) // 2
    top_left_y = (height - square_size) // 2
    bottom_right_x = top_left_x + square_size
    bottom_right_y = top_left_y + square_size

    cv2.rectangle(original_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    cv2.putText(original_img, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar a imagem com o reconhecimento
    cv2.imshow("Resultado", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Teste com uma imagem
if __name__ == "__main__":
    image_path = "testimg/testehidrica2.jpg"  # Substitua pelo caminho da imagem que deseja testar
    run_inference(image_path)
