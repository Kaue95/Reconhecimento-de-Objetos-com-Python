import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image

# Configurações
IMG_SIZE = (128, 128)  # Tamanho das imagens
BATCH_SIZE = 16
EPOCHS = 30

# Caminho dos dados
DATASET_PATH = r"C:\Users\kaue9\PycharmProjects\pythonProject\dataset"

# Preparar os dados
def load_data(dataset_path, img_size):
    images, labels = [], []
    class_names = os.listdir(dataset_path)
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).resize(img_size)
            images.append(np.array(img))
            labels.append(class_map[class_name])

    return np.array(images), np.array(labels), class_map

# Carregar dados
images, labels, class_map = load_data(DATASET_PATH, IMG_SIZE)

# Normalizar e dividir dados
images = images / 255.0  # Normalizar
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Construir o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_map), activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# Testar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia no teste: {accuracy * 100:.2f}%")

# Salvar o modelo
model.save("modelo_reconhecimento_imagem.h5")

# Testar com uma imagem específica
def predict_image(model, img_path, class_map):
    img = Image.open(img_path).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = list(class_map.keys())[np.argmax(prediction)]
    return predicted_class

# Exemplo de teste
imagem_teste = r"C:\Users\kaue9\PycharmProjects\pythonProject\testimg\testesolar.jpg"
classe_prevista = predict_image(tf.keras.models.load_model("modelo_reconhecimento_imagem.h5"), imagem_teste, class_map)
print(f"A classe prevista é: {classe_prevista}")