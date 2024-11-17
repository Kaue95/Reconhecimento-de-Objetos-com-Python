from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Gerador de imagens
datagen = ImageDataGenerator()
train_generator = datagen.flow_from_directory(r'C:\Users\kaue9\PycharmProjects\pythonProject\dataset', target_size=(128, 128))

# Exibir o mapeamento
print(train_generator.class_indices)
