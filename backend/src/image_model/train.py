from keras.preprocessing.image import ImageDataGenerator
from model import build_simple_cnn

def train_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        '../data/images/train', target_size=(128, 128),
        class_mode='binary', subset='training')

    val_gen = datagen.flow_from_directory(
        '../data/images/train', target_size=(128, 128),
        class_mode='binary', subset='validation')

    model = build_simple_cnn()
    model.fit(train_gen, epochs=5, validation_data=val_gen)
    model.save('models/cnn_image_model.h5')

if __name__ == '__main__':
    train_model()
