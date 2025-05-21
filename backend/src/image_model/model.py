from keras import layers, models, Input
from config.logger_config import configure_logger

# Configuration du logger
logger = configure_logger()

def build_simple_cnn(input_shape=(128, 128, 3), num_classes=1):
    model = models.Sequential([
    Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
