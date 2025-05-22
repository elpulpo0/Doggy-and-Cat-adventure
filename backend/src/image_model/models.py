from keras import layers, models, Input

def build_simple_cnn(input_shape=(128,128,3), num_classes=2):
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

def build_complex_cnn(input_shape=(128,128,3), num_classes=2):
    model = models.Sequential([
        Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
