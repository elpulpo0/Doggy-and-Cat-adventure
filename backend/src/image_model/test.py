import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

def predict_on_test_images(model_path='models/cnn_image_model.h5', test_dir='../data/images/test'):
    model = load_model(model_path)
    results = []

    for fname in sorted(os.listdir(test_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, fname)
            img = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Batch size 1

            pred = model.predict(img_array)[0][0]  # Prédit un score binaire (entre 0 et 1)

            # Convertir la prédiction en label (par exemple, seuil 0.5)
            label = 'dog' if pred > 0.5 else 'cat'

            print(f"Image: {fname}, Prediction score: {pred:.4f}, Label: {label}")

            results.append((fname, pred, label))

    return results

if __name__ == '__main__':
    predict_on_test_images()
