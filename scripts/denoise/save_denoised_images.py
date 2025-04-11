import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import save_img

from Project.etl.extract_labels import extract_labels
from Project.etl.preprocess_image import preprocess_image
from Project.etl.load_data import load_data


def load_data(data_dir, img_size=(224, 224)):
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image_paths.append(file_path)

    return image_paths


def preprocess_image(file_path, img_size=(224, 224)):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image


def save_denoised_images(model, image_paths, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_path in image_paths:
        image = preprocess_image(img_path)
        image = tf.expand_dims(image, axis=0)
        denoised_image = model.predict(image)

        if len(denoised_image.shape) == 4:
            denoised_image = np.squeeze(denoised_image, axis=0)
        denoised_image = np.clip(denoised_image * 255.0, 0, 255).astype(np.uint8)

        save_path = os.path.join(save_dir, os.path.basename(img_path))
        save_img(save_path, denoised_image)


def main():
    print('Tensorflow Version:', tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

    data_dir = '../../data/Test'
    save_dir = '../../data/Denoised_Test'

    image_paths = load_data(data_dir)

    timestamp = '1717022464_AutoEncoder'
    checkpoint_path = f'../../scripts/runs/{timestamp}/checkpoints/best_model.keras'
    model = load_model(checkpoint_path)

    save_denoised_images(model, image_paths, save_dir)


if __name__ == '__main__':
    main()
