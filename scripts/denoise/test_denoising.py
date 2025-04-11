import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from Project.etl.load_data import load_data


def visualize_results(model, dataset, num_images=20):
    for images, _ in dataset.take(1):
        predictions = model.predict(images)
        for i in range(num_images):
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.title("Input Image")
            plt.imshow(images[i])

            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.title("Denoised Image")
            plt.imshow(predictions[i])

            plt.show()


def main():
    print('Tensorflow Version:', tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

    val_data_dir = '../../data/Validation'
    batch_size = 32

    val_dataset = load_data(val_data_dir, batch_size)

    timestamp = '1717022464_AutoEncoder'
    checkpoint_path = f'../../scripts/runs/{timestamp}/checkpoints/best_model.keras'
    model = load_model(checkpoint_path)

    visualize_results(model, val_dataset)


if __name__ == '__main__':
    main()
