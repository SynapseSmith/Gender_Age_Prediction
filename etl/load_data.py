import os
import numpy as np
import tensorflow as tf
from Project.etl.extract_labels import extract_labels
from Project.etl.preprocess_image import preprocess_image


def load_data(data_dir, batch_size):
    image_paths = []
    gender_labels = []
    age_labels = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                gender_label, age_label = extract_labels(file)
                if gender_label is not None and age_label is not None:
                    image_paths.append(file_path)
                    gender_labels.append(gender_label)
                    age_labels.append(age_label)

    image_paths = np.array(image_paths, dtype=str)
    gender_labels = np.array(gender_labels)
    age_labels = np.array(age_labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, gender_labels, age_labels))
    dataset = dataset.map(lambda file_path, gender, age: preprocess_image(file_path, gender, age),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset