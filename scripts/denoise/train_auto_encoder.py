import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def preprocess_image(file_path, gender, age):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    return image, image


def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs, decoded)
    return autoencoder


def main():
    print('Tensorflow Version:', tf.__version__)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

    train_data_dir = '../../data/Training'
    val_data_dir = '../../data/Validation'
    batch_size = 32

    train_dataset = load_data(train_data_dir, batch_size)
    val_dataset = load_data(val_data_dir, batch_size)

    input_shape = (224, 224, 3)
    model = build_autoencoder(input_shape)

    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['accuracy'])

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath((os.path.join("../runs", timestamp)))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.keras")

    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        TensorBoard(log_dir='../logs', histogram_freq=1)
    ]

    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=10,
                        callbacks=callbacks)

    print("=====================================")
    print("Evaluating on validation dataset...")
    val_results = model.evaluate(val_dataset)
    for name, value in zip(model.metrics_names, val_results):
        print(f"{name}: {value:.4f}")


if __name__ == '__main__':
    main()
