import tensorflow as tf
import cv2


def preprocess_image(file_path, gender_label, age_label):
    def load_and_preprocess_image(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    image = tf.numpy_function(load_and_preprocess_image, [file_path], tf.float32)
    image.set_shape((224, 224, 3))
    gender_label = tf.one_hot(gender_label, 2)
    age_label = tf.cast(age_label, tf.float32)
    return image, (gender_label, age_label)