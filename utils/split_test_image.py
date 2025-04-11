import os
import random
import shutil

def split_data(training_dir, test_dir, num_test_images=10000):
    all_images = os.listdir(training_dir)

    image_files = [f for f in all_images]

    test_images = random.sample(image_files, num_test_images)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for image in test_images:
        src_path = os.path.join(training_dir, image)
        dst_path = os.path.join(test_dir, image)
        shutil.move(src_path, dst_path)


if __name__ == '__main__':
    training_directory = '../data/Training'
    test_directory = '../data/Test'

    split_data(training_directory, test_directory)