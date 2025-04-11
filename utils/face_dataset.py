import os
# import pywt
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from Project.utils.extract_labels import extract_labels


class FaceImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, denoise_method=None):
        self.img_dir = img_dir
        self.transform = transform
        self.denoise_method = denoise_method
        self.image_filenames = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        file_path = os.path.join(self.img_dir, self.image_filenames[idx])

        with Image.open(file_path) as img:
            image = img.convert('RGB')

        image_np = np.array(image)

        if self.denoise_method:
            image_np = denoise_image(image_np, method=self.denoise_method)

        image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)
        else:
            image = ToTensor()(image)

        gender, age = extract_labels(file_path)

        return image, gender, age


def denoise_image(image, method='median'):
    if method == 'fastnlm':
        return cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)

    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 1.0)

    elif method == 'median':
        image = cv2.medianBlur(image, 5)
        return cv2.medianBlur(image, 5)

    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)

    elif method == 'anisotropic':
        return anisotropic_diffusion(image)

    elif method == 'wavelet':
        return wavelet_denoising(image)

    else:
        raise ValueError("Invalid method. Choose from 'fastnlm', 'gaussian', 'median', 'bilateral', 'anisotropic', 'wavelet'.")


def anisotropic_diffusion(image, num_iter=10, kappa=50, gamma=0.1, option=1):
    img = image.astype(np.float32)
    for t in range(num_iter):
        nabla_north = np.roll(img, -1, axis=0) - img
        nabla_south = np.roll(img, 1, axis=0) - img
        nabla_east = np.roll(img, -1, axis=1) - img
        nabla_west = np.roll(img, 1, axis=1) - img

        if option == 1:
            c_north = np.exp(-(nabla_north / kappa) ** 2)
            c_south = np.exp(-(nabla_south / kappa) ** 2)
            c_east = np.exp(-(nabla_east / kappa) ** 2)
            c_west = np.exp(-(nabla_west / kappa) ** 2)
        elif option == 2:
            c_north = 1 / (1 + (nabla_north / kappa) ** 2)
            c_south = 1 / (1 + (nabla_south / kappa) ** 2)
            c_east = 1 / (1 + (nabla_east / kappa) ** 2)
            c_west = 1 / (1 + (nabla_west / kappa) ** 2)

        img += gamma * (c_north * nabla_north + c_south * nabla_south + c_east * nabla_east + c_west * nabla_west)

    return np.uint8(img)


# def wavelet_denoising(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     coeffs = pywt.wavedec2(gray_image, 'db1', level=2)
#     coeffs[1:] = [(pywt.threshold(i, value=10, mode='soft') for i in level) for level in coeffs[1:]]
#
#     denoised_image = pywt.waverec2(coeffs, 'db1')
#     denoised_image = np.uint8(denoised_image)
#
#     denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2BGR)
#     return denoised_image
