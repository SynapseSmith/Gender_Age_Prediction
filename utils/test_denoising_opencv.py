import numpy as np
import matplotlib.pyplot as plt
from Project.utils.face_dataset import FaceImageDataset, denoise_image

img_dir = '../data/Combined_Test'
sample_idx = 5

denoise_method = 'median'  # 'fastnlm', 'gaussian', 'median', 'bilateral', 'anisotropic', 'wavelet'

dataset = FaceImageDataset(img_dir=img_dir)

original_image, gender, age = dataset[sample_idx]

image_np = np.array(original_image.permute(1, 2, 0) * 255, dtype=np.uint8)
denoised_image_np = denoise_image(image_np, method=denoise_method)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(image_np)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(denoised_image_np)
axs[1].set_title(f'Denoised Image ({denoise_method})')
axs[1].axis('off')

plt.show()