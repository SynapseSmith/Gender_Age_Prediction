import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
import matplotlib.pyplot as plt
from Project.utils.extract_labels import extract_labels
from Project.utils.preprocess_image import preprocess_image
from Project.models.gender_age_model import GenderAgeModel
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 이미지 여러 장 추론, 시각화
def main():
    img_dir = "../data/Test"  # 추론할 데이터셋 폴더
    print(img_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = '1717297807_MobileNetV3Small'

    checkpoint_path = f'../scripts/runs/{timestamp}/checkpoints/best.pth'
    model = GenderAgeModel().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image_files = [f for f in os.listdir(img_dir)]
    random.shuffle(image_files)

    for img_file in image_files:
        img_path = os.path.join(img_dir, img_file)
        actual_gender, actual_age = extract_labels(img_path)

        img_tensor, _ = preprocess_image(img_path, actual_gender, actual_age)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            gender_pred, age_pred = model(img_tensor)
        end_time = time.time()

        prediction_time = end_time - start_time
        print(f'이미지 {img_file} 예측 시간: {prediction_time:.2f}초')

        actual_gender_label = '남성' if actual_gender == 0 else '여성'

        gender_label = torch.argmax(gender_pred, dim=1).item()
        predicted_gender = '남성' if gender_label == 0 else '여성'
        predicted_age = round(age_pred.item())

        img = Image.open(img_path).resize((224, 224))

        plt.imshow(img)
        plt.title(f'정답 - {int(actual_age)}세 {actual_gender_label}\n예측 - {predicted_age}세 {predicted_gender}', fontdict={'fontsize': 15})
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()
