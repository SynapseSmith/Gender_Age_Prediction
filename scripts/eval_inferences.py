import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from Project.utils.extract_labels import extract_labels
from Project.utils.preprocess_image import preprocess_image
from Project.models.gender_age_model import GenderAgeModel

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 모델의 추론 성능 평가
def main():
    num_predictions = 10000
    plot_images = True

    img_folder = '../data/Test'  # 평가할 데이터셋
    print(img_folder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    timestamp = '1718549703_MobileNetV3Small'  # 체크포인트
    checkpoint_path = f'../scripts/runs/{timestamp}/checkpoints/best.pth'
    model = GenderAgeModel().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(model.model_name)
    model.eval()

    incorrect_age_predictions = []
    incorrect_gender_predictions = []
    total_prediction_time = 0

    for img_file in tqdm(os.listdir(img_folder)[:num_predictions]):
        img_path = os.path.join(img_folder, img_file)

        actual_gender, actual_age = extract_labels(img_path)

        img_tensor, _ = preprocess_image(img_path, actual_gender, actual_age)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        start_time = time.time()
        with torch.no_grad():
            gender_pred, age_pred = model(img_tensor)
        end_time = time.time()

        total_prediction_time += end_time - start_time

        actual_gender_str = '남성' if actual_gender == 0 else '여성'

        gender_label = torch.argmax(gender_pred, dim=1).item()
        predicted_gender = '남성' if gender_label == 0 else '여성'
        predicted_age = round(age_pred.item())

        if predicted_gender != actual_gender_str:
            incorrect_gender_predictions.append((img_path, actual_age, actual_gender_str, predicted_age, predicted_gender))

        if abs(predicted_age - actual_age) >= 10:
            incorrect_age_predictions.append((img_path, actual_age, actual_gender_str, predicted_age, predicted_gender))

    print(f'이미지 {num_predictions}장 예측 시간: {total_prediction_time:.2f}초')
    print(f'성별을 잘못 분류한 경우: {len(incorrect_gender_predictions)}장')
    print(f'나이 예측 오류가 10 이상인 경우: {len(incorrect_age_predictions)}장')

    # 예측 실수한 경우 시각화
    if plot_images:
        for img_path, actual_age, actual_gender_str, predicted_age, predicted_gender in incorrect_gender_predictions:
            print(img_path)
            img = Image.open(img_path).resize((224, 224))
            plt.imshow(img)
            plt.title(f'정답 - {actual_age}세 {actual_gender_str}\n예측 - {predicted_age}세 {predicted_gender}',
                      fontdict={'fontsize': 15})
            plt.axis('off')
            plt.show()

        for img_path, actual_age, actual_gender_str, predicted_age, predicted_gender in incorrect_age_predictions:
            print(img_path)
            img = Image.open(img_path).resize((224, 224))
            plt.imshow(img)
            plt.title(f'정답 - {actual_age}세 {actual_gender_str}\n예측 - {predicted_age}세 {predicted_gender}',
                      fontdict={'fontsize': 15})
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()
