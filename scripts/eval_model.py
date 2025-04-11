import numpy as np
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from Project.utils.face_dataset import FaceImageDataset
from Project.models.gender_age_model import GenderAgeModel

# 모델의 예측 성능 (성별 정확도, 성별 f1 score, 나이 MAE) 평가
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timestamp = '1725727621'  # 체크포인트
    checkpoint_path = f'../scripts/runs/{timestamp}/checkpoints/best.pth'
    model = GenderAgeModel().to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('model:', model.model_name)
    n = sum(p.numel() for p in model.parameters())
    print(f'The Number of Parameters of {model.model_name} : {n:,}')

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_dir = "../data/Test"  # 평가할 데이터셋
    print('img_dir:', img_dir)
    test_data = FaceImageDataset(img_dir, transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()

    correct_gender = 0
    total_gender = 0
    all_genders = []
    all_preds = []
    age_errors = []

    start_time = time.time()
    with torch.no_grad():
        for img, gender, age in test_loader:
            img = img.to(device)
            gender = gender.to(device)
            age = age.float().to(device)

            output_gender, output_age = model(img)
            output_age = output_age.squeeze()

            _, gender_pred = torch.max(output_gender, 1)
            correct_gender += (gender_pred == gender).sum().item()
            total_gender += gender.size(0)

            all_genders.extend(gender.cpu().numpy())
            all_preds.extend(gender_pred.cpu().numpy())

            age_errors.append(torch.abs(output_age - age).cpu().numpy())
    end_time = time.time()
    total_time = end_time - start_time
    print(f'\nEvaluation Time: {total_time:.2f} seconds')

    gender_acc = correct_gender / total_gender * 100
    f1 = f1_score(all_genders, all_preds) * 100
    age_mae = np.mean(np.concatenate(age_errors))

    print(f"Test Gender Accuracy: {gender_acc:.2f}%")
    print(f"Test F1 Score: {f1:.2f}%")
    print(f"Test Age MAE: {age_mae:.4f}")


if __name__ == '__main__':
    main()