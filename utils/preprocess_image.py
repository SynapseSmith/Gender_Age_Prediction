from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_image(img_path, actual_gender, actual_age):
    img = Image.open(img_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(img)

    gender_tensor = torch.tensor(actual_gender, dtype=torch.long)
    age_tensor = torch.tensor(actual_age, dtype=torch.float)

    return img_tensor, (gender_tensor, age_tensor)