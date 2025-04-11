import os
import random
from torch import nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from Project.models.gender_age_model import GenderAgeModel


class GradCAM:
    def __init__(self, model, main, sub):
        self.model = model.eval()
        self.register_hook(main, sub)

    def register_hook(self, main, sub):
        for name, module in self.model.named_children():
            if name == main:
                for sub_name, sub_module in module.named_children():
                    if sub_name == sub:
                        sub_module.register_forward_hook(self.forward_hook)
                        sub_module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.feature_map = output

    def backward_hook(self, module, grad_iuput, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x):
        output = self.model(x)
        output = output[0] # 0: 성별, 1: 나이
        index = output.argmax(axis=1)
        one_hot = torch.zeros_like(output)
        for  i in range(output.size(0)):
            one_hot[i][index[i]] = 1

        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)

        a_k = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(a_k * self.feature_map, dim=1)
        grad_cam = torch.relu(grad_cam)
        return grad_cam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = '1718549703_MobileNetV3Small'
checkpoint_path = f'../scripts/runs/{timestamp}/checkpoints/best.pth'
model = GenderAgeModel().to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# files = ['../data/Test/Aihub1_00021167_Female_19_110.png',
# '../data/Test/MALE_Aihub2_00002749_MALE_12_125.png',
# '../data/Test/Aihub1_00049056_Male_56_125.png',
# '../data/Test/Aihub1_00027159_Female_59_110.png']

data_dir = "../data/Training"
files = os.listdir(data_dir)
random.seed(17)
random.shuffle(files)
files = files[:100]

images, tensors = [], []
for file in files:
    file = os.path.join(data_dir, file)
    image = Image.open(file)
    images.append(image)
    tensors.append(transform(image))
tensors = torch.stack(tensors).to(device)

model = GradCAM(
    model=model, main='model', sub='features'
)

grad_cams = model(tensors)

for idx, image in enumerate(images):
    grad_cam = F.interpolate(
        input=grad_cams[idx].unsqueeze(0).unsqueeze(0),
        size=(image.size[1], image.size[0]),
        mode='bilinear'
    ).squeeze().detach().cpu().numpy()

    plt.imshow(image)
    plt.imshow(grad_cam, cmap='jet', alpha=0.45)
    plt.axis('off')
    plt.show()
