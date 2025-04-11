from torch import nn
from torchvision import models
from facenet_pytorch import InceptionResnetV1
from transformers import AutoModelForImageClassification

class GenderAgeModel(nn.Module):
    def __init__(self, model_name='efficientnet_v2_m'):
        super(GenderAgeModel, self).__init__()
        self.model_name = model_name
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Identity()
        elif model_name == 'efficientnet_b0_huggingface':
            self.model = AutoModelForImageClassification.from_pretrained('google/efficientnet-b0')
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'efficientnet_b4':
            self.model = models.efficientnet_b4(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Identity()
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Identity()
        elif model_name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=True)
            num_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Identity()
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name == 'mnasnet0_75':
            self.model = models.mnasnet0_75(pretrained=True)
            num_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Identity()
        elif model_name == 'squeezenet1_0':
            self.model = models.squeezenet1_0(pretrained=True)
            num_features = 512
            self.model.classifier[1] = nn.Identity()
        elif model_name == 'shufflenet_v2_x0_5':
            self.model = models.shufflenet_v2_x0_5(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
        elif model_name == 'convnext_tiny':
            self.model = models.convnext_tiny(pretrained=True)
            num_features = self.model.classifier[2].in_features
            self.model.classifier[2] = nn.Identity()
        elif model_name == 'densenet169':
            self.model = models.densenet169(pretrained=True)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif model_name == 'vit_b_32':
            self.model = models.vit_b_32(pretrained=True)
            num_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()
        elif model_name == 'efficientnet_v2_m':
            self.model = models.efficientnet_v2_m(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Identity()
        elif model_name == 'facenet':
            self.model = InceptionResnetV1(pretrained='vggface2')  # vggface2 또는 casia-webface 사용 가능
            num_features = 512  # FaceNet의 경우 512차원 임베딩 벡터를 출력

        for p in self.model.parameters():
            p.requires_grad = True

        self.fc_gender = nn.Linear(num_features, 2)

        self.fc_age = nn.Linear(num_features, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.model(x)
        gender_out = self.fc_gender(x)
        age_out = self.fc_age(x)
        return gender_out, age_out


# model = GenderAgeModel()
# for name, module in model.named_children():
#     print(name, module)


