from model import *
import torch
import torch.nn as nn
import torchvision
# from model import build_efficientVit, poolformerv2_s12
from model.metaformer import poolformerv2_s12

class DoubleImageModel(nn.Module):
    def __init__(self, num_classes, pretrained_path):
        super().__init__()
        # self.model1 = poolformer_s12(num_classes=1000)
        # self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        self.model1 = efficientnetv2_s(num_classes=1000)
        self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model1.head = torch.nn.Linear(self.model1.head.in_features, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(1000, num_classes),
            # nn.ReLU(),
            # nn.Linear(10, num_classes),
        )
        self.activate_layer = nn.Sigmoid()
    def forward(self, x, y):
        x1 = self.model1(x)
        x1 = self.fc(x1)
        x1_sigmoid = self.activate_layer(x1)
        y1 = self.model1(y)
        y1 = self.fc(y1)
        y1_sigmoid = self.activate_layer(y1)
        return x1, x1_sigmoid, y1, y1_sigmoid

class DoubleImageModel7Class_1(nn.Module):
    def __init__(self, num_classes, pretrained_path='None'):
        super().__init__()
        # self.model1 = poolformer_s12(num_classes=1000)
        # self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model1.head = torch.nn.Linear(self.model1.head.in_features, num_classes)
        self.model1 = efficientnetv2_s(num_classes=1000)
        self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        self.model1.head.classifier = torch.nn.Linear(self.model1.head.classifier.in_features, num_classes)
        # self.model2 = poolformer_s12(num_classes=1000)
        # self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model2.head = torch.nn.Linear(self.model2.head.in_features, num_classes)
        self.model2 = efficientnetv2_s(num_classes=1000)
        self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        self.model2.head.classifier = torch.nn.Linear(self.model2.head.classifier.in_features, num_classes)
        self.fc1 = nn.Linear(2 * num_classes, 10)
        self.fc2 = nn.Linear(10, num_classes)
    def forward(self, x, y):
        x1 = self.model1(x)
        y1 = self.model2(y)
        fusion = torch.cat((x1, y1), dim=1)
        result = self.fc1(fusion)
        result = self.fc2(result)
        return result


class DoubleImageModel7Class(nn.Module):
    def __init__(self, num_classes, pretrained_path):
        super().__init__()
        # self.model1 = poolformer_s12(num_classes=1000)
        # self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model1.head = torch.nn.Linear(self.model1.head.in_features, num_classes)

        # self.model1 = efficientnetv2_s(num_classes=1000)
        # self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model1.head.classifier = torch.nn.Linear(self.model1.head.classifier.in_features, num_classes)

        # self.model1 = torchvision.models.inception_v3(num_classes=1000, weights='IMAGENET1K_V1')
        # self.model1 = torchvision.models.inception_v3(num_classes=1000)
        # self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))

        # self.model1 = build_efficientVit.EfficientViT_M5(pretrained='efficientvit_m5')
        self.model1 = build_efficientVit.EfficientViT_M0(pretrained='efficientvit_m0')

        # self.model1 = torch.hub.load(pretrained_path, 'resnet50', source='local', trust_repo=True)
        # self.model1 = torch.hub.load('facebookresearch/swav:main', 'resnet50')

        # self.model2 = poolformer_s12(num_classes=1000)
        # self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model2.head = torch.nn.Linear(self.model2.head.in_features, num_classes)
        # self.model2 = efficientnetv2_s(num_classes=1000)
        # self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model2.head.classifier = torch.nn.Linear(self.model2.head.classifier.in_features, num_classes)
        # 冻结 efficientnetv2_s 参数训练
        # for name, para in self.model1.named_parameters():
        #     para.requires_grad_(False)
        self.fc = nn.Sequential(
            nn.Linear(2 * 1000, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64, num_classes),
        )
    def forward(self, left_right_concat):
        x = left_right_concat[:, :3, :, :]
        y = left_right_concat[:, 3:, :, :]
        # if self.train():
        #     x1 = self.model1(x)[0]
        #     y1 = self.model1(y)[0]
        # else:
        #     x1 = self.model1(x)
        #     y1 = self.model1(y)
        x1 = self.model1(x)
        y1 = self.model1(y)
        fusion = torch.cat((x1, y1), dim=1)
        result = self.fc(fusion)
        return result

class SingleImageModel7Class(nn.Module):
  def __init__(self, num_classes, pretrained_path):
      super().__init__()
      # self.model1 = build_efficientVit.EfficientViT_M5(pretrained='efficientvit_m5')
      # self.model1 = build_efficientVit.EfficientViT_M0(pretrained='efficientvit_m0')
      # self.model1.head[1] = nn.Linear(192, num_classes)
      self.model1 = poolformerv2_s12()
      self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
      self.model1.head = torch.nn.Linear(self.model1.head.in_features, num_classes)

  def forward(self, image):
      # x = left_right_concat[:, :3, :, :]
      # y = left_right_concat[:, 3:, :, :]
      result = self.model1(image)
      # y1 = self.model1(y)
      # fusion = torch.cat((x1, y1), dim=1)
      return result


class DoubleImageModel7ClassFeature(nn.Module):
    def __init__(self, num_classes, pretrained_path):
        super().__init__()
        self.model1 = efficientnetv2_s(num_classes=1000)
        self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
    def forward(self, x, y):
        x1 = self.model1(x)
        y1 = self.model1(y)
        result = torch.concat([x1, y1], dim=1)
        return result



def get_model(num_class, pretrained_path, device):

    # model = poolformer_s12(num_classes=1000)
    # model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # model.head = torch.nn.Linear(model.head.in_features, num_class)
    # self_model = model.to(device)

    # model = resnet34(num_classes=1000)
    # model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    # model.fc = torch.nn.Linear(model.fc.in_features, num_class)  # 修改全连接层
    # model = model.to(device)


    # model = EyeSeeNet(num_class=7).to(device)

    self_model = efficientnetv2_s(num_classes=1000)
    self_model.load_state_dict(torch.load(pretrained_path, weights_only=True))
    self_model.head.classifier = torch.nn.Linear(self_model.head.classifier.in_features, num_class)
    self_model = self_model.to(device)
    # for name, para in model.named_parameters():
    #     # 除head外，其他权重全部冻结
    #     if "head" not in name:
    #         para.requires_grad_(False)
    #     else:
    #         print("training {}".format(name))


    return self_model

