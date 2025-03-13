from model import *
import torch
import torch.nn as nn

class DoubleImageModel(nn.Module):
    def __init__(self, num_classes, pretrained_path):
        super().__init__()
        self.model1 = poolformer_s12(num_classes=1000)
        self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        self.model1.head = torch.nn.Linear(self.model1.head.in_features, num_classes)
        self.activate_layer = nn.Sigmoid()
    def forward(self, x, y):
        x1 = self.model1(x)
        x1_sigmoid = self.activate_layer(x1)
        y1 = self.model1(y)
        y1_sigmoid = self.activate_layer(y1)
        return x1, x1_sigmoid, y1, y1_sigmoid

class DoubleImageModel7Class_1(nn.Module):
    def __init__(self, num_classes, pretrained_path):
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
        self.model1 = efficientnetv2_s(num_classes=1000)
        self.model1.load_state_dict(torch.load(pretrained_path, weights_only=True))
        self.model1.head.classifier = torch.nn.Linear(self.model1.head.classifier.in_features, num_classes)
        # self.model2 = poolformer_s12(num_classes=1000)
        # self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model2.head = torch.nn.Linear(self.model2.head.in_features, num_classes)
        # self.model2 = efficientnetv2_s(num_classes=1000)
        # self.model2.load_state_dict(torch.load(pretrained_path, weights_only=True))
        # self.model2.head.classifier = torch.nn.Linear(self.model2.head.classifier.in_features, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
        )
    def forward(self, x, y):
        x1 = self.model1(x)
        y1 = self.model1(y)
        fusion = torch.cat((x1, y1), dim=1)
        result = self.fc(fusion)
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
