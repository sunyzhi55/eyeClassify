"""
PoolFormer implementation
"""
from datetime import datetime
from pathlib import Path
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dataset import EyesDataset
from model import *
from Resnet import *
from train_test import train_sector, test_sector, k_fold_cross_validation


class EyeSeeNet(nn.Module):
    def __init__(self, num_class):
        super(EyeSeeNet, self).__init__()
        self.num_class = num_class
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_class)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * 14 * 14)
        x = self.fc(x)
        return x


# Use the Friendly-SAM optimizer
# 训练
def train_sector_use_friendly_sam(net, device, train_load, loss_fn, optimizer, epoch, writer):
    net.train()
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    for batch_index, (image, label) in enumerate(tqdm(train_load, desc=f'Epoch {epoch}', unit='batch')):
        image, label = image.to(device), label.to(device)

        enable_running_stats(net)
        # first forward-backward step
        predictions = net(image)
        loss = loss_fn(predictions, label)
        total_train_loss += loss.item()
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(net)
        output_adv = net(image)
        loss_adv = loss_fn(output_adv, label)
        # total_train_loss += loss.item()
        loss_adv.mean().backward()
        optimizer.second_step(zero_grad=True)


        # output = net(image)
        # loss = loss_fn(output, label)
        # total_train_loss += loss.item()
        accuracy = (predictions.argmax(dim=1) == label).sum().item()
        total_train_accuracy += accuracy
        # loss.backward()
        # optimizer.step()

        # if batch_index % 100 == 0:
        #     print(f'Train Epoch:{epoch} [{batch_index}/{len(train_load)}] loss:{loss.item():.6f}')
    total_train_loss /= len(train_load.dataset)
    total_train_accuracy /= len(train_load.dataset)
    writer.add_scalar("train_loss", total_train_loss, epoch)
    writer.add_scalar("train_accuracy", total_train_accuracy, epoch)
    print(f"Train Average Loss:{total_train_loss},Accuracy:{total_train_accuracy}")




# if __name__ == "__main__":
#     model = poolformer_s12(num_classes=10)
#     print("参数量", sum(p.numel() for p in model.parameters()))
#     x = torch.randn(1, 3, 112, 112)
#     y = model(x)
#     print(y.shape)
#     """
#     参数量 11407306
#     input torch.Size([1, 3, 32, 32])
#     forward_embeddings torch.Size([1, 64, 8, 8])
#     forward_tokens torch.Size([1, 512, 1, 1])
#     norm torch.Size([1, 512, 1, 1])
#     mean[-2, -1] torch.Size([1, 512])
#     cls_out torch.Size([1, 10])
#     torch.Size([1, 10])
#     """




if __name__ == '__main__':
    current_time = "{0:%Y%m%d_%H_%M}".format(datetime.now())
    args = parse_args()
    # 获取运行的设备
    if args.device != 'cpu':
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # # 加载和预处理数据集
    # trans_train = transforms.Compose(
    #     [transforms.RandomResizedCrop(112),  # 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；
    #      # （即先随机采集，然后对裁剪得到的图像缩放为同一大小） 默认scale=(0.08, 1.0)
    #      transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]
    # )
    # trans_valid = transforms.Compose(
    #     [transforms.Resize(256),  # 是按照比例把图像最小的一个边长放缩到256，另一边按照相同比例放缩。
    #      transforms.CenterCrop(112),  # 依据给定的size从中心裁剪
    #      transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    #      # 归一化至[0-1]是直接除以255，若自己的ndarray数据尺度有变化，则需要自行修改。
    #      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ]  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
    # )
    #
    # train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=trans_train)
    # test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=trans_valid)
    #
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    image_size = args.Image_size

    # 定义数据增强和预处理
    # transform = transforms.Compose([
    #     transforms.Resize((image_size, image_size)),  # 调整图片大小
    #     # transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；
    #     transforms.ToTensor(),  # 转换为 Tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    # ])

    # Define transforms for each dataset separately

    train_validation_test_transform={
        'train_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomRotation(45),
        transforms.RandomAdjustSharpness(1.3, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化

        ]),
        'validation_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ]),
        'test_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    }
    # 初始化自定义数据集
    dataset = EyesDataset(csv_file=args.csv_file_path, img_prefix=args.data_dir, transform=None)


    # dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=None)
    # data_path = r"/home/shenxiangyuhd/public_dataset/cifar10"
    # train_data = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
    # test_data = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform, download=True)
    # dataset = torch.utils.data.ConcatDataset([train_data, test_data])




    # 定义 DataLoader
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # # 测试 DataLoader
    # for batch_idx, (images, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"Images shape: {images.shape}")
    #     print(f"Labels: {labels}")

    # train_data_size = len(train_data)
    # test_data_size = len(test_data)
    # print(f"训练数据集的长度:{train_data_size}")
    # print(f"测试数据集的长度:{test_data_size}")
    # writer = SummaryWriter("Custom_metaFormer_logs")

    # 定义模型
    # Net = poolformer_s12(num_classes=8).to(device)

    # 查看总参数及训练参数
    # total_params = sum(p.numel() for p in Net.parameters())
    # print('总参数个数:{}'.format(total_params))
    # 损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)  # 损失函数
    # 学习率
    lr = args.learning_rate
    # 优化器
    # optimizer = torch.optim.SGD(Net.parameters(), lr=lr, weight_decay=1e-3, momentum=0.9)  # 优化器
    # optimizer = torch.optim.AdamW(Net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    # optimizer = torch.optim.Adam(Net.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8)
    # base_optimizer = torch.optim.SGD
    # optimizer = FriendlySAM(Net.parameters(), base_optimizer, rho=0.2, sigma=1, lmbda=0.6,
    #                         adaptive=0, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=False)

    # base_optimizer = torch.optim.Adam
    # optimizer = FriendlySAM(Net.parameters(), base_optimizer, rho=0.2, sigma=1, lmbda=0.6,
    #                         adaptive=0, lr=lr, weight_decay=1e-4)

    # print(optimizer)

    # 定义模型保存的文件夹
    model_dir = args.checkpoint_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # 训练的总轮数
    EPOCH = args.epochs
    epoch = 0
    k_fold_cross_validation(device, dataset, loss_fn, EPOCH, args, lr=lr, writer=None, k_fold=args.k_split_value,
                            batch_size=args.batch_size, workers=2, print_freq=1, checkpoint_dir=model_dir,
                            best_result_model_path="model", total_transform=train_validation_test_transform)