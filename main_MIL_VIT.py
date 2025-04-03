import time
import torch
from datetime import datetime
from pathlib import Path
from torchvision import transforms
from Dataset import *
from Config_MIL_VIT import parse_args
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from othersModel import *
from utils.metrics import *
from utils.basic import get_scheduler


class DoubleTransformedSubsetForMILVIT(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y, label_2d_int, label_2d_float = self.subset[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y, label_2d_int, label_2d_float
    def __len__(self):
        return len(self.subset)


def k_fold_cross_validation_double_8_class_for_MIL_VIT(device, eyes_dataset, args, workers=2, print_freq=1,
                                            best_result_model_path="model", total_transform=None):
    # skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    checkpoint_dir = args.checkpoint_dir
    epoch_acc_dict = {}

    # 获取数据集的标签
    # labels = [data[1] for data in eyes_dataset]  # 假设dataset[i]的第2项是label
    kf = KFold(n_splits=args.k_split_value, shuffle=True, random_state=0)  # init KFold
    batch_size = args.batch_size
    # 学习率
    lr = args.learning_rate

    # 训练的总轮数
    epochs = args.epochs

    # 清空日志文件
    with open(f'logs.txt', 'w') as f:
        pass

    for fold, (train_index, test_index) in enumerate(kf.split(eyes_dataset)):  # split
        with open(f'logs.txt', 'a') as f:
            f.write(f"Fold {fold + 1}\n")

        k_train_fold = Subset(eyes_dataset, train_index)
        k_test_fold = Subset(eyes_dataset, test_index)
        # 应用转换
        train_dataset = DoubleTransformedSubsetForMILVIT(k_train_fold, transform=total_transform['train_transforms'])
        val_dataset = DoubleTransformedSubsetForMILVIT(k_test_fold, transform=total_transform['validation_transforms'])

        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        # model = DoubleImageModel7Class(num_classes=7, pretrained_path=args.pretrainedModelPath)
        model = MIL_VIT_Double(image_size=args.Image_size, MODEL_PATH_finetune=args.pretrainedModelPath, num_classes=7)
        model = model.to(device)

        multiLayers = list()
        for name, layer in model._modules.items():
            if name.__contains__('MIL_'):
                multiLayers.append({'params': layer.parameters(), 'lr': 5 * lr})
            else:
                multiLayers.append({'params': layer.parameters()})


        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
        optimizer = torch.optim.Adam(multiLayers, lr=lr, eps=1e-8, weight_decay=1e-5)

        scheduler = get_scheduler(optimizer, args)

        # 损失函数
        ratio = torch.zeros(7)
        Count = 0
        for (_, _, labels_int, labels_float) in train_dataloader:
            ratio += torch.sum(labels_float, dim=0)
            Count += labels_float.shape[0]
            # break
            # print(ratio, Count)
        # ratio = torch.tensor([100., 100., 100., 100., 100., 100., 100.])
        print("各疾病总数:", ratio, Count)
        ratio = Count / ratio - 1
        print("各疾病损失权重:", ratio)
        # X = torch.tensor([12.5706])
        # loss_fn = nn.BCEWithLogitsLoss().to(device)  # 损失函数
        # pos_weight_for_7_disease = torch.tensor([6.89, 16.51, 17.47, 19.88, 38.74, 22.61, 3.55])  # true
        # pos_weight_for_7_disease = torch.tensor([6.89, 16.51, 17.47, 10.0, 18.0, 22.61, 3.55]) # false
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_7_disease).to(device)  # 损失函数
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=ratio).to(device)
        # loss_fn = FocalLoss(device=device, position_weight=ratio).to(device)

        # loss_fn = nn.CrossEntropyLoss().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print('总参数个数:{}'.format(total_params))

        # metrics_single_label = MetricsWithSingleLabel(num_classes=2, device=device)
        metrics_mul_label = MetricsWithMultiLabelWithTorchmetrics(num_classes=7, device=device)
        for e in range(1, epochs + 1):
            model.train()
            metrics_mul_label.reset()
            # 获取当前学习率（假设只有一个参数组）
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {e}, LR {current_lr:.6f}", unit="batch")
            for batch_idx, (left_image, right_image, label_index_2d_int, label_index_2d_float) in enumerate(train_iterator):
                left_image, right_image, = left_image.to(device), right_image.to(device)
                label_index_2d_int, label_index_2d_float = label_index_2d_int.to(device), label_index_2d_float.to(device)
                # print(f"image:{images.shape}")
                # print(f"labels:{labels}")  # torch.Size([2])
                optimizer.zero_grad()
                outputs_class1, outputs_MIL1, outputs_class2, outputs_MIL2, result = model(left_image, right_image)
                # logit = model(left_image, right_image)
                # outputs_class, outputs_MIL = model(inputs)

                # _, predictions = torch.max(prob, dim=1)
                # prob_positive = prob[:, 1]
                loss1_1 = loss_fn(outputs_class1, label_index_2d_float)
                loss1_2 = loss_fn(outputs_MIL1, label_index_2d_float)
                loss2_1 = loss_fn(outputs_class2, label_index_2d_float)
                loss2_2 = loss_fn(outputs_MIL2, label_index_2d_float)
                loss3 = loss_fn(result, label_index_2d_float)
                loss = (loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3) / 5.0

                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                metrics_mul_label.train_update(loss, result, label_index_2d_int)
            if scheduler:
                scheduler.step()
            with torch.no_grad():
                model.eval()
                eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
                for batch_idx, (left_image, right_image, label_index_2d_int, label_index_2d_float) in enumerate(eval_iterator):
                    left_image, right_image, = left_image.to(device), right_image.to(device)
                    label_index_2d_int, label_index_2d_float = label_index_2d_int.to(device), label_index_2d_float.to(device)
                    # logit = model(left_image, right_image)
                    outputs_class1, outputs_class2, result = model(left_image, right_image)

                    # _, predictions = torch.max(prob, dim=1)
                    # prob_positive = prob[:, 1]
                    loss1_1 = loss_fn(outputs_class1, label_index_2d_float)
                    loss2_1 = loss_fn(outputs_class2, label_index_2d_float)
                    loss3 = loss_fn(result, label_index_2d_float)
                    loss = (loss1_1 + loss2_1 + loss3) / 3.0
                    metrics_mul_label.eval_update(loss, result, label_index_2d_int)
            metrics_mul_label.compute_result()
            metrics_mul_label.average_train_loss = metrics_mul_label.total_train_loss / len(train_dataloader.dataset)
            metrics_mul_label.average_eval_loss = metrics_mul_label.total_eval_loss / len(eval_dataloader.dataset)
            if metrics_mul_label.get_best(e) :
                torch.save(model.state_dict(), checkpoint_dir + f'/{best_result_model_path}_fold{fold}.pth')
            if e % print_freq == 0:
                metrics_mul_label.print_result(e, epochs)
        metrics_mul_label.print_best_result(fold)
        epoch_acc_dict[fold] = metrics_mul_label.best_dicts['average_accuracy']
    print(f"Cross-validation result:{epoch_acc_dict}")

if __name__ == '__main__':
    current_time = "{0:%Y%m%d_%H_%M}".format(datetime.now())
    # 设置随机数种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    args = parse_args()
    # 获取运行的设备
    if args.device != 'cpu':
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    image_size = args.Image_size

    train_validation_test_transform={
        'train_transforms':transforms.Compose([
        # transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45),
        # transforms.RandomAdjustSharpness(1.3, 1),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
        # # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化

        transforms.Resize((image_size + 40, image_size + 40)),
        transforms.RandomCrop((image_size, image_size)),  # padding=10
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(hue=.05, saturation=.05, brightness=.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]),
        'validation_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ]),
        'test_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
    }
    # 初始化自定义数据集
    # dataset = EyesDataset(csv_file=args.csv_file_path, img_prefix=args.data_dir, transform=None)
    # dataset = DoubleEyesDatasetTwoClass(csv_file=args.csv_file_path, img_prefix=args.data_dir, transform=None)
    dataset = DoubleEyesDatasetAllClass(csv_file=args.csv_file_path, img_prefix=args.data_dir, transform=None)

    # 定义模型保存的文件夹
    model_dir = args.checkpoint_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    start = time.time()
    # k_fold_cross_validation(device, dataset, args, workers=2, print_freq=1,
    #                         best_result_model_path="model", total_transform=train_validation_test_transform)

    # k_fold_cross_validation_double_text_0_1(device, dataset, args, workers=2, print_freq=1,
    #                         best_result_model_path="model", total_transform=train_validation_test_transform)
    k_fold_cross_validation_double_8_class_for_MIL_VIT(device, dataset, args, workers=2, print_freq=1,
                                           best_result_model_path="model", total_transform=train_validation_test_transform)

    end = time.time()
    total_seconds = end - start
    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 打印总训练时间
    print(f"Total training time: {hours}h {minutes}m {seconds}s")