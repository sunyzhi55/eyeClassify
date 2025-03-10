import time
from datetime import datetime
from pathlib import Path
from torchvision import transforms
from Dataset import *
from model.Resnet import *
from train_test import k_fold_cross_validation
from double_text_0_1 import k_fold_cross_validation_double_text_0_1
from double_8_class import k_fold_cross_validation_double_8_class
from Config import parse_args

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
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAdjustSharpness(1.3, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ]),
        'validation_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
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
    k_fold_cross_validation_double_8_class(device, dataset, args, workers=2, print_freq=1,
                                           best_result_model_path="model", total_transform=train_validation_test_transform)

    end = time.time()
    total_seconds = end - start
    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 打印总训练时间
    print(f"Total training time: {hours}h {minutes}m {seconds}s")