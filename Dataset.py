import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision


# class EyesDataset(Dataset):
#     def __init__(self, csv_file, img_prefix, transform=None):
#         """
#         初始化 Dataset
#         :param csv_file: str, CSV 文件路径
#         :param img_prefix: str, 图片路径前缀
#         :param transform: torchvision.transforms, 数据增强和预处理
#         """
#         self.data = pd.read_csv(csv_file)
#         self.img_prefix = img_prefix
#         self.labels = {'N':0, 'D':1, 'G':2, 'C':3, 'A':4, 'H':5, 'M':6, 'O':7}
#         self.transform = transform
#
#     def __len__(self):
#         """
#         返回数据集的长度
#         """
#         return len(self.data)
#
#     def get_label_tensor(self, row):
#         """
#         根据标签列返回类别张量
#         :param row: DataFrame 的一行
#         :return: torch.Tensor, 标签张量
#         """
#         label_tensor = torch.tensor([row[label] for label in self.labels], dtype=torch.float)
#         return label_tensor
#
#     def __getitem__(self, idx):
#         """
#         获取指定索引的数据
#         :param idx: int, 数据索引
#         :return: tuple (image, label_tensor)
#         """
#         idx = int(idx)
#         if isinstance(idx, int):
#             row = self.data.iloc[idx]
#         else:
#             raise TypeError("Index must be an integer.")
#
#         # 构造图片路径
#         img_name = row.iloc[0]  # 第一列是图片的名字
#         img_path = f"{self.img_prefix}/{img_name}"
#
#         # 打开图片
#         try:
#             image = Image.open(img_path).convert('RGB')
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Image not found: {img_path}")
#
#         # 数据增强和预处理
#         if self.transform:
#             image = self.transform(image)
#
#         # 获取类别张量
#         label_tensor = self.get_label_tensor(row)
#
#         return image, label_tensor

class EyesDataset(Dataset):
    def __init__(self, csv_file, img_prefix, transform=None):
        """
        初始化 Dataset
        :param csv_file: str, CSV 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.data)

    def get_label_index(self, row):
        """
        根据标签列返回类别索引
        :param row: DataFrame 的一行
        :return: int, 类别索引
        """
        for label, index in self.labels.items():
            if row[label] == 1:
                return index
        return -1  # 如果没有疾病标签，则返回 -1

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: int, 数据索引
        :return: tuple (image, label_index)
        """
        idx = int(idx)
        if isinstance(idx, int):
            row = self.data.iloc[idx]
        else:
            raise TypeError("Index must be an integer.")

        # 构造图片路径
        img_name = row.iloc[0]  # 第一列是图片的名字
        img_path = f"{self.img_prefix}/{img_name}"

        # 打开图片
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 数据增强和预处理
        if self.transform:
            image = self.transform(image)

        # 获取类别索引
        label_index = self.get_label_index(row)

        return image, label_index


class EyesDatasetMultiClass(Dataset):
    def __init__(self, csv_file, img_prefix, transform=None):
        """
        初始化 Dataset
        :param csv_file: str, CSV 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.data)

    def get_label(self, row):
        """
        根据标签列返回类别索引的张量
        :param row: DataFrame 的一行
        :return: tensor, 8*1 的张量表示每个类别的存在情况
        """
        label_tensor = torch.zeros(8)  # 初始化一个长度为8的张量

        for label, index in self.labels.items():
            if row[label] == 1:  # 假设标签列直接使用了字典中的键作为列名
                label_tensor[index] = 1  # 如果存在该疾病，则将对应位置置为1

        return label_tensor  # 调整形状为8*1并返回

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: int, 数据索引
        :return: tuple (image, label_index)
        """
        idx = int(idx)
        if isinstance(idx, int):
            row = self.data.iloc[idx]
        else:
            raise TypeError("Index must be an integer.")

        # 构造图片路径
        img_name = row.iloc[0]  # 第一列是图片的名字
        img_path = f"{self.img_prefix}/{img_name}"

        # 打开图片
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 数据增强和预处理
        if self.transform:
            image = self.transform(image)

        # 获取类别索引
        label_index = self.get_label(row)

        return image, label_index

def create_custom_dataset_with_imagefolder(data_path, transform):


    # Use ImageFolder to create the dataset
    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    return len(dataset.classes)


# if __name__ == "__main__":
#     transform = transforms.Compose([
#         transforms.Resize((112, 112)),  # 调整图片大小
#         transforms.ToTensor(),  # 转换为 Tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
#     ])
#     dataset = torchvision.datasets.ImageFolder(root=r"D:\BaiduNetdiskDownload\服务外包\OtherDataset\dataset", transform=transform)
#
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for batch in dataloader:
#         # print(batch)
#         images, labels = batch
#         print(images.shape)
#         print(labels)

if __name__ == "__main__":
    # 定义数据增强和预处理
    transform = transforms.Compose([

        transforms.Resize((112, 112)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 初始化自定义数据集
    dataset = EyesDataset(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\Left_group_0_or_1_diseases_no1209_left_7class.csv",
                            img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Enhanced",
                            transform=transform)
    # datasetMultiClass = EyesDatasetMultiClass(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\Left_Fundus_Classification_no_invalid_no1209_left.csv",
    #                         img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Enhanced",
    #                         transform=transform)
    # 遍历一下datasetMultiClass，注意是dataset，不是dataloader
    # for item in datasetMultiClass:
    #     data, label = item
    #     print(f"data: {data.shape}, label: {label}")


    # 定义 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    print("dataset", len(dataset))
    # dataloaderMultiClass = DataLoader(datasetMultiClass, batch_size=2, shuffle=True, num_workers=0)
    # print("dataloader", len(dataloader))


    # 测试 DataLoader
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Images shape: {images.shape}")  # torch.Size([2, 3, 112, 112])
        print(f"Labels: {labels}")  # torch.Size([2])


"""
/data3/wangchangmiao/shenxy/Enhanced
"""