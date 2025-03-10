from cProfile import label

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

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

    # 单标签分类 label
    # def get_label_index(self, row):
    #     """
    #     根据标签列返回类别索引
    #     :param row: DataFrame 的一行
    #     :return: int, 类别索引
    #     """
    #     for label, index in self.labels.items():
    #         if row[label] == 1:
    #             return index
    #     return -1  # 如果没有疾病标签，则返回 -1

    # 多标签分类 label
    def get_label_index(self, row):
        """
        根据标签列返回类别索引
        :param row: DataFrame 的一行
        :return: int, 类别索引
        """
        label_index = torch.zeros(8)
        for label, index in self.labels.items():
            if row[label] == 1:
                label_index[index] = 1
        # label_index = label_index[1:]

        return label_index

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

class DoubleEyesDataset(Dataset):
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
        label_index = torch.zeros(8)
        for label, index in self.labels.items():
            if row[label] == 1:
                label_index[index] = 1

        return label_index

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
        # print(row)
        left_img_name = row.iloc[3]
        left_img_path = f"{self.img_prefix}/{left_img_name}"

        right_img_name = row.iloc[4]
        right_img_path = f"{self.img_prefix}/{right_img_name}"

        # 打开图片
        left_image = Image.open(left_img_path).convert('RGB')
        right_image = Image.open(right_img_path).convert('RGB')

        # 数据增强和预处理
        # if self.transform:
        #     image = self.transform(image)

        # 获取类别索引
        label_index = self.get_label_index(row)

        return left_image, right_image, label_index

class DoubleEyesDatasetTwoClass(Dataset):
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

    # def get_label_index(self, row):
    #     """
    #     根据标签列返回类别索引
    #     :param row: DataFrame 的一行
    #     :return: int, 类别索引
    #     """
    #     label_index = torch.zeros(8)
    #     for label, index in self.labels.items():
    #         if row[label] == 1:
    #             label_index[index] = 1
    #
    #     return label_index

    # 单标签分类 label
    def get_label_index(self, row):
        """
        根据标签列返回类别索引
        :param row: DataFrame 的一行
        :return: int, 类别索引
        """

        if row['N'] == 1:
            label_1d = 0
        else:
            label_1d = 1

        label_index_2d = torch.zeros(2)
        if row['N'] == 1:
            label_index_2d[0] = 1
        else:
            label_index_2d[1] = 1
        return label_1d, label_index_2d

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
        # print(row)
        left_img_name = row.iloc[3]
        left_img_path = f"{self.img_prefix}/{left_img_name}"

        right_img_name = row.iloc[4]
        right_img_path = f"{self.img_prefix}/{right_img_name}"

        # 打开图片
        left_image = Image.open(left_img_path).convert('RGB')
        right_image = Image.open(right_img_path).convert('RGB')

        # 数据增强和预处理
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        # 获取类别索引
        label_1d, label_index_2d = self.get_label_index(row)

        return left_image, right_image, label_1d, label_index_2d

class DoubleEyesDatasetAllClass(Dataset):
    def __init__(self, csv_file, img_prefix, transform=None):
        """
        初始化 Dataset
        :param csv_file: str, CSV 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.data = pd.read_csv(csv_file)

        # 过滤掉 'N' == 1 的行
        self.data = self.data[self.data['N'] != 1]

        self.img_prefix = img_prefix
        # self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        self.labels = {'D': 0, 'G': 1, 'C': 2, 'A': 3, 'H': 4, 'M': 5, 'O': 6}
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
        # label_index = torch.zeros(7)
        # # if row['N'] == 1:
        # #     pass
        # # else:
        # for label, index in self.labels.items():
        #     if row[label] == 1:
        #         label_index[index] = 1
        #
        # return label_index

        label_index_int = torch.zeros(7, dtype=torch.int)
        label_index_float = torch.zeros(7, dtype=torch.float)

        for label, index in self.labels.items():
            if row[label] == 1:
                label_index_int[index] = 1
                label_index_float[index] = 1.0

        return label_index_int, label_index_float

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
        # print(row)
        left_img_name = row.iloc[3]
        left_img_path = f"{self.img_prefix}/{left_img_name}"

        right_img_name = row.iloc[4]
        right_img_path = f"{self.img_prefix}/{right_img_name}"

        # 打开图片
        left_image = Image.open(left_img_path).convert('RGB')
        right_image = Image.open(right_img_path).convert('RGB')

        # 数据增强和预处理
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        # 获取类别索引
        label_index_int, label_index_float = self.get_label_index(row)

        return left_image, right_image, label_index_int, label_index_float


if __name__ == "__main__":
    # 定义数据增强和预处理
    transform = transforms.Compose([

        transforms.Resize((112, 112)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 初始化自定义数据集
    # dataset = EyesDataset(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\Left_group_0_or_1_diseases_no1209_left_7class.csv",
    #                         img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced",
    #                         transform=transform)
    dataset = DoubleEyesDatasetTwoClass(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\Traning_Dataset.csv",
                            img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced",
                            transform=transform)
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
    # for batch_idx, (images, labels) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"Images shape: {images.shape}")  # torch.Size([2, 3, 112, 112])
    #     print(f"Labels: {labels}")  # torch.Size([2])

    # 测试 DataLoader
    for batch_idx, (left_image, right_image, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"left image shape: {left_image.shape}")  # torch.Size([2, 3, 112, 112])
        print(f"right image shape: {right_image.shape}")  # torch.Size([2, 3, 112, 112])
        print(f"Labels: {labels.shape}")  # torch.Size([2])


"""
/data3/wangchangmiao/shenxy/Enhanced
"""

