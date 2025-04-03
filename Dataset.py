import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

    def crop_black_pixels(self, image):
        """
        裁剪图像中的黑色像素
        :param image: PIL.Image, 输入图像
        :return: PIL.Image, 裁剪后的图像
        """
        image_np = np.array(image)
        # 创建掩码，标记非黑色像素
        mask = image_np > 0
        # 找到非黑色像素的坐标
        coordinates = np.argwhere(mask)
        # 计算边界框
        x0, y0, _ = coordinates.min(axis=0)
        x1, y1, _ = coordinates.max(axis=0) + 1  # 切片是排他的，所以需要加 1
        # 裁剪图像
        cropped_image_np = image_np[x0:x1, y0:y1]
        cropped_image = Image.fromarray(cropped_image_np)
        return cropped_image

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
        # 裁剪黑色像素
        image = self.crop_black_pixels(image)
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
        # self.data = self.data[self.data['N'] != 1]

        self.img_prefix = img_prefix
        self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        # self.labels = {'D': 0, 'G': 1, 'C': 2, 'A': 3, 'H': 4, 'M': 5, 'O': 6}
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.data)
    def crop_black_pixels(self, image):
        """
        裁剪图像中的黑色像素
        :param image: PIL.Image, 输入图像
        :return: PIL.Image, 裁剪后的图像
        """
        image_np = np.array(image)
        # 创建掩码，标记非黑色像素
        mask = image_np > 0
        # 找到非黑色像素的坐标
        coordinates = np.argwhere(mask)
        # 计算边界框
        x0, y0, _ = coordinates.min(axis=0)
        x1, y1, _ = coordinates.max(axis=0) + 1  # 切片是排他的，所以需要加 1
        # 裁剪图像
        cropped_image_np = image_np[x0:x1, y0:y1]
        cropped_image = Image.fromarray(cropped_image_np)
        return cropped_image

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

        label_index_int = torch.zeros(8, dtype=torch.int)
        label_index_float = torch.zeros(8, dtype=torch.float)

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
        # 裁剪黑色像素
        # left_image = self.crop_black_pixels(left_image)
        # right_image = self.crop_black_pixels(right_image)

        # 数据增强和预处理
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        # 获取类别索引
        label_index_int, label_index_float = self.get_label_index(row)

        return left_image, right_image, label_index_int, label_index_float

class DoubleEyesDatasetAllClassLRTotal(Dataset):
    def __init__(self, total_csv_file, single_eye_csv_file, img_prefix, transform=None):
        """
        初始化 Dataset
        :param csv_file: str, CSV 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.total_data = pd.read_csv(total_csv_file)
        self.single_eye_csv_file = single_eye_csv_file

        # 过滤掉 'N' == 1 的行
        # self.data = self.data[self.data['N'] != 1]

        self.img_prefix = img_prefix
        self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        # self.labels = {'D': 0, 'G': 1, 'C': 2, 'A': 3, 'H': 4, 'M': 5, 'O': 6}
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.total_data)

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

        label_index_int = torch.zeros(8, dtype=torch.int)
        label_index_float = torch.zeros(8, dtype=torch.float)

        for label, index in self.labels.items():
            if row[label] == 1:
                label_index_int[index] = 1
                label_index_float[index] = 1.0

        return label_index_int, label_index_float

    def find_matching_rows_with_labels(self, csv_file, target_string):
        """
        在 CSV 文件中找到第一列与目标字符串相等的行，并返回对应的类别标签。

        :param csv_file: str, CSV 文件路径
        :param target_string: str, 要匹配的目标字符串（图片名称）
        :return: torch.Tensor, 长度为 8 的张量，表示类别标签
        """
        # 加载 CSV 文件
        df = pd.read_csv(csv_file)

        # 获取第一列的列名
        first_column = df.columns[0]

        # 筛选第一列等于目标字符串的行
        matching_rows = df[df[first_column] == target_string]

        # 如果没有匹配的行，抛出异常或返回全零张量
        if matching_rows.empty:
            print(f"No matching row found for img_name: {target_string}")
            return torch.zeros(8, dtype=torch.int)

        # 提取匹配的第一行（假设每张图片唯一对应一行）
        matched_row = matching_rows.iloc[0]

        # 定义类别顺序
        label_columns = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

        # 初始化类别标签张量

        label_index_int = torch.zeros(8, dtype=torch.int)
        label_index_float = torch.zeros(8, dtype=torch.float)

        # 遍历每一列，如果值为 1，则在对应位置赋值为 1
        for i, col in enumerate(label_columns):
            if matched_row[col] == 1:
                label_index_int[i] = 1
                label_index_float[i] = 1.0

        return label_index_int, label_index_float

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: int, 数据索引
        :return: tuple (image, label_index)
        """
        idx = int(idx)
        if isinstance(idx, int):
            row = self.total_data.iloc[idx]
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
        total_label_index_int, total_label_index_float = self.get_label_index(row)
        left_eye_label_index_int, left_eye_label_index_float = self.find_matching_rows_with_labels(self.single_eye_csv_file, left_img_name)
        right_eye_label_index_int, right_eye_label_index_float = self.find_matching_rows_with_labels(self.single_eye_csv_file, right_img_name)
        data_dictionary = {
            'left_image': left_image,
            'left_eye_label_index_int': left_eye_label_index_int,
            'left_eye_label_index_float': left_eye_label_index_float,
            'right_image': right_image,
            'right_eye_label_index_int': right_eye_label_index_int,
            'right_eye_label_index_float': right_eye_label_index_float,
            'total_label_index_int': total_label_index_int,
            'total_label_index_float': total_label_index_float
        }
        return data_dictionary

if __name__ == "__main__":
    # 定义数据增强和预处理
    transform = transforms.Compose([

        transforms.Resize((112, 112)),  # 调整图片大小
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 初始化自定义数据集
    # dataset = EyesDataset(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\old\Left_group_0_or_1_diseases_no1209_left.csv",
    #                         img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced",
    #                         transform=transform)
    dataset = DoubleEyesDatasetAllClassLRTotal(
        total_csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\select.csv",
        single_eye_csv_file = r"D:\BaiduNetdiskDownload\服务外包\csv\old\total_valid.csv" ,
        img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced",
        transform=transform)
    # dataset = DoubleEyesDatasetTwoClass(csv_file=r"D:\BaiduNetdiskDownload\服务外包\csv\old\Traning_Dataset.csv",
    #                         img_prefix=r"D:\BaiduNetdiskDownload\服务外包\Data\Enhanced",
    #                         transform=transform)
    # 遍历一下datasetMultiClass，注意是dataset，不是dataloader
    # for item in datasetMultiClass:
    #     data, label = item
    #     print(f"data: {data.shape}, label: {label}")


    # 定义 DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print("dataset", len(dataset))
    # dataloaderMultiClass = DataLoader(datasetMultiClass, batch_size=2, shuffle=True, num_workers=0)
    # print("dataloader", len(dataloader))

    # 测试 DataLoader
    # for batch_idx, (images, labels) in enumerate(dataloader):
    #     images = images.view(3, 112, 112).permute(1, 2, 0)
    #
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(images)
    #     plt.tight_layout()
    #     plt.savefig('input.png')
    #
    #     print(f"Batch {batch_idx}:")
    #     print(f"Images shape: {images.shape}")  # torch.Size([2, 3, 112, 112])
    #     print(f"Labels: {labels}")  # torch.Size([2])

    # 测试 DataLoader
    # for batch_idx, (left_image, right_image, label_1d, label_index_2d) in enumerate(dataloader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"left image shape: {left_image.shape}")  # torch.Size([2, 3, 112, 112])
    #     print(f"right image shape: {right_image.shape}")  # torch.Size([2, 3, 112, 112])
    #     print(f"label_1d: {label_1d}")  # torch.Size([2])
    #     print(f"label_index_2d: {label_index_2d}")

    for batch in dataloader:
        print("batch['total_label_index_float']", batch['total_label_index_float'])



"""
/data3/wangchangmiao/shenxy/Enhanced
"""

