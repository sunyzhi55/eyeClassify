from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from model import *
from utils.metrics import *
from utils.basic import get_scheduler
import torch
from samUtility import *

class DoubleTransformedSubset(Dataset):
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

def k_fold_cross_validation_double_8_class_sam(device, eyes_dataset, args, workers=2, print_freq=1,
                                            best_result_model_path="model", total_transform=None):
    # skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    checkpoint_dir = args.checkpoint_dir
    epoch_acc_dict = {}

    # 获取数据集的标签
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
        train_dataset = DoubleTransformedSubset(k_train_fold, transform=total_transform['train_transforms'])
        val_dataset = DoubleTransformedSubset(k_test_fold, transform=total_transform['validation_transforms'])
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                       shuffle=True, drop_last=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                      shuffle=False)

        model = DoubleImageModel7Class(num_classes=7, pretrained_path=args.pretrainedModelPath)
        model = model.to(device)

        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
        # scheduler = get_scheduler(optimizer, args)

        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

        # 损失函数
        ratio = torch.zeros(8)
        Count = 0
        for (_, _, labels_int, labels_float) in train_dataloader:
            ratio += torch.sum(labels_float, dim=0)
            Count += labels_float.shape[0]
        print("各疾病总数:", ratio, Count)
        ratio = Count / ratio - 1
        print("各疾病损失权重:", ratio)
        # X = torch.tensor([12.5706])
        # loss_fn = nn.BCEWithLogitsLoss().to(device)  # 损失函数
        # pos_weight_for_7_disease = torch.tensor([6.89, 16.51, 17.47, 19.88, 38.74, 22.61, 3.55])  # true
        # pos_weight_for_7_disease = torch.tensor([6.89, 16.51, 17.47, 10.0, 18.0, 22.61, 3.55]) # false
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_for_7_disease).to(device)  # 损失函数
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=ratio[1:]).to(device)
        # loss_fn = FocalLoss(device=device, position_weight=ratio[1:]).to(device)

        # loss_fn = nn.CrossEntropyLoss().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print('总参数个数:{}'.format(total_params))

        # metrics_single_label = MetricsWithSingleLabel(num_classes=2, device=device)
        metrics_mul_label = MetricsWithMultiLabelWithTorchmetrics(num_classes=8, device=device)
        for e in range(1, epochs + 1):
            model.train()
            metrics_mul_label.reset()
            # 获取当前学习率（假设只有一个参数组）
            current_lr = optimizer.param_groups[0]['lr']
            train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {e}, LR {current_lr:.6f}", unit="batch")
            for batch_idx, (left_image, right_image, label_index_2d_int, label_index_2d_float) in enumerate(train_iterator):
                left_image, right_image, = left_image.to(device), right_image.to(device)
                label_index_2d_int, label_index_2d_float = label_index_2d_int.to(device), label_index_2d_float.to(device)
                # first forward-backward step
                enable_running_stats(model)
                logit = model(left_image, right_image)
                loss = loss_fn(logit, label_index_2d_float[:, 1:])
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                # second forward-backward step
                disable_running_stats(model)
                loss_fn(model(left_image, right_image), label_index_2d_float[:, 1:]).mean().backward()
                optimizer.second_step(zero_grad=True)

                metrics_mul_label.train_update(loss, logit, label_index_2d_int)
            if scheduler:
                scheduler(e)
            with torch.no_grad():
                model.eval()
                eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
                for batch_idx, (left_image, right_image, label_index_2d_int, label_index_2d_float) in enumerate(eval_iterator):
                    left_image, right_image, = left_image.to(device), right_image.to(device)
                    label_index_2d_int, label_index_2d_float = label_index_2d_int.to(device), label_index_2d_float.to(device)
                    logit = model(left_image, right_image)
                    # _, predictions = torch.max(prob, dim=1)
                    # prob_positive = prob[:, 1]
                    loss = loss_fn(logit, label_index_2d_float[:, 1:])
                    metrics_mul_label.eval_update(loss, logit, label_index_2d_int)
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