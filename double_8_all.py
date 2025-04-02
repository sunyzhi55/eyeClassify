from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import KFold
from model import *
from utils.metrics import *
from utils.basic import get_scheduler
import torch
class SingleImageModel7Class(nn.Module):
  def __init__(self, num_classes, pretrained_path):
    super().__init__()
    # self.model1 = build_efficientVit.EfficientViT_M5(pretrained='efficientvit_m5')
    self.model1 = build_efficientVit.EfficientViT_M0(pretrained='efficientvit_m0')
    self.model1.head[1] = nn.Linear(192, num_classes)

  def forward(self, image):
    # x = left_right_concat[:, :3, :, :]
    # y = left_right_concat[:, 3:, :, :]
    result = self.model1(image)
    # y1 = self.model1(y)
    # fusion = torch.cat((x1, y1), dim=1)
    return result

class DoubleTransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        batch = self.subset[index]
        if self.transform:
            batch['left_image'] = self.transform(batch['left_image'])
            batch['right_image'] = self.transform(batch['right_image'])
        return batch
    def __len__(self):
        return len(self.subset)

def all_train_double_8_class(device, eyes_dataset, args, workers=2, print_freq=1,
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
    all_dataset = DoubleTransformedSubset(eyes_dataset, transform=total_transform['train_transforms'])
    all_dataloader = torch.utils.data.DataLoader(dataset=all_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    num_classes = 7
    model_single_eye = SingleImageModel7Class(num_classes=num_classes, pretrained_path=args.pretrainedModelPath)
    head = nn.Sequential(
      nn.Linear(2 * num_classes, 8),
      nn.BatchNorm1d(8),
      nn.ReLU(),
      # nn.Dropout(),
      nn.Linear(8, num_classes)
    ).to(device)
    model_single_eye = model_single_eye.to(device)
    model = nn.Sequential(model_single_eye, head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
    scheduler = get_scheduler(optimizer, args)
    # 损失函数
    ratio = torch.zeros(8)
    Count = 0
    for batch in all_dataloader:
        # print("batch['total_label_index_float']", batch['total_label_index_float'])
        ratio += torch.sum(batch['total_label_index_float'], dim=0)
        Count += batch['total_label_index_float'].shape[0]
    print("各疾病总数:", ratio, Count)
    ratio = Count / ratio - 1
    print("各疾病损失权重:", ratio)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=ratio[1:]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print('总参数个数:{}'.format(total_params))
    metrics_mul_label = MetricsWithMultiLabelWithTorchmetrics(num_classes=8, device=device)
    for e in range(1, epochs + 1):
        model_single_eye.train()
        head.train()
        # model.train()
        metrics_mul_label.reset()
        # 获取当前学习率（假设只有一个参数组）
        current_lr = optimizer.param_groups[0]['lr']
        all_iterator = tqdm(all_dataloader, desc=f"Training Epoch {e}, LR {current_lr:.6f}", unit="batch")
        for batch in all_iterator:
            left_image, right_image = batch['left_image'], batch['right_image']
            left_eye_label_index_int, left_eye_label_index_float = batch['left_eye_label_index_int'], batch['left_eye_label_index_float'],
            right_eye_label_index_int, right_eye_label_index_float = batch['right_eye_label_index_int'], batch['right_eye_label_index_float'],
            total_label_index_int, total_label_index_float = batch['total_label_index_int'], batch['total_label_index_float']
            left_image, right_image, = left_image.to(device), right_image.to(device)
            total_label_index_int, total_label_index_float = (total_label_index_int.to(device),
                                                              total_label_index_float.to(device))
            left_eye_label_index_int, left_eye_label_index_float = (left_eye_label_index_int.to(device),
                                                                    left_eye_label_index_float.to(device))
            right_eye_label_index_int, right_eye_label_index_float = (right_eye_label_index_int.to(device),
                                                                      right_eye_label_index_float.to(device))
            optimizer.zero_grad()
            left_logit = model_single_eye(left_image)
            right_logit = model_single_eye(right_image)
            logit = head(torch.cat((left_logit, right_logit), dim=1))
            loss_1 = loss_fn(left_logit, left_eye_label_index_float[:, 1:])
            loss_2 = loss_fn(right_logit, right_eye_label_index_float[:, 1:])
            loss_3 = loss_fn(logit, total_label_index_float[:, 1:])
            loss = (loss_1 + loss_2 + loss_3) / 3.0
            loss.backward()
            optimizer.step()
            metrics_mul_label.train_update(loss, logit, total_label_index_int)
        if scheduler:
            scheduler.step()
        with torch.no_grad():
            model_single_eye.eval()
            head.eval()
            # model.eval()
            eval_iterator = tqdm(all_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
            for batch in eval_iterator:
                left_image, right_image = batch['left_image'], batch['right_image']
                left_eye_label_index_int, left_eye_label_index_float = batch['left_eye_label_index_int'], batch['left_eye_label_index_float'],
                right_eye_label_index_int, right_eye_label_index_float = batch['right_eye_label_index_int'], batch['right_eye_label_index_float'],
                total_label_index_int, total_label_index_float = batch['total_label_index_int'], batch['total_label_index_float']

                left_image, right_image, = left_image.to(device), right_image.to(device)
                total_label_index_int, total_label_index_float = (total_label_index_int.to(device),
                                                                  total_label_index_float.to(device))
                left_eye_label_index_int, left_eye_label_index_float = (left_eye_label_index_int.to(device),
                                                                        left_eye_label_index_float.to(device))
                right_eye_label_index_int, right_eye_label_index_float = (right_eye_label_index_int.to(device),
                                                                          right_eye_label_index_float.to(device))
                left_logit = model_single_eye(left_image)
                right_logit = model_single_eye(right_image)
                logit = head(torch.cat((left_logit, right_logit), dim=1))
                loss_1 = loss_fn(left_logit, left_eye_label_index_float[:, 1:])
                loss_2 = loss_fn(right_logit, right_eye_label_index_float[:, 1:])
                loss_3 = loss_fn(logit, total_label_index_float[:, 1:])
                loss = (loss_1 + loss_2 + loss_3) / 3.0
                metrics_mul_label.eval_update(loss, logit, total_label_index_int)
        metrics_mul_label.compute_result()
        metrics_mul_label.average_train_loss = metrics_mul_label.total_train_loss / len(all_dataloader.dataset)
        metrics_mul_label.average_eval_loss = metrics_mul_label.total_eval_loss / len(all_dataloader.dataset)
        if metrics_mul_label.get_best(e):
            torch.save(model_single_eye.state_dict(), checkpoint_dir + f'/{best_result_model_path}_single_eye.pth')
            torch.save(head.state_dict(), checkpoint_dir + f'/{best_result_model_path}_head.pth')
        if e % print_freq == 0:
            metrics_mul_label.print_result(e, epochs)
    metrics_mul_label.print_best_result(0)
    epoch_acc_dict[0] = metrics_mul_label.best_dicts['average_accuracy']
    print(f"Cross-validation result:{epoch_acc_dict}")