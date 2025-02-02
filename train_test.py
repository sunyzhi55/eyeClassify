import time
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from model import *
from Resnet import *

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Use the original optimizer
# 训练
def train_sector(net, device, train_load, loss_fn, optimizer, epoch, writer):
    net.train()
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    for batch_index, (image, label) in enumerate(tqdm(train_load, desc=f'Epoch {epoch}', unit='batch')):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(image)
        loss = loss_fn(output, label)
        total_train_loss += loss.item()
        accuracy = (output.argmax(dim=1) == label).sum().item()
        total_train_accuracy += accuracy
        loss.backward()
        optimizer.step()
        # if batch_index % 100 == 0:
        #     print(f'Train Epoch:{epoch} [{batch_index}/{len(train_load)}] loss:{loss.item():.6f}')
    total_train_loss /= len(train_load.dataset)
    total_train_accuracy /= len(train_load.dataset)
    writer.add_scalar("train_loss", total_train_loss, epoch)
    writer.add_scalar("train_accuracy", total_train_accuracy, epoch)
    print(f"Train Average Loss:{total_train_loss},Accuracy:{total_train_accuracy}")

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

# 测试
def test_sector(net, device, test_load, loss_fn, epoch, writer):
    net.eval()
    total_test_loss = 0.0
    total_test_accuracy = 0.0
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(tqdm(test_load, desc=f'Epoch {epoch}', unit='batch')):
            image, label = image.to(device), label.to(device)
            output = net(image)
            loss = loss_fn(output, label)
            total_test_loss += loss.item()
            accuracy = (output.argmax(dim=1) == label).sum().item()
            total_test_accuracy += accuracy

        total_test_loss /= len(test_load.dataset)
        total_test_accuracy /= len(test_load.dataset)
        writer.add_scalar("test_loss", total_test_loss, epoch)
        writer.add_scalar("test_accuracy", total_test_accuracy, epoch)
        print(f"Test Average Loss:{total_test_loss},Accuracy:{total_test_accuracy}")


def k_fold_cross_validation(device, eyes_dataset, loss_fn, epochs, args, lr, writer, k_fold=5,
                            batch_size=8, workers=2, print_freq=1, checkpoint_dir="./result",
                            best_result_model_path="model", total_transform=None):
    start = time.time()
    # skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    # best_fold_metrics = []
    best_acc_overall = 0.
    epoch_acc_dict = {}

    # 获取数据集的标签
    # labels = [data[1] for data in eyes_dataset]  # 假设dataset[i]的第2项是label
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=0)  # init KFold
    # Iterations = 1
    for fold, (train_index, test_index) in enumerate(kf.split(eyes_dataset)):  # split
    # for fold, (train_idx, val_idx) in enumerate(skf.split(eyes_dataset, labels), 1):

        # 准备数据
        # train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        # val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        # train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
        #                               num_workers=workers)
        # eval_dataloader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size,
        #                              num_workers=workers)

        # get train, val
        k_train_fold = Subset(eyes_dataset, train_index)
        k_test_fold = Subset(eyes_dataset, test_index)
        # 应用转换
        train_dataset = TransformedSubset(k_train_fold, transform=total_transform['train_transforms'])
        val_dataset = TransformedSubset(k_test_fold, transform=total_transform['validation_transforms'])


        # package type of DataLoader
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        # model = poolformer_s12(num_classes=1000)
        # # exit()
        # model.load_state_dict(torch.load(args.pretrainedModelPath, weights_only=True))
        # model.head = torch.nn.Linear(model.head.in_features, 8)
        # model = model.to(device)

        model = resnet34(num_classes=1000)
        model.load_state_dict(torch.load(args.pretrainedModelPath, weights_only=True))
        model.fc = torch.nn.Linear(model.fc.in_features, 8)  # 修改全连接层
        model = model.to(device)


        # model = EyeSeeNet(num_class=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.98))
        scheduler = get_scheduler(optimizer, args)

        total_params = sum(p.numel() for p in model.parameters())
        print('总参数个数:{}'.format(total_params))


        # train model
        # test_acc = torch.zeros([Iterations])
        # test_acc = 0.0

        best_epoch = 0
        best_acc = 0.
        best_metrics = {}
        for e in range(1, epochs + 1):
            model.train()
            total_train_loss = 0.0
            total_eval_loss = 0.0
            total_train_accuracy = 0.0
            total_eval_accuracy = 0.0
            # 获取当前学习率（假设只有一个参数组）
            current_lr = optimizer.param_groups[0]['lr']

            # y_train_true = []
            # y_train_logit = []
            # y_train_prob = []
            train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {e}, LR {current_lr:.6f}", unit="batch")
            for batch_idx, (images, labels) in enumerate(train_iterator):
                images, labels = images.to(device), labels.to(device)
                # print(f"image:{images.shape}")
                # print(f"labels:{labels}")
                optimizer.zero_grad()

                predictions = model(images)
                # print("predictions", predictions.shape)
                # print("labels", labels.shape)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

                # _, train_predicted = torch.max(predictions.data, 1)
                accuracy = (predictions.argmax(dim=1) == labels).sum().item()
                total_train_accuracy += accuracy
                # y_train_true.extend(labels.cpu().detach().numpy())
                # y_train_logit.extend(train_predicted.cpu().detach().numpy())
                # y_train_prob.extend(train_prob[:, 1].cpu().detach().numpy())

            if scheduler:
                scheduler.step()

            # 验证
            # y_val_true = []
            # y_val_logit = []
            # y_val_prob = []

            with torch.no_grad():
                model.eval()
                eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
                for batch_idx, (images, labels) in enumerate(eval_iterator):
                    images, labels = images.to(device), labels.to(device)
                    predictions = model(images)
                    loss = loss_fn(predictions, labels)
                    total_eval_loss += loss.item()

                    # _, val_predicted = torch.max(predictions.data, 1)
                    accuracy = (predictions.argmax(dim=1) == labels).sum().item()
                    total_eval_accuracy += accuracy
                    # y_val_true.extend(labels.cpu().detach().numpy())
                    # y_val_logit.extend(val_predicted.cpu().detach().numpy())
                    # y_val_prob.extend(eval_prob[:, 1].cpu().detach().numpy())

            # train_metrics = estimate(y_train_true, y_train_logit, y_train_prob)
            # val_metrics = estimate(y_val_true, y_val_logit, y_val_prob)
            # (train_tp, train_fp, train_fn, train_tn,
            #  train_acc, train_recall, train_precision, train_f1_score,
            #  train_specificity, train_sensitivity, train_auc, train_mcc) = train_metrics
            # (val_tp, val_fp, val_fn, val_tn,
            #  val_acc, val_recall, val_precision, val_f1_score,
            #  val_specificity, val_sensitivity, val_auc, val_mcc) = val_metrics
            total_train_loss /= len(train_dataloader.dataset)
            total_train_accuracy /= len(train_dataloader.dataset)
            total_eval_loss /= len(eval_dataloader.dataset)
            total_eval_accuracy /= len(eval_dataloader.dataset)


            if total_eval_accuracy > best_acc:
                best_acc = total_eval_accuracy
                best_epoch = e
                # best_metrics = {
                #     'tp': val_tp, 'fp': val_fp, 'fn': val_fn, 'tn': val_tn,
                #     'accuracy': val_acc, 'recall': val_recall, 'precision': val_precision,
                #     'f1_score': val_f1_score, 'specificity': val_specificity,
                #     'sensitivity': val_sensitivity, 'auc': val_auc, 'mcc': val_mcc
                # }
                torch.save(model.state_dict(), checkpoint_dir + f'/{best_result_model_path}_fold{fold}.pth')

            if e % print_freq == 0:
                print(f'Epoch [{e}/{epochs}]: train_loss={total_train_loss:.3f}, train_acc={total_train_accuracy:.3f},'
                      f'eval_loss={total_eval_loss:.3f}, eval_acc={total_eval_accuracy:.3f}')
                # # 打印当前 epoch 的信息
                # output_result = f'Epoch [{e}/{epochs}]\n' \
                #                 f'Train Loss: {train_loss:.3f}\n' \
                #                 f'TP:{train_tp}\t FP:{train_fp}\t FN:{train_fn}\t TN:{train_tn}\n' \
                #                 f'Train Accuracy: {train_acc:.3f}\tTrain recall: {train_recall:.3f}\t' \
                #                 f'Train precision: {train_precision:.3f}\tTrain F1 Score: {train_f1_score:.3f}\t' \
                #                 f'Train Specificity: {train_specificity:.3f}\tTrain Sensitivity:{train_sensitivity:.3f}\n' \
                #                 f'Train Auc: {train_auc:.3f}\tTrain MCC: {train_mcc:.3f}\n' \
                #                 f'Val Loss: {val_loss:.3f}\n' \
                #                 f'TP:{val_tp}\t FP:{val_fp}\t FN:{val_fn}\t TN:{val_tn}\n' \
                #                 f'Val Accuracy: {val_acc:.3f}\tVal recall: {val_recall:.3f}\t' \
                #                 f'Val Precision: {val_precision:.3f}\tVal F1 Score: {val_f1_score:.3f}\t' \
                #                 f'Val Specificity: {val_specificity:.3f}\tVal Sensitivity:{val_sensitivity:.3f}\n' \
                #                 f'Val Auc: {val_auc:.3f}\tVal MCC: {val_mcc:.3f}\n'
                # print(output_result)
                #
                # # 将完整的字符串写入文件
                # with open(f'{log_dir}/{train_log_path}', 'a') as f:
                #     f.write(output_result + '\n')  # 在最后添加一个换行符以保持格式整洁



        # best_fold_metrics.append(best_metrics)
        print(f"Fold {fold} Best Epoch: {best_epoch}, Best Val Acc: {best_acc:.3f}")
        epoch_acc_dict[fold] = best_acc

        if best_acc > best_acc_overall:
            best_acc_overall = best_acc

    end = time.time()
    print(f"Cross-validation result:{epoch_acc_dict}")

    # with open(f'{log_dir}/{train_log_path}', 'a') as f:
    #     f.write("Cross-Validation Best Fold Results:\n")
    #     for fold, metrics in enumerate(best_fold_metrics, 1):
    #         fold_result = (f"Fold {fold}: "
    #                        f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}, \n"
    #                        f"Accuracy: {metrics['accuracy']:.3f}, Recall: {metrics['recall']:.3f}, "
    #                        f"Precision: {metrics['precision']:.3f}, F1 Score: {metrics['f1_score']:.3f}, \n"
    #                        f"Specificity: {metrics['specificity']:.3f}, Sensitivity: {metrics['sensitivity']:.3f}, "
    #                        f"AUC: {metrics['auc']:.3f}, MCC: {metrics['mcc']:.3f}\n")
    #         print(fold_result)
    #         f.write(fold_result)

    print(f"Total training time: {(end - start) // 60}m {(end - start) % 60}s")