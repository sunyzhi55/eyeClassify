import torchmetrics
import numpy as np
import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelRecall, MultilabelSpecificity
from torchmetrics.classification import MultilabelPrecision, MultilabelF1Score, MultilabelConfusionMatrix
def get_metrics(predictions, labels, normal=False):
    TP = TN = FP = FN = 0
    if not normal:
        for index in range(len(predictions)):
            pred = predictions[index]
            actual = labels[index]
            if pred == 1 and actual == 1:
                TP += 1
            elif pred == 0 and actual == 0:
                TN += 1
            elif pred == 1 and actual == 0:
                FP += 1
            elif pred == 0 and actual == 1:
                FN += 1
            # print(pred)
    else:
        # print("predictions", len(predictions)) # 64
        for index in range(len(predictions)):
            if 1 in predictions[index]:
                pred = 0
            else:
                pred = 1
            actual = labels[index]

            # print(pred, actual)

            if pred == 1 and actual == 1:
                TP += 1
            elif pred == 0 and actual == 0:
                TN += 1
            elif pred == 1 and actual == 0:
                FP += 1
            elif pred == 0 and actual == 1:
                FN += 1

    return TP, TN, FP, FN

def to0_1(tensor):
    # 阈值
    threshold = 0.5
    bool_tensor = tensor > threshold

    # 将布尔型张量转换为浮点型张量，True变为1.0，False变为0.0
    result_tensor = bool_tensor.float()
    return result_tensor

# 写一个类，用于计算多标签分类评价指标
class MetricsWithMultiLabel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.best_epoch = 0
        self.best_acc = 0.
        self.best_b_acc = 0.
        self.best_recall = 0.
        self.best_precision = 0.
        self.best_specificity = 0.
        self.best_f1_score = 0.

        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.train_TP = np.zeros(self.num_classes)
        self.train_TN = np.zeros(self.num_classes)
        self.train_FP = np.zeros(self.num_classes)
        self.train_FN = np.zeros(self.num_classes)
        self.total_train_TP = np.zeros(self.num_classes)
        self.total_train_TN = np.zeros(self.num_classes)
        self.total_train_FP = np.zeros(self.num_classes)
        self.total_train_FN = np.zeros(self.num_classes)
        self.total_train_accuracy = 0.
        self.total_train_precision = 0.
        self.total_train_recall = 0.
        self.total_train_specificity = 0.
        self.total_train_f1_score = 0.
        self.total_train_auc = 0.
        self.total_train_balanced_accuracy = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.

        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.eval_TP = np.zeros(self.num_classes)
        self.eval_TN = np.zeros(self.num_classes)
        self.eval_FP = np.zeros(self.num_classes)
        self.eval_FN = np.zeros(self.num_classes)
        self.total_eval_TP = np.zeros(self.num_classes)
        self.total_eval_TN = np.zeros(self.num_classes)
        self.total_eval_FP = np.zeros(self.num_classes)
        self.total_eval_FN = np.zeros(self.num_classes)
        self.total_eval_accuracy = 0.
        self.total_eval_precision = 0.
        self.total_eval_recall = 0.
        self.total_eval_specificity = 0.
        self.total_eval_f1_score = 0.
        self.total_eval_auc = 0.
        self.total_eval_balanced_accuracy = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.

        self.eps = 0.00001

    def reset(self):
        self.total_train_loss = 0.
        self.train_TP = np.zeros(self.num_classes)
        self.train_TN = np.zeros(self.num_classes)
        self.train_FP = np.zeros(self.num_classes)
        self.train_FN = np.zeros(self.num_classes)
        self.total_train_TP = np.zeros(self.num_classes)
        self.total_train_TN = np.zeros(self.num_classes)
        self.total_train_FP = np.zeros(self.num_classes)
        self.total_train_FN = np.zeros(self.num_classes)
        self.total_train_accuracy = 0.
        self.total_train_precision = 0.
        self.total_train_recall = 0.
        self.total_train_specificity = 0.
        self.total_train_f1_score = 0.
        self.total_train_balanced_accuracy = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.

        self.total_eval_loss = 0.
        self.eval_TP = np.zeros(self.num_classes)
        self.eval_TN = np.zeros(self.num_classes)
        self.eval_FP = np.zeros(self.num_classes)
        self.eval_FN = np.zeros(self.num_classes)
        self.total_eval_TP = np.zeros(self.num_classes)
        self.total_eval_TN = np.zeros(self.num_classes)
        self.total_eval_FP = np.zeros(self.num_classes)
        self.total_eval_FN = np.zeros(self.num_classes)
        self.total_eval_accuracy = 0.
        self.total_eval_precision = 0.
        self.total_eval_recall = 0.
        self.total_eval_specificity = 0.
        self.total_eval_f1_score = 0.
        self.total_eval_balanced_accuracy = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.
    def train_update(self, loss, predictions, labels):
        self.total_train_loss += loss.item()
        predictions_sigmoid = torch.sigmoid(predictions)
        # print(predictions[:, 0], labels[:, 0])
        predictions_0_1 = to0_1(predictions_sigmoid)
        # _, val_predicted = torch.max(predictions.data, 1)
        self.train_TP[0], self.train_TN[0], self.train_FP[0], self.train_FN[0] = get_metrics(predictions_0_1[:, 0], labels[:, 1])
        self.train_TP[1], self.train_TN[1], self.train_FP[1], self.train_FN[1] = get_metrics(predictions_0_1[:, 1], labels[:, 2])
        self.train_TP[2], self.train_TN[2], self.train_FP[2], self.train_FN[2] = get_metrics(predictions_0_1[:, 2], labels[:, 3])
        self.train_TP[3], self.train_TN[3], self.train_FP[3], self.train_FN[3] = get_metrics(predictions_0_1[:, 3], labels[:, 4])
        self.train_TP[4], self.train_TN[4], self.train_FP[4], self.train_FN[4] = get_metrics(predictions_0_1[:, 4], labels[:, 5])
        self.train_TP[5], self.train_TN[5], self.train_FP[5], self.train_FN[5] = get_metrics(predictions_0_1[:, 5], labels[:, 6])
        self.train_TP[6], self.train_TN[6], self.train_FP[6], self.train_FN[6] = get_metrics(predictions_0_1[:, 6], labels[:, 7])
        self.train_TP[7], self.train_TN[7], self.train_FP[7], self.train_FN[7] = get_metrics(predictions_0_1[:], labels[:, 0], normal=True)

        self.total_train_TP += self.train_TP
        self.total_train_TN += self.train_TN
        self.total_train_FP += self.train_FP
        self.total_train_FN += self.train_FN

    def eval_update(self, loss, predictions, labels):
        self.total_eval_loss += loss.item()

        predictions_sigmoid = torch.sigmoid(predictions)
        # print(predictions[:, 0], labels[:, 0])
        predictions_0_1 = to0_1(predictions_sigmoid)
        # _, val_predicted = torch.max(predictions.data, 1)
        self.eval_TP[0], self.eval_TN[0], self.eval_FP[0], self.eval_FN[0] = get_metrics(predictions_0_1[:, 0], labels[:, 1])
        self.eval_TP[1], self.eval_TN[1], self.eval_FP[1], self.eval_FN[1] = get_metrics(predictions_0_1[:, 1], labels[:, 2])
        self.eval_TP[2], self.eval_TN[2], self.eval_FP[2], self.eval_FN[2] = get_metrics(predictions_0_1[:, 2], labels[:, 3])
        self.eval_TP[3], self.eval_TN[3], self.eval_FP[3], self.eval_FN[3] = get_metrics(predictions_0_1[:, 3], labels[:, 4])
        self.eval_TP[4], self.eval_TN[4], self.eval_FP[4], self.eval_FN[4] = get_metrics(predictions_0_1[:, 4], labels[:, 5])
        self.eval_TP[5], self.eval_TN[5], self.eval_FP[5], self.eval_FN[5] = get_metrics(predictions_0_1[:, 5], labels[:, 6])
        self.eval_TP[6], self.eval_TN[6], self.eval_FP[6], self.eval_FN[6] = get_metrics(predictions_0_1[:, 6], labels[:, 7])
        self.eval_TP[7], self.eval_TN[7], self.eval_FP[7], self.eval_FN[7] = get_metrics(predictions_0_1[:], labels[:, 0], normal=True)

        self.total_eval_TP += self.eval_TP
        self.total_eval_TN += self.eval_TN
        self.total_eval_FP += self.eval_FP
        self.total_eval_FN += self.eval_FN
    def compute_result(self):
        self.total_train_accuracy = (self.total_train_TP + self.total_train_TN) / (
                    self.total_train_TP + self.total_train_TN + self.total_train_FP + self.total_train_FN)
        self.total_train_recall = self.total_train_TP / (self.total_train_TP + self.total_train_FN + self.eps)
        self.total_train_precision = self.total_train_TP / (self.total_train_TP + self.total_train_FP + self.eps)
        self.total_train_specificity = self.total_train_TN / (self.total_train_TN + self.total_train_FP + self.eps)
        self.total_train_f1_score = 2 * self.total_train_precision * self.total_train_recall / (
                self.total_train_precision + self.total_train_recall + self.eps)
        self.total_train_balanced_accuracy = (self.total_train_recall + self.total_train_specificity) / 2
        self.average_train_accuracy = np.mean(self.total_train_accuracy)
        self.average_train_recall = np.mean(self.total_train_recall)
        self.average_train_precision = np.mean(self.total_train_precision)
        self.average_train_specificity = np.mean(self.total_train_specificity)
        self.average_train_f1_score = np.mean(self.total_train_f1_score)
        self.average_train_balanced_accuracy = np.mean(self.total_train_balanced_accuracy)

        self.total_eval_accuracy = (self.total_eval_TP + self.total_eval_TN) / (
                    self.total_eval_TP + self.total_eval_TN + self.total_eval_FP + self.total_eval_FN)
        self.total_eval_recall = self.total_eval_TP / (self.total_eval_TP + self.total_eval_FN + self.eps)
        self.total_eval_precision = self.total_eval_TP / (self.total_eval_TP + self.total_eval_FP + self.eps)
        self.total_eval_specificity = self.total_eval_TN / (self.total_eval_TN + self.total_eval_FP + self.eps)
        self.total_eval_f1_score = 2 * self.total_eval_precision * self.total_eval_recall / (
                self.total_eval_precision + self.total_eval_recall + self.eps)
        self.total_eval_balanced_accuracy = (self.total_eval_recall + self.total_eval_specificity) / 2
        self.average_eval_accuracy = np.mean(self.total_eval_accuracy)
        self.average_eval_recall = np.mean(self.total_eval_recall)
        self.average_eval_precision = np.mean(self.total_eval_precision)
        self.average_eval_specificity = np.mean(self.total_eval_specificity)
        self.average_eval_f1_score = np.mean(self.total_eval_f1_score)
        self.average_eval_balanced_accuracy = np.mean(self.total_eval_balanced_accuracy)

    def print_result(self, e, epochs):
        train_output_result = (f"Epoch [{e}/{epochs}]:, train_loss={self.average_train_loss:.3f}, \n"
                              f"train_accuracy={self.total_train_accuracy}, \n"
                              f"train_recall={self.total_train_recall}, \n"
                              f"train_precision={self.total_train_precision}, \n"
                              f"train_specificity={self.total_train_specificity}, \n"
                              f"train_balance_acc={self.total_train_balanced_accuracy},\n "
                              f"train_f1_score={self.total_train_f1_score},\n "
                              f"average_train_acc={self.average_train_accuracy:.3f}, "
                              f"average_train_recall={self.average_train_recall:.3f}, "
                              f"average_train_precision={self.average_train_precision:.3f}, "
                              f"average_train_specificity={self.average_train_specificity:.3f}\n"
                              f"average_train_balance_acc = {self.average_train_balanced_accuracy:.3f}, "
                              f"average_train_f1_score={self.average_train_f1_score:.3f} \n")

        eval_output_result = (f"Epoch [{e}/{epochs}]:, val_loss={self.average_eval_loss:.3f}, \n"
                              f"eval_accuracy={self.total_eval_accuracy}, \n"
                              f"eval_recall={self.total_eval_recall}, \n"
                              f"eval_precision={self.total_eval_precision}, \n"
                              f"eval_specificity={self.total_eval_specificity}, \n"
                              f"eval_balance_acc={self.total_eval_balanced_accuracy},\n "
                              f"eval_f1_score={self.total_eval_f1_score},\n "
                              f"average_eval_acc={self.average_eval_accuracy:.3f}, "
                              f"average_eval_recall={self.average_eval_recall:.3f}, "
                              f"average_eval_precision={self.average_eval_precision:.3f}, "
                              f"average_eval_specificity={self.average_eval_specificity:.3f}\n"
                              f"average_eval_balance_acc = {self.average_eval_balanced_accuracy:.3f}, "
                              f"average_eval_f1_score={self.average_eval_f1_score:.3f} \n")

        print(train_output_result)
        print(eval_output_result)
        # 将完整的字符串写入文件
        with open(f'logs.txt', 'a') as f:
            f.write(train_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁
            f.write(eval_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁

    def get_best(self, e):
        if self.average_eval_balanced_accuracy > self.best_b_acc:
            self.best_epoch = e
            self.best_acc = self.average_eval_accuracy
            self.best_b_acc = self.average_eval_balanced_accuracy
            self.best_recall = self.average_eval_recall
            self.best_precision = self.average_eval_precision
            self.best_specificity = self.average_eval_specificity
            self.best_f1_score = self.average_eval_f1_score
            return True
        return False

    def print_best_result(self, fold):
        best_result = (f"Fold {fold} Best Epoch: {self.best_epoch}, Best Val Acc: {self.best_acc:.3f}, "
                       f"Best Balance Acc = {self.best_b_acc:.3f}\n Best Recall: {self.best_recall:.3f}, "
                       f"Best Precision: {self.best_precision:.3f}, Best Specificity: {self.best_specificity:.3f} "
                       f"Best F1: {self.best_f1_score:.3f}")
        print(best_result)
        with open(f'logs.txt', 'a') as f:
            f.write(best_result + '\n')

# 写一个类，用于计算多标签分类评价指标， 使用torchmetrics
class MetricsWithMultiLabelWithTorchmetrics:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.best_dicts = {'epoch': 0,
                           'total_accuracy': None, 'total_precision': None, 'total_recall': None,
                           'total_specificity': None, 'total_balance_accuracy': None, 'total_f1_score': None,
                           'confusionMatrix': None,
                           'average_accuracy': 0., 'average_precision': 0., 'average_recall': 0.,
                           'average_specificity': 0., 'average_balance_accuracy':0., 'average_f1_score': 0.
                           }

        self.total_train_loss = 0.
        self.total_train_accuracy = 0.
        self.total_train_recall = 0.
        self.total_train_precision = 0.
        self.total_train_f1_score = 0.
        self.total_train_specificity = 0.
        self.total_train_balanced_accuracy = 0.
        self.average_train_loss = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.

        self.total_eval_loss = 0.
        self.total_eval_accuracy = 0.
        self.total_eval_recall = 0.
        self.total_eval_precision = 0.
        self.total_eval_f1_score = 0.
        self.total_eval_specificity = 0.
        self.total_eval_balanced_accuracy = 0.
        self.average_eval_loss = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.

        # train stage
        self.train_acc = MultilabelAccuracy(num_labels=num_classes, average=None).to(device)
        self.train_recall = MultilabelRecall(num_labels=num_classes, average=None).to(device)
        self.train_precision = MultilabelPrecision(num_labels=num_classes, average=None).to(device)
        self.train_F1 = MultilabelF1Score(num_labels=num_classes, average=None).to(device)
        self.train_spe = MultilabelSpecificity(num_labels=num_classes, average=None).to(device)
        self.train_weight_acc = MultilabelAccuracy(num_labels=num_classes, average='weighted').to(device)
        self.train_weight_recall = MultilabelRecall(num_labels=num_classes, average='weighted').to(device)
        self.train_weight_precision = MultilabelPrecision(num_labels=num_classes, average='weighted').to(device)
        self.train_weight_F1 = MultilabelF1Score(num_labels=num_classes, average='weighted').to(device)
        self.train_weight_spe = MultilabelSpecificity(num_labels=num_classes, average='weighted').to(device)
        self.train_confusionMatrix = MultilabelConfusionMatrix(num_labels=num_classes).to(device)
        self.train_total_confusionMatrix = None
        # eval stage
        self.eval_acc = MultilabelAccuracy(num_labels=num_classes, average=None).to(device)
        self.eval_recall = MultilabelRecall(num_labels=num_classes, average=None).to(device)
        self.eval_precision = MultilabelPrecision(num_labels=num_classes, average=None).to(device)
        self.eval_F1 = MultilabelF1Score(num_labels=num_classes, average=None).to(device)
        self.eval_spe = MultilabelSpecificity(num_labels=num_classes, average=None).to(device)
        self.eval_weight_acc = MultilabelAccuracy(num_labels=num_classes, average='weighted').to(device)
        self.eval_weight_recall = MultilabelRecall(num_labels=num_classes, average='weighted').to(device)
        self.eval_weight_precision = MultilabelPrecision(num_labels=num_classes, average='weighted').to(device)
        self.eval_weight_F1 = MultilabelF1Score(num_labels=num_classes, average='weighted').to(device)
        self.eval_weight_spe = MultilabelSpecificity(num_labels=num_classes, average='weighted').to(device)
        self.eval_confusionMatrix = MultilabelConfusionMatrix(num_labels=num_classes).to(device)
        self.eval_total_confusionMatrix = None
    def reset(self):
        self.total_train_loss = 0.
        self.total_train_accuracy = 0.
        self.total_train_recall = 0.
        self.total_train_precision = 0.
        self.total_train_f1_score = 0.
        self.total_train_specificity = 0.
        self.total_train_balanced_accuracy = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.
        self.train_acc.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_F1.reset()
        self.train_spe.reset()
        self.train_weight_acc.reset()
        self.train_weight_recall.reset()
        self.train_weight_precision.reset()
        self.train_weight_F1.reset()
        self.train_weight_spe.reset()
        self.train_confusionMatrix.reset()
        self.train_total_confusionMatrix = None

        self.total_eval_loss = 0.
        self.total_eval_accuracy = 0.
        self.total_eval_recall = 0.
        self.total_eval_precision = 0.
        self.total_eval_f1_score = 0.
        self.total_eval_specificity = 0.
        self.total_eval_balanced_accuracy = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.
        self.eval_acc.reset()
        self.eval_recall.reset()
        self.eval_precision.reset()
        self.eval_F1.reset()
        self.eval_spe.reset()
        self.eval_weight_acc.reset()
        self.eval_weight_recall.reset()
        self.eval_weight_precision.reset()
        self.eval_weight_F1.reset()
        self.eval_weight_spe.reset()
        self.eval_confusionMatrix.reset()
        self.eval_total_confusionMatrix = None
    def train_update(self, loss, predictions, labels):
        self.total_train_loss += loss.item()
        predictions_sigmoid = torch.sigmoid(predictions)
        self.train_acc.update(predictions_sigmoid, labels)
        self.train_recall.update(predictions_sigmoid, labels)
        self.train_precision.update(predictions_sigmoid, labels)
        self.train_F1.update(predictions_sigmoid, labels)
        self.train_spe.update(predictions_sigmoid, labels)
        self.train_weight_acc.update(predictions_sigmoid, labels)
        self.train_weight_recall.update(predictions_sigmoid, labels)
        self.train_weight_precision.update(predictions_sigmoid, labels)
        self.train_weight_F1.update(predictions_sigmoid, labels)
        self.train_weight_spe.update(predictions_sigmoid, labels)
        self.train_confusionMatrix.update(predictions_sigmoid, labels)
        # self.train_total_confusionMatrix.update(predictions_sigmoid, labels)
    def eval_update(self, loss, predictions, labels):
        self.total_eval_loss += loss.item()
        predictions_sigmoid = torch.sigmoid(predictions)
        self.eval_acc.update(predictions_sigmoid, labels)
        self.eval_recall.update(predictions_sigmoid, labels)
        self.eval_precision.update(predictions_sigmoid, labels)
        self.eval_F1.update(predictions_sigmoid, labels)
        self.eval_spe.update(predictions_sigmoid, labels)
        self.eval_weight_acc.update(predictions_sigmoid, labels)
        self.eval_weight_recall.update(predictions_sigmoid, labels)
        self.eval_weight_precision.update(predictions_sigmoid, labels)
        self.eval_weight_F1.update(predictions_sigmoid, labels)
        self.eval_weight_spe.update(predictions_sigmoid, labels)
        self.eval_confusionMatrix.update(predictions_sigmoid, labels)
        # self.train_total_confusionMatrix.update(predictions_sigmoid, labels)
    def compute_result(self):
        self.total_train_accuracy = self.train_acc.compute()
        self.total_train_recall = self.train_recall.compute()
        self.total_train_precision = self.train_precision.compute()
        self.total_train_f1_score = self.train_F1.compute()
        self.total_train_specificity = self.train_spe.compute()
        self.total_train_balanced_accuracy = (self.total_train_recall + self.total_train_specificity) / 2.0
        self.average_train_accuracy = self.train_weight_acc.compute()
        self.average_train_recall = self.train_weight_recall.compute()
        self.average_train_precision = self.train_weight_precision.compute()
        self.average_train_f1_score = self.train_weight_F1.compute()
        self.average_train_specificity = self.train_weight_spe.compute()
        self.average_train_balanced_accuracy = (self.average_train_recall + self.average_train_specificity) / 2.0
        self.train_total_confusionMatrix = self.train_confusionMatrix.compute()

        self.total_eval_accuracy = self.eval_acc.compute()
        self.total_eval_recall = self.eval_recall.compute()
        self.total_eval_precision = self.eval_precision.compute()
        self.total_eval_f1_score = self.eval_F1.compute()
        self.total_eval_specificity = self.eval_spe.compute()
        self.total_eval_balanced_accuracy = (self.total_eval_recall + self.total_eval_specificity) / 2.0
        self.average_eval_accuracy = self.eval_weight_acc.compute()
        self.average_eval_recall = self.eval_weight_recall.compute()
        self.average_eval_precision = self.eval_weight_precision.compute()
        self.average_eval_f1_score = self.eval_weight_F1.compute()
        self.average_eval_specificity = self.eval_weight_spe.compute()
        self.average_eval_balanced_accuracy = (self.average_eval_recall + self.average_eval_specificity) / 2.0
        self.eval_total_confusionMatrix = self.eval_confusionMatrix.compute()
    def print_result(self, e, epochs):
        train_output_result = (f"Epoch [{e}/{epochs}]:, train_loss={self.average_train_loss:.3f}, \n"
                              f"train_confusionMatrix:\n{self.train_total_confusionMatrix}\n"
                              f"train_accuracy={self.total_train_accuracy}, \n"
                              f"train_recall={self.total_train_recall}, \n"
                              f"train_precision={self.total_train_precision}, \n"
                              f"train_specificity={self.total_train_specificity}, \n"
                              f"train_balance_acc={self.total_train_balanced_accuracy},\n "
                              f"train_f1_score={self.total_train_f1_score},\n "
                              f"average_train_acc={self.average_train_accuracy:.3f}, "
                              f"average_train_recall={self.average_train_recall:.3f}, "
                              f"average_train_precision={self.average_train_precision:.3f}, "
                              f"average_train_specificity={self.average_train_specificity:.3f}\n"
                              f"average_train_balance_acc = {self.average_train_balanced_accuracy:.3f}, "
                              f"average_train_f1_score={self.average_train_f1_score:.3f} \n")
        eval_output_result = (f"Epoch [{e}/{epochs}]:, val_loss={self.average_eval_loss:.3f}, \n"
                              f"eval_confusionMatrix:{self.eval_total_confusionMatrix}\n"
                              f"eval_accuracy={self.total_eval_accuracy}, \n"
                              f"eval_recall={self.total_eval_recall}, \n"
                              f"eval_precision={self.total_eval_precision}, \n"
                              f"eval_specificity={self.total_eval_specificity}, \n"
                              f"eval_balance_acc={self.total_eval_balanced_accuracy},\n "
                              f"eval_f1_score={self.total_eval_f1_score},\n "
                              f"average_eval_acc={self.average_eval_accuracy:.3f}, "
                              f"average_eval_recall={self.average_eval_recall:.3f}, "
                              f"average_eval_precision={self.average_eval_precision:.3f}, "
                              f"average_eval_specificity={self.average_eval_specificity:.3f}\n"
                              f"average_eval_balance_acc = {self.average_eval_balanced_accuracy:.3f}, "
                              f"average_eval_f1_score={self.average_eval_f1_score:.3f} \n")
        print(train_output_result)
        print(eval_output_result)
        # 将完整的字符串写入文件
        with open(f'logs.txt', 'a') as f:
            f.write(train_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁
            f.write(eval_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁
    def get_best(self, e):
        if self.average_eval_balanced_accuracy > self.best_dicts['average_balance_accuracy']:
            self.best_dicts['epoch'] = e
            self.best_dicts['confusionMatrix'] = self.eval_total_confusionMatrix
            self.best_dicts['total_accuracy'] = self.total_eval_accuracy
            self.best_dicts['total_precision'] = self.total_eval_precision
            self.best_dicts['total_recall'] = self.total_eval_recall
            self.best_dicts['total_specificity'] = self.total_eval_specificity
            self.best_dicts['total_balance_accuracy'] = self.total_eval_balanced_accuracy
            self.best_dicts['total_f1_score'] = self.total_eval_f1_score
            self.best_dicts['average_accuracy'] = self.average_eval_accuracy
            self.best_dicts['average_balance_accuracy'] = self.average_eval_balanced_accuracy
            self.best_dicts['average_recall'] = self.average_eval_recall
            self.best_dicts['average_precision'] = self.average_eval_precision
            self.best_dicts['average_specificity'] = self.average_eval_specificity
            self.best_dicts['average_f1_score'] = self.average_eval_f1_score
            return True
        return False
    def print_best_result(self, fold):
        best_result = (f"Fold {fold} Best Epoch: {self.best_dicts['epoch']}\n"
                              f"Best Total confusionMatrix : {self.best_dicts['confusionMatrix']}\n"
                              f"Best Total accuracy : {self.best_dicts['total_accuracy']}, \n"
                              f"Best Total recall : {self.best_dicts['total_recall']}, \n"
                              f"Best Total precision : {self.best_dicts['total_precision']}, \n"
                              f"Best Total specificity : {self.best_dicts['total_specificity']}, \n"
                              f"Best Total balance_acc : {self.best_dicts['total_balance_accuracy']},\n "
                              f"Best Total f1_score : {self.best_dicts['total_f1_score']},\n "
                              f"Best Average accuracy : {self.best_dicts['average_accuracy']:.3f}, "
                              f"Best Average recall : {self.best_dicts['average_recall']:.3f}, "
                              f"Best Average precision : {self.best_dicts['average_precision']:.3f}, "
                              f"Best Average specificity : {self.best_dicts['average_specificity']:.3f}\n"
                              f"Best Average balance_acc :  {self.best_dicts['average_balance_accuracy']:.3f}\n"
                              f"Best Average f1_score : {self.best_dicts['average_f1_score']:.3f} \n")
        print(best_result)
        with open(f'logs.txt', 'a') as f:
            f.write(best_result + '\n')

# 写一个类，用于评价单标签分类评价指标
class MetricsWithSingleLabel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.best_dicts = {'epoch': 0, 'acc': 0., 'pre': 0., 'recall': 0., 'spe': 0., 'bacc':0., 'f1': 0.}

        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.

        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.

        self.train_acc = torchmetrics.Accuracy(num_classes=num_classes, task='binary').to(device)
        self.train_recall = torchmetrics.Recall(num_classes=num_classes, task='binary').to(device)
        self.train_precision = torchmetrics.Precision(num_classes=num_classes, task='binary').to(device)
        # self.train_auc = torchmetrics.AUROC(num_classes=num_classes, task='binary').to(device)
        self.train_F1 = torchmetrics.F1Score(num_classes=num_classes, task='binary').to(device)
        self.train_spe = torchmetrics.Specificity(num_classes=num_classes, task='binary').to(device)
        self.train_confusionMatrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='binary').to(device)
        self.train_total_confusionMatrix = None

        self.eval_acc = torchmetrics.Accuracy(num_classes=num_classes, task='binary').to(device)
        self.eval_recall = torchmetrics.Recall(num_classes=num_classes, task='binary').to(device)
        self.eval_precision = torchmetrics.Precision(num_classes=num_classes, task='binary').to(device)
        # self.eval_auc = torchmetrics.AUROC(num_classes=num_classes, task='binary').to(device)
        self.eval_F1 = torchmetrics.F1Score(num_classes=num_classes, task='binary').to(device)
        self.eval_spe = torchmetrics.Specificity(num_classes=num_classes, task='binary').to(device)
        self.eval_confusionMatrix = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='binary').to(device)
        self.eval_total_confusionMatrix = None

    def reset(self):
        self.total_train_loss = 0.
        self.average_train_loss = 0.
        self.average_train_accuracy = 0.
        self.average_train_recall = 0.
        self.average_train_precision = 0.
        self.average_train_specificity = 0.
        self.average_train_f1_score = 0.
        self.average_train_balanced_accuracy = 0.
        self.train_total_confusionMatrix = None

        self.total_eval_loss = 0.
        self.average_eval_loss = 0.
        self.average_eval_accuracy = 0.
        self.average_eval_recall = 0.
        self.average_eval_precision = 0.
        self.average_eval_specificity = 0.
        self.average_eval_f1_score = 0.
        self.average_eval_balanced_accuracy = 0.
        self.eval_total_confusionMatrix = None

        self.train_acc.reset()
        # self.train_auc.reset()
        self.train_recall.reset()
        self.train_precision.reset()
        self.train_F1.reset()
        self.train_spe.reset()
        self.train_confusionMatrix.reset()

        self.eval_acc.reset()
        # self.eval_auc.reset()
        self.eval_recall.reset()
        self.eval_precision.reset()
        self.eval_F1.reset()
        self.eval_spe.reset()
        self.eval_confusionMatrix.reset()

    def train_update(self, loss, predictions, labels):
        self.total_train_loss += loss.item()
        self.train_acc.update(predictions, labels)
        # self.train_auc.update(prob, labels)
        self.train_recall.update(predictions, labels)
        self.train_precision.update(predictions, labels)
        self.train_F1.update(predictions, labels)
        self.train_spe.update(predictions, labels)
        self.train_confusionMatrix.update(predictions, labels)
        # self.train_total_confusionMatrix += self.train_confusionMatrix(predictions, labels)
        # print("self.train_confusionMatrix", self.train_confusionMatrix)

    def eval_update(self, loss, predictions, labels):
        # predictions_sigmoid = torch.sigmoid(predictions)
        # # print(predictions[:, 0], labels[:, 0])
        # predictions_0_1 = to0_1(predictions_sigmoid)
        # # _, val_predicted = torch.max(predictions.data, 1)

        self.total_eval_loss += loss.item()
        self.eval_acc.update(predictions, labels)
        # self.eval_auc.update(prob, labels)
        self.eval_recall.update(predictions, labels)
        self.eval_precision.update(predictions, labels)
        self.eval_F1.update(predictions, labels)
        self.eval_spe.update(predictions, labels)
        # self.eval_confusionMatrix(predictions, labels)
        self.eval_confusionMatrix.update(predictions, labels)
        # self.eval_total_confusionMatrix += self.eval_confusionMatrix(predictions, labels)
        # print("self.eval_confusionMatrix", self.eval_confusionMatrix)

    def compute_result(self):
        # self.total_eval_accuracy = self.eval_acc.compute()
        # self.total_eval_recall = self.eval_recall.compute()
        # self.total_eval_precision = self.eval_precision.compute()
        # # total_auc = self.eval_auc.compute()
        # self.total_eval_f1_score = self.eval_F1.compute()
        # self.total_eval_specificity = self.eval_spe.compute()
        self.average_train_accuracy = self.train_acc.compute()
        self.average_train_recall = self.train_recall.compute()
        self.average_train_precision = self.train_precision.compute()
        # total_auc = self.train_auc.compute()
        self.average_train_f1_score = self.train_F1.compute()
        self.average_train_specificity = self.train_spe.compute()
        self.average_train_balanced_accuracy = (self.average_train_recall + self.average_train_specificity) / 2.0
        self.train_total_confusionMatrix = self.train_confusionMatrix.compute()

        self.average_eval_accuracy = self.eval_acc.compute()
        self.average_eval_recall = self.eval_recall.compute()
        self.average_eval_precision = self.eval_precision.compute()
        # total_auc = self.eval_auc.compute()
        self.average_eval_f1_score = self.eval_F1.compute()
        self.average_eval_specificity = self.eval_spe.compute()
        self.average_eval_balanced_accuracy = (self.average_eval_recall + self.average_eval_specificity) / 2.0
        self.eval_total_confusionMatrix = self.eval_confusionMatrix.compute()

    def print_result(self, e, epochs):
        train_output_result = (f"Epoch [{e}/{epochs}]:, train_loss={self.average_train_loss:.3f}, \n"
                               f"train_confusionMatrix:{self.train_total_confusionMatrix}\n"
                               f"average_eval_acc={self.average_train_accuracy:.3f}, "
                               f"average_eval_recall={self.average_train_recall:.3f}, "
                               f"average_train_precision={self.average_train_precision:.3f}, "
                               f"average_train_specificity={self.average_train_specificity:.3f}\n"
                               f"average_train_balance_acc = {self.average_train_balanced_accuracy:.3f}, "
                               f"average_train_f1_score={self.average_train_f1_score:.3f} \n")

        eval_output_result = (f"Epoch [{e}/{epochs}]:, val_loss={self.average_eval_loss:.3f}, \n"
                              f"eval_confusionMatrix:{self.eval_total_confusionMatrix}\n"
                              f"average_eval_acc={self.average_eval_accuracy:.3f}, "
                              f"average_eval_recall={self.average_eval_recall:.3f}, "
                              f"average_eval_precision={self.average_eval_precision:.3f}, "
                              f"average_eval_specificity={self.average_eval_specificity:.3f}\n"
                              f"average_eval_balance_acc = {self.average_eval_balanced_accuracy:.3f}, "
                              f"average_eval_f1_score={self.average_eval_f1_score:.3f} \n")

        print(train_output_result)
        print(eval_output_result)
        # 将完整的字符串写入文件
        with open(f'logs.txt', 'a') as f:
            f.write(train_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁
            f.write(eval_output_result + '\n')  # 在最后添加一个换行符以保持格式整洁

    def get_best(self, e):
        if self.average_eval_balanced_accuracy > self.best_dicts['bacc']:
            self.best_dicts['epoch'] = e
            self.best_dicts['acc'] = self.average_eval_accuracy
            self.best_dicts['bacc'] = self.average_eval_balanced_accuracy
            self.best_dicts['recall'] = self.average_eval_recall
            self.best_dicts['pre'] = self.average_eval_precision
            self.best_dicts['spe'] = self.average_eval_specificity
            self.best_dicts['f1'] = self.average_eval_f1_score
            return True
        return False

    def print_best_result(self, fold):
        best_result = (f"Fold {fold} Best Epoch: {self.best_dicts['epoch']}, Best Val Acc: {self.best_dicts['acc']:.3f}, "
                       f"Best Balance Acc = {self.best_dicts['bacc']:.3f}\n Best Recall: {self.best_dicts['recall']:.3f}, "
                       f"Best Precision: {self.best_dicts['pre']:.3f}, Best Specificity: {self.best_dicts['spe']:.3f} "
                       f"Best F1: {self.best_dicts['f1']:.3f}")
        print(best_result)
        with open(f'logs.txt', 'a') as f:
            f.write(best_result + '\n')
