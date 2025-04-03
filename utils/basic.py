import numpy as np
import torch
from torch import optim
from torch.nn import init
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
# 定义scheduler
def get_scheduler(optimizer, opt):
    """
    scheduler definition
    :param optimizer:  original optimizer
    :param opt: corresponding parameters
    :return: corresponding scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=0.00001)
    elif opt.lr_policy == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# 初始化权重
def init_weights(net, init_type='normal', gain=0.02):
    """
    initialize the network weights
    :param net: the network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :return: the initialized network
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=None):
    """
    initial the network
    :param net:  to be initialized network
    :param init_type:  initialized method
    :param gain: corresponding gain
    :param gpu_ids: the gpu ids
    :return: the initialized network
    """
    # if gpu_ids is None:
    #     gpu_ids = [-1, ]
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


# 评估指标
def estimate(y_true, y_logit, y_prob):
    train_cm = confusion_matrix(y_true, y_logit)
    tp = train_cm[1, 1]  # 正类被正确预测为正类的数量
    fp = train_cm[0, 1]  # 负类被错误预测为正类的数量
    fn = train_cm[1, 0]  # 正类被错误预测为负类的数量
    tn = train_cm[0, 0]  # 负类被正确预测为负类的数量（在二分类中通常不单独使用）

    acc = accuracy_score(y_true, y_logit)
    recall = recall_score(y_true, y_logit, average='binary')  # 对于二分类问题
    precision = precision_score(y_true, y_logit, average='binary')  # 对于二分类问题
    f1_scores = f1_score(y_true, y_logit, average='binary')  # 对于二分类问题
    # 计算Specificity（真负率）
    specificity = tn / (tn + fp)
    # Sensitivity其实就是recall，但为了完整性，我们也可以在这里返回它
    sensitivity = recall
    # 计算AUC（需要y_true和正类的预测概率）
    auc = roc_auc_score(y_true, y_prob)
    # 计算MCC
    mcc = matthews_corrcoef(y_true, y_logit)
    return tp, fp, fn, tn, acc, recall, precision, f1_scores, specificity, sensitivity, auc, mcc

def make_weights_for_balanced_classes(dataset):
    """
    在单次遍历中计算每个样本的权重以平衡不同类别的影响。

    :param dataset: 数据集对象，必须支持迭代，并且每个元素应包含标签信息。
    :return: list, 每个样本对应的权重列表。
    """
    class_counts = {}
    weights = []
    total_samples = len(dataset)

    # 单次遍历完成类别计数和权重计算
    for _, _, label_1d, _ in dataset:
        if label_1d not in class_counts:
            class_counts[label_1d] = 0
        class_counts[label_1d] += 1
        # total_samples += 1
    for _, _, label_1d, _ in dataset:
        class_weight = total_samples / float(class_counts[label_1d])
        weights.append(class_weight)

    return weights
