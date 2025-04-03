import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Cnn For Any Dataset')
    parser.add_argument('--data_dir', type=str, default='/data4/wangchangmiao/shenxy/EyeDataset/Enhanced', help='Path to dataset')
    parser.add_argument('--csv_file_path', type=str, default='./csv/select.csv', help='Path to csv file')
    parser.add_argument('--data_choice', type=str, default='1', help='Dataset to choose, '
                                                                     '1 for MNIST, 2 for CIFAR10, 3 for Custom Dataset')
    parser.add_argument('--Image_size', type=int, default=512, help='Size to reshape image')
    parser.add_argument('--checkpoint_dir', type=str, default='./result', help='Path to save model')
    parser.add_argument('--pretrainedModelPath', type=str, default='/data4/wangchangmiao/shenxy/PretrainedModel/fundus_pretrained_VT_small_patch16_384_5Class.pth.tar', help='Model to use')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of Epoch')
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    parser.add_argument('--lr_policy', type=str, default='cosine', help='Scheduler to use')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='initial lambda decay value')
    parser.add_argument('--niter', type=int, default=50, help='lr decay step')
    parser.add_argument('--model', type=str, choices=['1', '2', '3', '4', '5', '6', '7'],
                        default='1', help='1 for DeFineNet, 2 for DeFineLeNet5, 3 for alexnet,'
                                          ' 4 for googlenet, 5 for vgg16, 6 for resnet18, 7 for mobilenet2')
    parser.add_argument('--loss', type=str, default='1', help='Loss function: CrossEntropy')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--activation', type=str, choices=['1', '2', '3'], default='2',
                        help='Activation function to use, 1 for Sigmoid, 2 for Relu, 3 for Tanh')
    parser.add_argument('--mode', type=str, choices=['1', '2'], default='1',
                        help='1 for common mode, 2 for k_fold mode')
    parser.add_argument('--k_split_value', type=int, default=5, help='k split value for k_fold mode')
    args = parser.parse_args()
    return args

# print(parse_args())
