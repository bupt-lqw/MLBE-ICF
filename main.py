import  torch
import  argparse
import random
import  scipy.stats
import numpy as np
import pandas as pd

from GetDataset import GetTrainDataset, GetTestDataset, GetTestTrainDataset
from torch.utils.data import DataLoader
from meta import Meta
from util import get_performances
from sklearn.model_selection import KFold


def save_train_test(df, save_path, name, random_state):
    '''
    获取每次交叉验证的训练集和测试集
    '''
    
    # 获取多数类样本和少数类样本的df
    df0 = df[df[df.columns[-1]] == 0]
    df1 = df[df[df.columns[-1]] == 1]
    
    # 初始化KFold以及结果列表
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # 初始化训练集和测试集列表
    # train_test_list = []
    i = 1
    # 获取每一次交叉验证中的训练集和测试集
    for index0, index1 in zip(kf.split(df0), kf.split(df1)):

        # 获取多数类样本和少数类样本的训练集和测试集index的列表，数据类型为numpy.ndarray
        train_index0, test_index0 = index0
        train_index1, test_index1 = index1

        # 根据上述index列表，获取一次交叉验证中所有的训练集样本和测试集样本，包括多数类样本和少数类样本
        train_df = pd.concat([df0.iloc[train_index0], df1.iloc[train_index1]], axis=0)
        test_df = pd.concat([df0.iloc[test_index0], df1.iloc[test_index1]], axis=0)
        train_df.to_csv(save_path + name + '_' + str(i) + '_train' + '.csv', index=False)
        test_df.to_csv(save_path + name + '_' + str(i) + '_test' + '.csv', index=False)

        i = i + 1
        # train_test_list.append([train_df, test_df])
    
    # return train_test_list


def main(name, j):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # print(args)
    dataset_name = name + str('_') + str(j)

    train = GetTrainDataset('/home/lqw/temp/MAML-Pytorch-master/datasets2/' + dataset_name + '_train.csv', 
                       n_way=args.n_way, k_shot=args.k_spt, k_query=args.k_qry,
                        batchsz=20000)
    
    x_dim = train.x_dim - 1
    config = [
        ('linear', [128, x_dim]),
        ('relu', [True]),

        ('linear', [256, 128]),
        ('relu', [True]),

        ('linear', [args.n_way, 256])
    ]

    device = torch.device('cuda:2')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    # print('Total trainable tensors:', num)

    
    f1_list, gmean_list = [], []
    # for epoch in range(args.epoch):
    db = DataLoader(train, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
    
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)

        # if (step + 1) % 30 == 0:
        #     # print('step:', step, '\ttraining acc:', accs)
        #     print('step:', step + 1, end='\t')


        if (step + 1) % 250 == 0:  # evaluation
            test = GetTestDataset('/home/lqw/temp/MAML-Pytorch-master/datasets2/' + dataset_name + '_test.csv')
            x_qry, y_qry = test.qry_x.to(device), test.qry_y.to(device)
            test_train = GetTestTrainDataset('/home/lqw/temp/MAML-Pytorch-master/datasets2/' + dataset_name + '_train.csv')
            x_spt, y_spt = test_train.qry_x.to(device), test_train.qry_y.to(device)
            pred_qs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            f1, gmean = get_performances(y_qry, pred_qs)
            
            print(f1, gmean)
            f1_list.append(f1)
            gmean_list.append(gmean)
            if f1 >= 0.99999 and gmean >= 0.99999:
                break

    return f1, gmean

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=30)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    
    f1s, gmeans = [], []
    # for i in range(10):  # 使用10次五折交叉验证求平均的方式
    name = 'glass4'

    path = '/home/lqw/temp/MAML-Pytorch-master/keel_datasets/'
    save_path = '/home/lqw/temp/MAML-Pytorch-master/datasets2/'

    df = pd.read_csv(path + name + '.csv')
    save_train_test(df, save_path, name, random_state=2024)

    for j in range(5):
        print('第 ' + str(j + 1) + ' 次交叉验证')
        f1, gmean = main(name, j + 1)
        f1s.append(f1)
        gmeans.append(gmean)
    
    print(f1s)
    print(gmeans)

    f1 = sum(f1s) / len(f1s)
    gmean = sum(gmeans) / len(gmeans)
    print('*' * 10)
    print('*' * 10)
    print('*' * 10)

    print(name + '    ' + 'f1 : ' + str(f1) + '    ' + 'gmean : ' + str(gmean))

