import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
import torch
import smote_variants as sv


class GetTrainDataset(Dataset):
    
    def __init__(self, root, batchsz, n_way, k_shot, k_query):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set

        self.df = self.load_dataset(root)
        # oversampler = sv.SMOTE(n_nighbors=5)  # 初始化smote
        # X_samp, y_samp = oversampler.sample(np.array(self.df.iloc[:, :-1]), np.array(self.df.iloc[:, -1]))  # 对训练集过采样
        # self.df = pd.DataFrame(np.hstack((X_samp, y_samp[:, np.newaxis])), columns=self.df.columns)  # 拼接过采样后的训练集
        
        self.x_dim = self.df.shape[1]
        self.data0 = np.array(self.df.loc[self.df[self.df.columns[-1]] == 0])
        self.data1 = np.array(self.df.loc[self.df[self.df.columns[-1]] == 1])
        self.data = [self.data0.shape[0], self.data1.shape[0]]

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for _ in range(batchsz):  # for each batch
            support_x = []
            query_x = []
            for cls in [0, 1]:
                selected_data_idx = np.random.choice(self.data[cls], self.k_shot + self.k_query, False)
                indexDtrain = np.array(selected_data_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_data_idx[self.k_shot:])  # idx for Dtest
                support_x.append(indexDtrain.tolist())
                query_x.append(indexDtest.tolist())

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        """
        support_x_y = torch.FloatTensor(np.random.permutation(np.vstack(
                (self.data0[self.support_x_batch[index][0]], 
                 self.data1[self.support_x_batch[index][1]]))))
        query_x_y = torch.FloatTensor(np.random.permutation(np.vstack(
                (self.data0[self.query_x_batch[index][0]], 
                 self.data1[self.query_x_batch[index][1]]))))
        
        return support_x_y[:, :-1], support_x_y[:, -1].long(), query_x_y[:, :-1], query_x_y[:, -1].long()

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

    def load_dataset(self, root):
        '''
        加载数据集并对其进行预处理，包括将字符串转为数字、特征提取、将样本属性值归一化到(0, 1)范围内
        '''

        df = pd.read_csv(root)

        # 字符串转为数字，忽略错误（默认返回dtype为float64或int64，具体取决于提供的数据）
        df = df.apply(pd.to_numeric, errors='ignore')

        # 进行特征提取，将非数字属性值转换为one-hot编码格式（特征提取时需要先转换为dict，再转换为pd.DataFrame）
        vec = DictVectorizer(sparse=False)
        df.iloc[:, :-1] = pd.DataFrame(data=vec.fit_transform(df.iloc[:, :-1].to_dict(orient='records')), columns=vec.feature_names_)
        
        # 将样本属性值归一化到(0, 1)范围内  出问题
        # 先判断某一列数据是否相同，如果相同，则将其赋值为0，不相同则进行归一化到(-1, 1)范围内
        # 按照上面的思路，可以先归一化，再将全为NaN的列赋值为0
        df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())
        df = df.fillna(0)
        df = df.replace(np.inf, 0)

        return df



class GetTestDataset(Dataset):
    
    def __init__(self, qry_root):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param startidx: start to index label from startidx
        """

        self.qry_df = self.load_dataset(qry_root)
        self.qry_x_y = torch.FloatTensor(np.random.permutation(np.array(self.qry_df)))
        self.qry_x = self.qry_x_y[:, :-1]
        self.qry_y = self.qry_x_y[:, -1].long()

    def load_dataset(self, root):
        '''
        加载数据集并对其进行预处理，包括将字符串转为数字、特征提取、将样本属性值归一化到(0, 1)范围内
        '''

        df = pd.read_csv(root)

        # 字符串转为数字，忽略错误（默认返回dtype为float64或int64，具体取决于提供的数据）
        df = df.apply(pd.to_numeric, errors='ignore')

        # 进行特征提取，将非数字属性值转换为one-hot编码格式（特征提取时需要先转换为dict，再转换为pd.DataFrame）
        vec = DictVectorizer(sparse=False)
        df.iloc[:, :-1] = pd.DataFrame(data=vec.fit_transform(df.iloc[:, :-1].to_dict(orient='records')), columns=vec.feature_names_)
        
        # 将样本属性值归一化到(0, 1)范围内  出问题
        # 先判断某一列数据是否相同，如果相同，则将其赋值为0，不相同则进行归一化到(-1, 1)范围内
        # 按照上面的思路，可以先归一化，再将全为NaN的列赋值为0
        df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())
        df = df.fillna(0)
        df = df.replace(np.inf, 0)

        return df


class GetTestTrainDataset(Dataset):
    
    def __init__(self, qry_root):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param startidx: start to index label from startidx
        """

        self.df = self.load_dataset(qry_root)
        oversampler = sv.Borderline_SMOTE2(n_nighbors=5)  # 初始化smote
        X_samp, y_samp = oversampler.sample(np.array(self.df.iloc[:, :-1]), np.array(self.df.iloc[:, -1]))  # 对训练集过采样
        self.qry_df = pd.DataFrame(np.hstack((X_samp, y_samp[:, np.newaxis])), columns=self.df.columns)  # 拼接过采样后的训练集
        
        self.qry_x_y = torch.FloatTensor(np.random.permutation(np.array(self.qry_df)))
        self.qry_x = self.qry_x_y[:, :-1]
        self.qry_y = self.qry_x_y[:, -1].long()

    def load_dataset(self, root):
        '''
        加载数据集并对其进行预处理，包括将字符串转为数字、特征提取、将样本属性值归一化到(0, 1)范围内
        '''

        df = pd.read_csv(root)

        # 字符串转为数字，忽略错误（默认返回dtype为float64或int64，具体取决于提供的数据）
        df = df.apply(pd.to_numeric, errors='ignore')

        # 进行特征提取，将非数字属性值转换为one-hot编码格式（特征提取时需要先转换为dict，再转换为pd.DataFrame）
        vec = DictVectorizer(sparse=False)
        df.iloc[:, :-1] = pd.DataFrame(data=vec.fit_transform(df.iloc[:, :-1].to_dict(orient='records')), columns=vec.feature_names_)
        
        # 将样本属性值归一化到(0, 1)范围内  出问题
        # 先判断某一列数据是否相同，如果相同，则将其赋值为0，不相同则进行归一化到(-1, 1)范围内
        # 按照上面的思路，可以先归一化，再将全为NaN的列赋值为0
        df.iloc[:, :-1] = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / (df.iloc[:, :-1].max() - df.iloc[:, :-1].min())
        df = df.fillna(0)
        df = df.replace(np.inf, 0)

        return df



