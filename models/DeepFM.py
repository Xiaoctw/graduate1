import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.data as Data
import torch.nn.functional as fun
import matplotlib.pyplot as plt

'''
DeepFM模型部分
'''


class DeepFM(nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size=4,
                 h_depth=2, deep_layers=None, dropout_shallow=None, dropout_deep=None):
        super(DeepFM, self).__init__()
        # 默认中间有两个连续层，12个节点和8个节点
        if dropout_deep is None:
            dropout_deep = [0.2, 0.2, 0.2]
        if dropout_shallow is None:
            dropout_shallow = [0.2, 0.2]
        if deep_layers is None:
            deep_layers = [12, 8]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deep_layers = deep_layers
        self.h_depth = h_depth
        self.emb_size = embedding_size
        self.feat_sizes = feature_sizes
        self.field_size = field_size
        self.dropout_deep = dropout_deep  # 这个是在deep网络部分使用的dropout系数
        self.dropout_shallow = dropout_shallow  # 这个是在一维特征和组合特征上使用的dropout
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        stdv = math.sqrt(1.0 / len(self.feat_sizes))
        print('初始化deepFM中FM部分')
        self.dropout1 = nn.Dropout(dropout_shallow[0])
        # 这一部分可以看做是LR
        self.fm_first = nn.Embedding(sum(feature_sizes), 1)
        self.fm_first.weight.data.normal_(0, std=stdv)
        # 交叉连接层
        self.fm_second = nn.Embedding(sum(feature_sizes), self.emb_size)
        self.dropout2 = nn.Dropout(dropout_shallow[1])
        self.fm_second.weight.data.normal_(0, std=stdv)
        print('初始化deepFM中Deep模型')
        # 一个全连接层
        self.lin_1 = nn.Linear(self.field_size * self.emb_size, self.deep_layers[0])
        self.deep_drop_0 = nn.Dropout(self.dropout_deep[0])
        self.batch_norm_1 = nn.BatchNorm1d(self.deep_layers[0])
        self.deep_drop_1 = nn.Dropout(self.dropout_deep[1])
        for i, h in enumerate(self.deep_layers[1:], 1):
            setattr(self, 'lin_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(self.deep_layers[i]))
            setattr(self, 'deep_drop_' + str(i + 1), nn.Dropout(self.dropout_deep[i + 1]))
        # self.dropout3 = nn.Dropout(dropout_shallow[2])
        print('初始化Deep模型完成')

    def forward(self, x):
        num_item = x.shape[0]
        shape = x.shape
        x1 = x.view(num_item * self.field_size)
        fm_first = self.fm_first(x1)
        fm_first = fm_first.view(x.size(0), -1)
        fm_first = self.dropout1(fm_first)
        fm_sec_emb = self.fm_second(x1).view(x.size(0), self.field_size, -1)  # (20,7,4)
        #  print('fm_sec_emb:{}'.format(fm_sec_emb.shape))
        fm_sum_sec_emb = torch.sum(fm_sec_emb, 1)  # (20,4)
        #  print('fm_sum_Sec_emb{}'.format(fm_sum_sec_emb.shape))
        # (20,4)
        fm_sum_sec_emb_squ = fm_sum_sec_emb * fm_sum_sec_emb  # (x+y)^2
        # (20,7,4)
        fm_sec_emb_squ = fm_sec_emb * fm_sec_emb
        # (20,4)
        fm_sec_emb_squ_sum = torch.sum(fm_sec_emb_squ, 1)  # x^2+y^2
        fm_second = (fm_sum_sec_emb_squ - fm_sec_emb_squ_sum) * 0.5
        # (20,4)
        fm_second = self.dropout2(fm_second)
        deep_emb = fm_sec_emb.reshape(num_item, -1)
        deep_emb = self.deep_drop_0(deep_emb)
        x_deep = fun.relu(self.batch_norm_1(self.lin_1(deep_emb)))
        x_deep = self.deep_drop_1(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'lin_' + str(i + 1))(x_deep)
            x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            x_deep = fun.relu(x_deep)
            x_deep = getattr(self, 'deep_drop_' + str(i + 1))(x_deep)
        # 返回总的结果
        total_sum = torch.sigmoid(torch.sum(fm_first, 1) + torch.sum(fm_second, 1) + torch.sum(x_deep, 1) + self.bias)
        return total_sum

    def update_deepFM( self,train_x, train_y, batch_x, batch_y, lr=3e-2, num_epoch=10, task='regression',
                      alpha=0.7):
        if task == 'regression':
            cri = nn.MSELoss(reduction='sum')
        else:
            cri = nn.BCELoss(reduction='sum')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idxes = np.array(range((train_y.shape[0])))
        np.random.shuffle(idxes)
        self.train()
        opt = torch.optim.Adam(lr=lr, params=self.parameters())
        num_train = int(train_y.shape[0] * alpha)
        train_x, train_y = torch.Tensor(train_x[idxes][:num_train]).long(), torch.Tensor(train_y[idxes][:num_train])
        batch_x, batch_y = torch.Tensor(batch_x).long(), torch.Tensor(batch_y)
        X = torch.cat([train_x, batch_x], dim=0)
        Y = torch.cat([train_y, batch_y])
        data_set = Data.TensorDataset(X, Y)
        data_loader = Data.DataLoader(data_set, batch_size=X.shape[0] // 5, shuffle=True, drop_last=True)
        for epoch in range(num_epoch):
            for step, (x1, y1) in enumerate(data_loader):
                opt.zero_grad()
                x1, y1 = x1.to(device), y1.to(device)
                out = self.forward(x1)
                loss = cri(out, y1)
                loss.backward()
                opt.step()
        return



def construct_deepfm_model(train_x, train_y, field_size, feat_sizes, lr=3e-2, task='regression', num_epoch=40):
    '''
    利用train_x,train_y训练得到模型
    :param train_x:
    :param train_y:
    :param field_size:
    :param feat_sizes:
    :param lr:
    :param task:
    :param num_epoch:
    :return: 整个模型
    '''
    if task == 'regression':  # 回归
        cri = nn.MSELoss(reduction='sum')
    else:  # 二分类
        cri = nn.BCELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_x, test_x, field_size, feat_sizes = find_deepfm_params(x1=train_x, x2=test_x)
    model: nn.Module = DeepFM(field_size=field_size, feature_sizes=feat_sizes)
    model = model.to(device)
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    # 注意这里全部都要转化为long形式，因为要嵌入
    train_x, train_y = torch.Tensor(train_x).long(), torch.Tensor(train_y)
    data_set = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=train_x.shape[0] // 5, shuffle=True,drop_last=True)
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = cri(outputs, batch_y)
            # if step == 0:
            #     print('预测:{}'.format(outputs[:15]))
            #     print('标准:{}'.format(batch_y[:15]))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 1 == 0:
            print('DeepFM训练过程,第{}次循环，当前loss为：{}'.format(epoch, total_loss))
        total_losses.append(total_loss)
    plt.plot(total_losses, ls='--', color='r')
    plt.scatter(list(range(len(total_losses))), total_losses, color='b')
    plt.title('deepfm losses')
    plt.show()
    return model


def eval_deep_model(model: nn.Module, test_x, test_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_x, train_y = torch.Tensor(test_x).long(), torch.Tensor(test_y)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_x)
    model.train()
    return outputs



