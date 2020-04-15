import torch.utils.data as Data
import torch
import math
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt


class Dense_Part(nn.Module):
    def __init__(self, field_size, embedding_size, task='regression'):
        super(Dense_Part, self).__init__()
        self.task = task
        self.lin1 = nn.Linear(field_size * embedding_size, 16)
        self.drop = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.lin2 = nn.Linear(16, 6)
        self.batch_norm2 = nn.BatchNorm1d(6)
        self.lin3 = nn.Linear(6, 1)

    def forward(self, x):
        num_item = x.shape[0]
        x = x.view(num_item, -1)
        x = torch.relu(self.lin1(x))
        x = self.drop(x)
        x = self.batch_norm1(x)
        x = torch.relu(self.lin2(x))
        x = self.drop(x)
        x = self.batch_norm2(x)
        x = self.lin3(x)
        return x


class Gbdt_Dense(nn.Module):
    def __init__(self, field_size, feat_sizes, embedding_size, num_group,task='regression'):
        super(Gbdt_Dense, self).__init__()
        self.num_group = num_group
        self.task=task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed = nn.Embedding(sum(feat_sizes), embedding_size)
        for i in range(1, num_group + 1):
            setattr(self, 'dense_part' + str(i), Dense_Part(field_size, embedding_size))

    def forward(self, x):
        x = self.embed(x)
        list1 = [getattr(self, 'dense_part' + str(i))(x) for i in range(1, self.num_group + 1)]
        x_cat = torch.cat(list1, 1)
        return x_cat

    def predict(self, x):
        self.eval()
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).long()
        with torch.no_grad():
            outputs = self.forward(x)
            preds = torch.sum(outputs, dim=1)
        if self.task != 'regression':
            #分类任务
            preds = torch.sigmoid(preds)
        self.train()
        return np.array(preds)

    def update_model(self,train_x,train_y,batch_x,batch_y,test_num_epoch=10,lr1=3e-2,alpha=0.7):
        cri = nn.MSELoss(reduction='sum')
        device = self.device
        idxes = np.array(range(train_x.shape[0]))
        np.random.shuffle(idxes)
        num_train = int(train_x.shape[0] * alpha)
        train_x = torch.Tensor(train_x[idxes][:num_train]).long()
        train_y = torch.Tensor(train_y[idxes][:num_train])
        batch_x = torch.Tensor(batch_x).long()
        batch_y = torch.Tensor(batch_y)
        X = torch.cat([train_x, batch_x], dim=0)
        Y = torch.cat([train_y, batch_y])
        # print(X.shape)
        # print(Y.shape)
        data_set = Data.TensorDataset(X, Y)
        #去掉最后一个单个的
        data_loader = Data.DataLoader(dataset=data_set, batch_size=X.shape[0] // 5, shuffle=True,drop_last=True)
        opt = torch.optim.Adam(lr=lr1, params=self.parameters())
        for epoch in range(test_num_epoch):
            # outputs = self.forward(batch_x)
            # loss = cri(outputs, batch_y)
            total_loss = 0
            for step, (x1, y1) in enumerate(data_loader):
                opt.zero_grad()
                x1, y1 = x1.to(device), y1.to(device)
                # print(x1.shape)
                # print(y1.shape)
                outputs = self.forward(x1)
                loss = cri(outputs, y1)
                loss.backward()
                total_loss += loss.item()
                opt.step()
            # if epoch % 2 == 0:
            #     print('训练deep模型,第{}轮,当前loss为:{}'.format(epoch, total_loss))
        return



def construct_GBDT_Dense(train_x, train_y, test_x, test_y, field_size, feat_sizes,
                         lr=3e-2, num_epoch=40, task='binary',
                         num_tree_a_group=4):
    if task == 'regression':
        objective = 'regression'
        metric = 'mse'
    else:
        objective = 'binary'
        metric = 'auc'
    params = {
        'task': 'train',
        # 设置提升类型
        'boosting_type': 'gbdt',
        # 目标函数
        'objective': objective,
        # 评估函数
        'metric': metric,
        # 叶子节点数目
        'num_leaves': 10,
        'boost_from_average': True,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'num_threads': -1,
        'learning_rate': 1
    }
    lgb_train = lgb.Dataset(train_x, train_y, params=params)
    lgb_val = lgb.Dataset(test_x, test_y, reference=lgb_train)
    Y = train_y
    gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    pred_leaf = gbm.predict(train_x, pred_leaf=True).reshape(train_x.shape[0], -1).astype('int')
    pred_train = (gbm.predict(data=train_x))
    gbdt_roc = roc_auc_score(Y, pred_train)
 #   print('gbdt,loss:{}'.format(gbdt_roc))
    num_item, num_tree = pred_leaf.shape
    num_group = math.ceil(num_tree / num_tree_a_group)
    print('一共有{}组树'.format(num_group))
    temp_y = np.zeros((num_item, num_group))
    for i in range(num_item):
        val = 0
        for t in range(num_tree):
            l = pred_leaf[i][t]
            val += gbm.get_leaf_output(t, l)
            if (t > 0 and t % num_tree_a_group == 0) or t == num_tree:
                temp_y[i][math.ceil(t / num_tree_a_group) - 1] = val
                val = 0
    train_x, train_y = torch.Tensor(train_x).long(), torch.Tensor(temp_y).float()
    cri = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: nn.Module = Gbdt_Dense(field_size, feat_sizes, embedding_size=3, num_group=num_group,task='binary')
    model = model.to(device)
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    data_set = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=256, shuffle=True, )
    total_losses = []
    roc_auces = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = cri(batch_y, outputs)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                pred_train = model.predict(train_x)
                # print(pred_train[:10])
                # print(Y[:10])
                roc_score = roc_auc_score(np.array(Y), np.array(pred_train))
                roc_auces.append(roc_score)
                print('epoch：{}，roc_auc:{}'.format(epoch, roc_score))
                roc_auces.append(roc_score)
            model.train()
        total_losses.append(total_loss)
    plt.plot(list(range(1, len(roc_auces) + 1)), roc_auces, label='deep_model_roc', color='g', lw=2, ls=':')
    plt.scatter(list(range(1, len(roc_auces) + 1)), roc_auces, color='y')
    plt.plot(list(range(1, len(roc_auces) + 1)), [gbdt_roc] * len(roc_auces), label='gbdt_roc', color='b', lw=2,ls='--')
    plt.scatter(list(range(1, len(roc_auces) + 1)), [gbdt_roc] * len(roc_auces), color='m')
    plt.title('roc_auc')
    plt.legend()
    plt.show()
    return gbm,model

def predict_gbdt_batch(gbm,batch_x,num_tree_a_group):
    num_item=batch_x.shape[0]
    pred_leaf = gbm.predict(batch_x, pred_leaf=True).reshape(batch_x.shape[0], -1).astype('int')
    num_tree = pred_leaf.shape[1]
    num_group = math.ceil(num_tree / num_tree_a_group)
    batch_y = np.zeros((num_item, num_group))
    for i in range(num_item):
        val = 0
        for t in range(num_tree):
            l = pred_leaf[i][t]
            val += gbm.get_leaf_output(t, l)
            if (t > 0 and t % num_tree_a_group == 0) or t == num_tree:
                batch_y[i][math.ceil(t / num_tree_a_group) - 1] = val
                val = 0
    return batch_y

