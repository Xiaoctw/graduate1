from models.GBDT2NN import *
from preprocess.data_preprocess import *


file_name='Chicago'
if file_name=='database':
    lr=6e-4
    num_tree_a_group=1
    num_epoch = 40
    alpha=0.98
elif file_name=='Chicago':
    lr = 3e-2
    num_epoch = 40
    num_tree_a_group = 3
    alpha=0.6
_, train_x, _, _, test_x, _, train_y, test_y = pre_data(file_name)
train_cate_x, test_cate_x, field_size, feat_sizes = find_deep_params(train_x, test_x)
# print(field_size)
# print(feat_sizes)

gbm,model=construct_GBDT_Dense(train_x=train_x,train_y=train_y,test_x=test_x,test_y=test_y,
                               field_size=field_size,feat_sizes=feat_sizes,num_epoch=num_epoch,lr=lr,
                               num_tree_a_group=num_tree_a_group)
num_test=test_y.shape[0]
num_update=20
batch_size=num_test//num_update
roc_es=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# _len = len(roc_es)
# pred_test = (gbm.predict(data=test_x))
# gbdt_roc = roc_auc_score(test_y, pred_test)
for i in range(num_update):
    beg = i * batch_size
    end = min((i + 1) * batch_size, num_test)
    batch_x = test_x[beg:end]
    tensor_x = torch.Tensor(test_x).long().to(device)
    out1=model.predict(tensor_x)
    roc_val=roc_auc_score(test_y,out1)
    roc_es.append(roc_val)
    print('经过更新次数:{},当前roc:{}'.format(i,roc_val))
    if end==num_test:
        break
    batch_y=predict_gbdt_batch(gbm,batch_x,num_tree_a_group=num_tree_a_group)
    train_y=predict_gbdt_batch(gbm,train_x,num_tree_a_group)
    model.update_model(train_x,train_y,batch_x,batch_y,lr1=lr,alpha=alpha)
len1=len(roc_es)
pred_test=gbm.predict(data=test_x)
gbdt_roc=roc_auc_score(test_y,pred_test)
gbdt=[gbdt_roc]*len1
x_labels=[i+1 for i in range(len1)]
plt.plot(x_labels, roc_es, color='g', label='Gbdt_Dense', lw=2, ls='-')
plt.scatter(x_labels, roc_es, color='m', marker='.')
plt.plot(x_labels, gbdt, color='b', label='gbdt', lw=2, ls='--')
plt.legend()
plt.title('approach gbdt retrain')
plt.show()