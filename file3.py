from models.DeepFM import *
from models.GBDT2NN import *
from preprocess.data_preprocess import *

file_name='Chicago'
train_num_epoch=40
dim=3
deep_lr=3e-2
alpha=0.6
num_tree_a_group=4
num_update=10

_, train_x, _, _, test_x, _, train_y, test_y = pre_data(file_name)
train_x,test_x,field_size,feat_sizes=find_deep_params(train_x,test_x)
gbm,model2=construct_GBDT_Dense(train_x=train_x,train_y=train_y,test_x=test_x,test_y=test_y,
                               field_size=field_size,feat_sizes=feat_sizes,num_epoch=train_num_epoch,
                               num_tree_a_group=num_tree_a_group)
model1=construct_deepfm_model(train_x=train_x,train_y=train_y,field_size=field_size,
                              feat_sizes=feat_sizes,lr=deep_lr,task='binary',num_epoch=train_num_epoch)
num_test=test_y.shape[0]
batch_size=num_test//num_test
roc_es=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2_roces = []
model1_roces = []
total_roces = []
for i in range(num_update):
    beg = i * batch_size
    end = min((i + 1) * batch_size, num_test)
    batch_x = test_x[beg:end]
    batch_y=test_y[beg:end]
    tensor_test_x = torch.Tensor(test_x).long().to(device)
    #这个才是gbdt对应的预测结果
    out2=np.array(model2.predict(tensor_test_x))
    model1.eval()
    with torch.no_grad():
         out1=np.array(model1(tensor_test_x))
    model1.train()
    model1_roces.append(roc_auc_score(test_y,out1))
    model2_roces.append(roc_auc_score(test_y,out2))
    out=alpha*out1+(1-alpha)*out2
    total_roces.append(roc_auc_score(test_y,out))
    print('epoch:{},roc1:{},roc2:{},roc3:{}'.format(i,round(roc_auc_score(test_y,out1),3),round(roc_auc_score(test_y,out2),3),round(roc_auc_score(test_y,out),3)))
    model1.update_deepFM(train_x,train_y,batch_x,batch_y,alpha=alpha)
    batch_tem_y=predict_gbdt_batch(gbm,batch_x,num_tree_a_group=num_tree_a_group)
    train_tem_y=predict_gbdt_batch(gbm,train_x,num_tree_a_group)
    model2.update_model(train_x,train_tem_y,batch_x,batch_tem_y,alpha=alpha)
    if end==num_test:
        break

plt.plot(model2_roces, color='b', lw=2, label='gbdt')
plt.scatter(list(range(len(model2_roces))), model2_roces, color='g', marker='^', s=40)
plt.plot(model1_roces, color='r', lw=2, label='deepFM')
plt.scatter(list(range(len(model1_roces))), model1_roces, color='g', marker='<', s=40)
plt.plot(total_roces, label='ComDeep', color='y', lw=2)
plt.scatter(list(range(len(total_roces))), total_roces, color='g', marker='>', s=40)
plt.title(file_name)
plt.legend()
plt.show()