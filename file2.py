from models.DeepFM import *
from preprocess.data_preprocess import *
from preprocess.helper import *
from sklearn.metrics import roc_auc_score
file_name='database'
train_num_epoch=80
dim=3
deep_lr=3e-3
alpha=0.6
num_tree_a_group=4
num_update=10

_, train_x, _, _, test_x, _, train_y, test_y = pre_data(file_name)
train_x,test_x,field_size,feat_sizes=find_deep_params(train_x,test_x)
model1=construct_deepfm_model(train_x=train_x,train_y=train_y,field_size=field_size,
                              feat_sizes=feat_sizes,lr=deep_lr,task='binary',num_epoch=train_num_epoch)

predicts=eval_deep_model(model1,test_x,test_y)
print(roc_auc_score(test_y,predicts))