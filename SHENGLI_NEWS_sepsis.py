#!/opt/anaconda3/envs/SA21001115/bin/python


import torch
import numpy as np
import pandas as pd
import numpy as np
import openpyxl
import os
import time
import concurrent.futures
import multiprocessing
import xlrd
import torch
import random
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import torch.nn as nn
import torch.nn.functional as tf
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

device=torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

#torch.distributed.init_process_group(backend="nccl")
def seed_torch(seed):

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)#禁止hash随机化（unknown）

    torch.manual_seed(seed)#CPU种子
    torch.cuda.manual_seed(seed)#当前GPU种子
    torch.cuda.manual_seed_all(seed)#所有GPU种子

    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True#cuDNN是NVIDIA针对神经网络的加速库，牺牲少量精度换取运算效率

seed_number=2333


def NEWS_score(data):
    NEWS=pd.DataFrame(index=data.index,columns=['NEWS_score','呼吸_score','血氧_score','收缩压_score','心率_score','意识_score','体温_score','sepsis','危险程度'],data=0)
    #用for循环简单明了
    for i in data.index:
        if (data.loc[i,'呼吸']<=8)|(data.loc[i,'呼吸']>=25):
            NEWS.loc[i,'呼吸_score']=3
        if (data.loc[i,'呼吸']>=21)&(data.loc[i,'呼吸']<=24):
            NEWS.loc[i,'呼吸_score']=2
        if (data.loc[i,'呼吸']>=9)&(data.loc[i,'呼吸']<=11):
            NEWS.loc[i,'呼吸_score']=1

        if data.loc[i,'血氧']<=91:
            NEWS.loc[i,'血氧_score']=3
        if (data.loc[i,'血氧']>=92)&(data.loc[i,'血氧']<=93):
            NEWS.loc[i,'血氧_score']=2
        if (data.loc[i,'血氧']>=94)&(data.loc[i,'血氧']<=95):
            NEWS.loc[i,'血氧_score']=1

        if (data.loc[i,'收缩压']<=90)|(data.loc[i,'收缩压']>=220):
            NEWS.loc[i,'收缩压_score']=3
        if (data.loc[i,'收缩压']>=91)&(data.loc[i,'收缩压']<=100):
            NEWS.loc[i,'收缩压_score']=2
        if (data.loc[i,'收缩压']>=101)&(data.loc[i,'收缩压']<=110):
            NEWS.loc[i,'收缩压_score']=1

        if (data.loc[i,'心率']<=40)|(data.loc[i,'心率']>=131):
            NEWS.loc[i,'心率_score']=3
        if (data.loc[i,'心率']>=111)&(data.loc[i,'心率']<=130):
            NEWS.loc[i,'心率_score']=2
        if ((data.loc[i,'心率']>=41)&(data.loc[i,'心率']<=50))|((data.loc[i,'心率']>=91)&(data.loc[i,'心率']<=110)):
            NEWS.loc[i,'心率_score']=1

        if data.loc[i,'意识']!=0:
            NEWS.loc[i,'意识_score']=3

        if data.loc[i,'体温']<=35:
            NEWS.loc[i,'体温_score']=3
        if data.loc[i,'体温']>=39.1:
            NEWS.loc[i,'体温_score']=2
        if ((data.loc[i,'体温']>=35.1)&(data.loc[i,'体温']<=36))|((data.loc[i,'体温']>=38.1)&(data.loc[i,'体温']<=39)):
            NEWS.loc[i,'体温_score']=1

    NEWS['NEWS_score']=NEWS.apply(lambda x:x.sum(),axis=1)
    if 'sepsis' in data.columns:
        NEWS['sepsis']=data.loc[:,'sepsis']
    if '危险程度' in data.columns:
        NEWS['危险程度']=data.loc[:,'危险程度']
    return NEWS


def NEWS_AUC(data,target_name):
    NEWS=NEWS_score(data)
    NEWS_fpr,NEWS_tpr,thersholds=roc_curve(NEWS.loc[:,target_name].astype('int64'),NEWS['NEWS_score'])
    NEWS_auc=auc(NEWS_fpr,NEWS_tpr)
    return NEWS_auc

def set_split(data,p1,p2,target_name):
    #seed_torch(seed_number)
    tfd_train = {
        0: p1,
        1: p1
    }

    tfd_valid ={
        0: p2/(1-p1),
        1: p2/(1-p1)
    }
    def tsampling(group, tfd):
        name = group.name
        frac = tfd[name]
        return group.sample(frac=frac)

    train = data.groupby(target_name,group_keys=False).apply(tsampling, tfd_train)
    d_da0 = pd.concat([data, train,train]).drop_duplicates(keep= False)
    valid = d_da0.groupby(target_name,group_keys=False).apply(tsampling, tfd_valid)
    test = pd.concat([d_da0, valid,valid]).drop_duplicates(keep= False)

    return train,valid,test

#/home/SA21001115/SHENGLI/
data_path = '/home/SA21001115/SHENGLI/wkp_lqy_220508-6.CSV'
da = pd.read_csv(data_path,encoding='gbk')
choose_feature=['意识','体温','心率','呼吸','收缩压','血氧','白细胞计数']
choose_feature_c=['体温','心率','呼吸','收缩压','血氧','白细胞计数']
data=da.loc[:,choose_feature]
data['sepsis']=da.loc[:,'sepsis']

data.loc[data['意识']=='A','意识']=0
data.loc[(data['意识']=='B')|(data['意识']=='C'),'意识']=1

#分别得到阳性病人集和阴性病人集
p_data=data.loc[data['sepsis']==1]
n_data=data.loc[data['sepsis']==0]

def Norm(dat,col):
    m=dat[col].mean()
    s=dat[col].std()
    return (dat[col]-m)/s
d = ['意识','sepsis']

def LR_AUC(train,valid,test,seed_number,target_name,choose_feature):
    for col in train.columns:
        if col in d:
            train[col]=train[col]
        else:
            train[col]=Norm(train,col)

    for col in valid.columns:
        if col in d:
            valid[col]=valid[col]
        else:
            valid[col]=Norm(valid,col)

    for col in test.columns:
        if col in d:
            test[col]=test[col]
        else:
            test[col]=Norm(test,col)

    #seed_torch(seed_number)
    x_train=torch.tensor(train.loc[:,choose_feature].values.astype("float"))
    y_train=torch.tensor(train.loc[:,target_name].values.astype("int64"))
    x_valid=torch.tensor(valid.loc[:,choose_feature].values.astype("float"))
    y_valid=torch.tensor(valid.loc[:,target_name].values.astype("int64"))
    x_test=torch.tensor(test.loc[:,choose_feature].values.astype("float"))
    y_test=torch.tensor(test.loc[:,target_name].values.astype("int64"))
    param_grid={
        'tol': list(np.arange(1e-2,1e-1,2e-2))
        #'C': list(np.arange(1e-2,1,5e-2))
    }
    #随机搜索最大次数
    MAX_EVALS=1

    #记录最优超参数
    best_score = 0
    for j in range(MAX_EVALS):

        #选取超参数
        #random.seed(j)
        random_params = {k: random.sample(v,1)[0] for k, v in param_grid.items()}
        tol=random_params['tol']
        LogR=LogisticRegression(
            penalty='none',
            tol=tol,
            max_iter=9e2,
            solver='newton-cg'
        ).fit(x_train,y_train)

        LogR_pre=LogR.predict_proba(x_valid)[:,1].flatten()
        LogR_fpr,LogR_tpr,thersholds=roc_curve(y_valid,LogR_pre)
        LogR_auc=auc(LogR_fpr,LogR_tpr)

        if LogR_auc>best_score:
            best_score=LogR_auc
            joblib.dump(LogR,'/home/SA21001115/SHENGLI/SHENGLI_NEWS_sepsis/best_sklearn_LogR.dat')

    LogR=joblib.load('/home/SA21001115/SHENGLI/SHENGLI_NEWS_sepsis/best_sklearn_LogR.dat')
    LogR_pre=LogR.predict_proba(x_test)[:,1].flatten()
    LogR_fpr,LogR_tpr,thersholds=roc_curve(y_test,LogR_pre)

    LogR_auc=auc(LogR_fpr,LogR_tpr)
    return LogR_auc,LogR


def cal_new_NEWS_g(data,target_name):
    NEWS=NEWS_score(data)
    g_0=NEWS[NEWS[target_name]==0]['NEWS_score']
    g_1=NEWS[NEWS[target_name]==1]['NEWS_score']
    new_g0=np.exp(g_0-3)/(1+np.exp(g_0-3))
    new_g1=np.exp(g_1-3)/(1+np.exp(g_1-3))
    return new_g0,new_g1

def NEWS_AUC(data,target_name):
    NEWS=NEWS_score(data)
    NEWS_fpr,NEWS_tpr,thersholds=roc_curve(NEWS.loc[:,target_name].astype('int64'),NEWS['NEWS_score'])
    NEWS_auc=auc(NEWS_fpr,NEWS_tpr)
    return NEWS_auc

def KLD_AUC(train,valid,test,seed_number,new_g0,new_g1,target_name,choose_feature):




    for col in train.columns:
        if col in d:
            train[col]=train[col]
        else:
            train[col]=Norm(train,col)

    for col in valid.columns:
        if col in d:
            valid[col]=valid[col]
        else:
            valid[col]=Norm(valid,col)

    for col in test.columns:
        if col in d:
            test[col]=test[col]
        else:
            test[col]=Norm(test,col)
    train0 = train.copy(deep=True)
    train0.loc[:, 'sepsis'] = 0

    train1 = train.copy(deep=True)
    train1.loc[:, 'sepsis'] = 1

    train_fin = pd.concat([train, train0, train1], ignore_index=True)

    x_train = torch.tensor(train_fin.loc[:, choose_feature].values.astype("float")).to(device)
    y_train = torch.tensor(train_fin.loc[:, 'sepsis'].values.astype("int64")).to(device)
    #NN的随机种子设置（一定要设，不然对结果影响非常大，不利于结果分析）
    #seed_torch(seed_number)
    param_grid={
        'learning_rate': list(np.arange(1e-3,2e-1,2e-3)),
        'epochs': list(np.arange(10,100,20)),
        'alpha': list(np.arange(0,1,1e-2)),
        'beta': list(np.arange(0,1,1e-2))
    }
    #print('1008')
    #数据x&y
    x_valid=torch.tensor(valid.loc[:,choose_feature].values.astype("float")).to(device)
    y_valid=torch.tensor(valid.loc[:,target_name].values.astype("int64")).to(device)
    x_test=torch.tensor(test.loc[:,choose_feature].values.astype("float")).to(device)
    y_test=torch.tensor(test.loc[:,target_name].values.astype("int64")).to(device)
    #随机搜索最大次数
    MAX_EVALS=50
    #print('1009')
    #记录最优超参数
    best_score = 0
    best_Lambda0=0
    best_Lambda1=0
    for j in range(MAX_EVALS):

        #选取超参数
        #random.seed(j)
        random_params = {k: random.sample(v,1)[0] for k, v in param_grid.items()}

        learning_rate=random_params['learning_rate']
        epoch=random_params['epochs']
        #epoch=70
        alpha=random_params['alpha']
        #LAMBDA0=0
        beta=random_params['beta']
        #LAMBDA1=0

        weight0 = []
        weight0.extend((1 - alpha) * beta * (1 - new_g0))
        weight0.extend((1 - alpha) * (1 - beta) * (1 - new_g1))

        weight1 = []
        weight1.extend((1 - alpha) * beta * new_g0)
        weight1.extend((1 - alpha) * (1 - beta) * new_g1)

        weight = []
        weight.extend(alpha * np.ones(train.shape[0]))
        weight.extend(np.array(weight0))
        weight.extend(np.array(weight1))
        weight = torch.tensor(weight).to(device)
        fc_net=nn.Sequential(
            nn.Linear(len(choose_feature),1)
        ).double()
        #if torch.cuda.device_count() > 1:
            #print("Let's use", torch.cuda.device_count(), "GPUs!")
            #fc_net = nn.DataParallel(fc_net)
        fc_net.to(device)


        #print('1010')
        #fc_net.apply(weight_init)

        loss_func=nn.functional.binary_cross_entropy
        #print('1111')
        #loss_func=loss_func.to(device)
        #print('1112')
        optimizer = torch.optim.Adam(fc_net.parameters(), lr=learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.05)
        #print('1011')
        for i in range(epoch):
            lpre_y = fc_net(x_train)
            pre_y = lpre_y.exp() / (1 + lpre_y.exp())
            #这两个loss的输入格式不一样，需要注意
            loss = loss_func(pre_y.flatten(), y_train.double(),weight=weight,reduction='sum')
            #epoch次数太多之后，f_0会非常接近0&1，这样在kl_div的计算中，f_0.log()会报错，此外，为了避免这种情况，还要对数据做标准化处理(总之数据标准化之后，就没出现数值错误)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        out= fc_net(x_valid)

        out=out.exp()/(1+out.exp())
        #out=out.to(device)
        #print(out[:,0])
        #fpr, tpr, thersholds = roc_curve(y_valid, out[:,0])

        #print(y_valid)
        #print(out[:,0].detach().cpu().numpy())
        auc_nn = roc_auc_score(y_valid.cpu().numpy(),out[:,0].detach().cpu().numpy())#涉及梯度传播的要用detach()来去梯度运算
        #print(auc_nn)
        score=auc_nn
        #print('1015')
        if score>best_score:
            best_score=score
            best_alpha=alpha
            best_beta=beta
            torch.save(fc_net,"/home/SA21001115/SHENGLI/SHENGLI_NEWS_sepsis/best_LRKLD.pkl")

    best_net=torch.load('/home/SA21001115/SHENGLI/SHENGLI_NEWS_sepsis/best_LRKLD.pkl')
    #结果
    out= best_net(x_test)
    out=out.exp()/(1+out.exp())
    auc_nn = roc_auc_score(y_test.cpu().numpy(), out[:,0].detach().cpu().numpy())
    return auc_nn,best_net,best_alpha,best_beta

all_p1=[0.05,0.1,0.2,0.4]
p2=0.2
choose_feature=['意识','体温','心率','呼吸','收缩压','血氧','白细胞计数']
p_NUM=[1000,1000,1000,1000,200,100]
n_NUM=[100,1000,5000,10000,10000,10000]
prop_set=['1:10','1:1','5:1','10:1','50:1','100:1']
TARGET_NAME='sepsis'
NUM=1000
best_Lambda0=best_Lambda1=k=0
choose_feature=['意识','体温','心率','呼吸','收缩压','血氧','白细胞计数']
METHODS=['NEWS','LR','LRKLD']
plt.rcParams.update({"font.size":20})
#fig=plt.figure(figsize=(25,25))
start= time.time()
for p1 in all_p1:
    AUC_RES=pd.DataFrame(columns=['AUC','prop','methods'])
    param_RES=pd.DataFrame(columns=['prop','alpha','beta'])
    k=0
    #print(p1)
    for p in prop_set:
        #print(p)
        for i in range(NUM):
            #print(i)
            try:
                p_set=p_data.sample(n=p_NUM[prop_set.index(p)],random_state=i)
                n_set=n_data.sample(n=n_NUM[prop_set.index(p)],random_state=i)
                pn_set=pd.concat([p_set,n_set])
                train,valid,test=set_split(pn_set,p1,p2,TARGET_NAME)
                new_g0,new_g1=cal_new_NEWS_g(train,TARGET_NAME)

                AUC_RES.loc[3*k,'AUC']=NEWS_AUC(pn_set,target_name=TARGET_NAME)
                AUC_RES.loc[3*k,'prop']=p
                AUC_RES.loc[3*k,'methods']='NEWS'


                AUC_RES.loc[3*k+1,'AUC'],best_LR=LR_AUC(train,valid,test,i,TARGET_NAME,choose_feature)
                AUC_RES.loc[3*k+1,'prop']=p
                AUC_RES.loc[3*k+1,'methods']='LR'

                AUC_RES.loc[3*k+2,'AUC'],best_LRKLD,best_alpha,best_beta= KLD_AUC(train,valid,test,seed_number,new_g0,new_g1,TARGET_NAME,choose_feature)
                AUC_RES.loc[3*k+2,'prop']=p
                AUC_RES.loc[3*k+2,'methods']='LRKLD'
                #print(best_Lambda0,best_Lambda1)
                #print('LRKLD')
                print(AUC_RES.loc[3*k,'AUC'],AUC_RES.loc[3*k+1,'AUC'],AUC_RES.loc[3*k+2,'AUC'])
                param_RES.loc[k,'alpha']=best_alpha
                param_RES.loc[k, 'beta']=best_beta
                param_RES.loc[k, 'prop']= p
                k=k+1
            except:
                continue
                #/data/SA21001115/
    AUC_RES.to_csv('/data/SA21001115/SHENGLI_NEWS_sepsis_auc%f'%p1+'.csv')
    param_RES.to_csv('/data/SA21001115/SHENGLI_NEWS_sepsis_best_param%f'%p1+'.csv')
'''
    ax=fig.add_subplot(2,2,all_p1.index(p1)+1)
    seaborn.boxplot(data=AUC_RES,
                    x='prop',
                    hue='methods',
                    y='AUC',
                    linewidth=0.75,
                    width=0.75,
                    palette='Blues',
                    fliersize=0
                    )
    ax.set_title('Train set Prop='+str(p1))
'''
end=time.time()
print(end-start)
