import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch
from sklearn.metrics import mean_squared_error,r2_score
from torch.utils.data import Dataset, DataLoader
import shap
from sklearn.model_selection import GridSearchCV

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)
def normalize_X_and_y(X, y, X_mean, y_mean, X_std, y_std):
    X_nor = (X - X_mean) / X_std
    y_nor = (y - y_mean) / y_std
    return X_nor, y_nor

def split_data(df_train_list1,df_test_list1,y_out):
    train =df_train_list1.values
    test=df_test_list1.values
    dim=train.shape[1]
    train_X, train_y = train[:, :dim-y_out], train[:, dim-y_out:]  
    test_X, test_y = test[:, :dim-y_out], test[:, dim-y_out:]
    X_mean, y_mean = train_X.mean(0), train_y.mean(0)
    X_std, y_std = train_X.std(0), train_y.std(0)
    X_train_nor, y_train_nor = normalize_X_and_y(train_X, train_y, X_mean=X_mean, y_mean=y_mean, X_std=X_std, y_std=y_std)
    X_test_nor, y_test_nor = normalize_X_and_y(test_X, test_y, X_mean=X_mean, y_mean=y_mean, X_std=X_std, y_std=y_std)
    return X_mean, y_mean,X_std,y_std,X_train_nor, y_train_nor, X_test_nor, y_test_nor,train_y,test_y,test_X,train_X

def evaluate_regress(y_pre, y_true,y_name=''):
    print('*****************************************************')
   
    MAE=np.sum(np.abs(y_pre-y_true))/len(y_true)
    print(y_name+'MAE: ',str(MAE))

    MCE=np.sum(np.abs((y_pre-y_true)/max(y_true)))/len(y_true)
    print(y_name+'MAPE: ',str(MCE))

    MSE=np.sum((y_pre-y_true) ** 2)/len(y_true)
    print(y_name+'MSE: ',str(MSE))
    
    RMSE=np.sqrt(MSE)
    print(y_name+'RMSE: ',str(RMSE))

    R2=r2_score(y_true, y_pre)
    print(y_name+'R2: ',str(R2))

    print('*****************************************************')

    return MAE,MCE,MSE,RMSE,R2

##################################################################################################

class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y
    
##################################################################################################
def train_mutimodel(X_train_nor, y_train_nor, X_test_nor,y_test_nor,y_mean,y_std,method_choose):
    #method_choose='LightGBM'/'XGBoost'/'CatBoost'/'SVR'/'RF'/'MLP'/'DT'/'ELM'/'Bay'/'GBR'/'KNN'/'Lasso'

    y_train_predict1=np.ones((X_train_nor.shape[0],y_train_nor.shape[1]))
    y_test_predict1=np.ones((X_test_nor.shape[0],y_train_nor.shape[1]))
    dataloader_train=0
    dataloader_test=0
    model_list=[]

    if method_choose  in ('KAN'):
        y_train_predict1,y_test_predict1,model_list,dataloader_train,dataloader_test=train_KAN_model(X_train_nor, y_train_nor, X_test_nor,y_test_nor,y_mean,y_std,method_choose)
        
    else:
        for i in range(0,y_train_nor.shape[1]):

            if method_choose=='XGBoost':
                from xgboost import XGBRegressor
                params2 = {
                     'n_estimators': [100, 200, 300],
                     'learning_rate': [0.01, 0.05, 0.1],
                     'max_depth': [4, 5, 6],
                     'subsample': [0.6, 0.8, 1.0],
                     'colsample_bytree': [0.6, 0.8, 1.0],
                     'reg_alpha': [0, 0.5, 1.0],
                     'reg_lambda': [0.5, 1.0, 2.0]
                }
                model2 = XGBRegressor(
                    random_state=42,
                    verbosity=0
                )
                model = GridSearchCV(model2, param_grid=params2, scoring='r2', cv=5, n_jobs=-1)
                model.fit(X_train_nor, y_train_nor[:, i])
                print("Best R²:", model.best_score_)
                print("Best Params:", model.best_params_)
                
            elif method_choose=='KNN':

                from sklearn.neighbors import KNeighborsRegressor
                params11 = {
                     'n_neighbors': [3, 5, 7],
                     'weights': ['uniform', 'distance'],
                     'p': [1, 2]
                }
                model11 = KNeighborsRegressor()
                model = GridSearchCV(model11, param_grid=params11, scoring='r2', cv=5, n_jobs=1)
                model.fit(X_train_nor, y_train_nor[:, i])
                print("Best R²:", model.best_score_)
                print("Best Params:", model.best_params_)
           
            else:
                raise ValueError(f"Unknown method: {method_choose}")

            model_list.append(model)

        y_test_predict_nor=model.predict(X_test_nor).flatten()

        y_train_predict_nor=model.predict(X_train_nor).flatten()

        y_test_predict=y_test_predict_nor*y_std[i]+y_mean[i]

        y_train_predict=y_train_predict_nor*y_std[i]+y_mean[i]

        y_train_predict1[:,i]=y_train_predict

        y_test_predict1[:,i]=y_test_predict

    return y_train_predict1,y_test_predict1,model_list,dataloader_train,dataloader_test

##################################################################################################
def test_method(y_train_predict,y_test_predict,train_y,test_y,method_label):
    train_ev=[]
    test_ev=[]

    dim=test_y.shape[1]

    for i in range(0,dim):
        
        method_label1=method_label+'Training set_Output'+str(i+1)+' '

        method_label3=method_label+'Test set_output '+str(i+1)+' '

        y_train_predict=y_train_predict.reshape((train_y.shape[0],train_y.shape[1]))

        y_test_predict=y_test_predict.reshape((test_y.shape[0],test_y.shape[1]))

        if dim==1:

            MAE_train,MCE_train,MSE_train,RMSE_train,R2_train=evaluate_regress(y_train_predict, train_y,y_name=method_label1)
            
            MAE_test,MCE_test,MSE_test,RMSE_test,R2_test=evaluate_regress(y_test_predict, test_y,y_name=method_label3)
        else:
            MAE_train,MCE_train,MSE_train,RMSE_train,R2_train=evaluate_regress(y_train_predict[:,i], train_y[:,i],y_name=method_label1)
            
            MAE_test,MCE_test,MSE_test,RMSE_test,R2_test=evaluate_regress(y_test_predict[:,i], test_y[:,i],y_name=method_label3)
    
        train_ev.append([MAE_train,MCE_train,MSE_train,RMSE_train,R2_train])

        test_ev.append([MAE_test,MCE_test,MSE_test,RMSE_test,R2_test])

    return train_ev,test_ev

##################################################################################################
df_all = pd.read_excel('ABC.xlsx')
df_process = df_all.dropna()
df_process = df_process.sample(frac=1, random_state=42).reset_index(drop=True)
train_test_split = [0.9, 0.1]
y_out=1
split_index = int(train_test_split[0] * len(df_process))
df_train_list = df_process.iloc[:split_index]
df_test_list = df_process.iloc[split_index:]
X_mean, y_mean,X_std,y_std,X_train_nor, y_train_nor, X_test_nor, y_test_nor,train_y,test_y,test_X,train_X=split_data(df_train_list,df_test_list,y_out)
indices = np.arange(X_train_nor.shape[0])
np.random.shuffle(indices)
X_train_nor = X_train_nor[indices]
y_train_nor = y_train_nor[indices]
train_y=train_y[indices]

method_label='KNN'
y_train_predict1,y_test_predict1,model_list,dataloader_train,dataloader_test=train_mutimodel(X_train_nor, y_train_nor, X_test_nor,y_test_nor,y_mean,y_std,method_label)
train_ev,test_ev=test_method(y_train_predict1,y_test_predict1,train_y,test_y,method_label)

##################################################################################################

method_choose=['KNN']
train_ev_list=[]
test_ev_list=[]
y_train_predict1_list=[]
y_test_predict1_list=[]
name_label_list=[]
dataloader_train_list=[]
dataloader_test_list=[]
model_list_all=[]

for method_label in method_choose:

    y_train_predict1,y_test_predict1,model_list,dataloader_train,dataloader_test=train_mutimodel(X_train_nor, y_train_nor,X_test_nor,y_test_nor,y_mean,y_std,method_label)

    train_ev,test_ev=test_method(y_train_predict1,y_test_predict1,train_y,test_y,method_label)

    name_label_list.append(method_label)

    train_ev_list.append(train_ev)

    test_ev_list.append(test_ev)

    y_train_predict1_list.append(y_train_predict1)

    y_test_predict1_list.append(y_test_predict1)

    model_list_all.append(model_list)

    dataloader_train_list.append(dataloader_train)

    dataloader_test_list.append(dataloader_test)

name_label_list

evaluate_list=['MAE','MAPE','MSE','RMSE','R2']

y_test_out_put1_Df_list=[]
y_test_out_put1_Df_com=0
for j in range(0,y_out):
    y_test_out_put1=np.zeros((len(name_label_list),5))
    for i in range(0,len(name_label_list)):
        test_ev1=test_ev_list[i]
        y_test_out_put1[i,:]=test_ev1[j]

    y_test_out_put1_Df=pd.DataFrame(data=y_test_out_put1,columns=evaluate_list, index=name_label_list)
    y_test_out_put1_Df_com=y_test_out_put1_Df_com+y_test_out_put1_Df
    print(y_test_out_put1_Df)
    y_test_out_put1_Df_list.append(y_test_out_put1_Df)

    
y_test_out_put1_Df_com=y_test_out_put1_Df_com/y_out
print("***************************")
print(y_test_out_put1_Df_com)
y_test_out_put1_Df_com

#######SHAP-KNN########
name_label_list
model_list_all
index_choose=0
model_get=model_list_all[index_choose]
shap_values_list=[]
mean_shap_value_m=[]

if name_label_list[index_choose] in ('LightGBM','XGBoost','RF','DT','Bay','GBR','CatBoost','SVR','KNN','ET') :
    print("** "+name_label_list[index_choose]+'**')
    for i in range(0,y_out):
        now_model=model_get[0]
        df_train_list_X=df_train_list.iloc[:,:-y_out]
        df_test_list_X=df_test_list.iloc[:,:-y_out]
        
        if isinstance(now_model, GridSearchCV):
            now_model = now_model.best_estimator_
        else:
            now_model = now_model
        if name_label_list[index_choose] == 'CatBoost':
            explainer = shap.Explainer(now_model)
        elif name_label_list[index_choose] in ('SVR', 'KNN'):
            print("KNN KernelExplainer")
            explainer = shap.KernelExplainer(now_model.predict, df_train_list_X.sample(50, random_state=42))
        else:
            explainer = shap.Explainer(now_model, df_train_list_X)

        shap_values = explainer(df_test_list_X)
        shap_values_list.append(shap_values)
        mean_shap_value = abs(shap_values.values)
        mean_shap_value1 = mean_shap_value.mean(0)
        mean_shap_value_m.append(mean_shap_value1)

        print('*******************************************************')
        print('**'+str(i+1)+'**')

        plt.figure()
        plt.gcf().set_size_inches(7, 6)
        plt.title("mean(|SHAP value|)(average impact on model output magnitude)")
        shap.summary_plot(shap_values, df_test_list_X, plot_type='bar', show=False)
        ax = plt.gca()
        for bar in ax.patches:
            bar.set_color("#ADD8E6")
        
        shap.summary_plot(shap_values, df_test_list_X, cmap='coolwarm', show=False)
        plt.show()

        plt.figure()
        plt.gcf().set_size_inches(7, 6)

        shap.plots.heatmap(shap_values)
        plt.show()

        plt.figure()
        plt.gcf().set_size_inches(7, 6)

        shap.plots.waterfall(shap_values[0])
        plt.show()



