import numpy as np
import torch
import sys, copy, math, time, pdb
import os.path
import random
import pdb
import csv
import argparse
import itertools
from itertools import permutations, product
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchdiffeq import odeint
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import optuna

# 路径直接在这里定义  
filepath_train = f'GenusSample.csv'
filepath_test = f'Ztest.csv'

def get_batch(ztrn, ptrn, mb_size):

    actual_mb_size = min(mb_size, ptrn.size(dim=0))
    s = torch.from_numpy(np.random.choice(np.arange(ptrn.size(dim=0), dtype=np.int64), actual_mb_size, replace=False))
    batch_p = ztrn[s, :]
    batch_q = ptrn[s, :]
    batch_t = t[:batch_time]
    return batch_p.to(device), batch_q.to(device), batch_t.to(device)

def loss_bc(p_i, q_i):
    return torch.sum(torch.abs(p_i - q_i)) / torch.sum(torch.abs(p_i + q_i))

def process_data(P):
    Z = P.copy()
    Z[Z > 0] = 1
    P = P / P.sum(axis=0)[np.newaxis, :]
    Z = Z / Z.sum(axis=0)[np.newaxis, :]
    
    P = P.astype(np.float32)
    Z = Z.astype(np.float32)

    P = torch.from_numpy(P.T)
    Z = torch.from_numpy(Z.T)
    return P, Z

class ODEFunc(torch.nn.Module):
    def __init__(self, N, reg_strength=0.000001):
        super(ODEFunc, self).__init__()
        self.N = N
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)
        self.reg_strength = reg_strength

    def forward(self, t, y):
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1), 1), y), torch.transpose(out, 0, 1))
        return torch.mul(y, out - torch.transpose(f, 0, 1))

    def regularization_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(torch.abs(param))
        return self.reg_strength * reg_loss

def train_reptile(max_epochs, mb, LR, ztrn, ptrn, ztst, ptst, zval, pval, zall, pall, patience=40):
    loss_train = []
    loss_val = []
    qtst = np.zeros((ztst.size(dim=0), N))
    qtrn = np.zeros((zall.size(dim=0), N))
    
    func = ODEFunc(N).to(device)
    optimizer = torch.optim.Adam(func.parameters(), lr=LR)

    Loss_opt = float('inf')
    best_model = None
    epochs_no_improve = 0

    for e in range(max_epochs):
        optimizer.zero_grad()
        batch_p, batch_q, batch_t = get_batch(ztrn, ptrn, mb)
        
        loss = 0
        for i in range(batch_p.size(dim=0)):
            p_pred = odeint(func, batch_p[i].unsqueeze(dim=0), batch_t).to(device)
            p_pred = torch.reshape(p_pred[-1, :, :], (1, N))
            loss += loss_bc(p_pred.unsqueeze(dim=0), batch_q[i].unsqueeze(dim=0))
        loss += func.regularization_loss()
        loss_train.append(loss.item() / batch_p.size(dim=0))

        l_val = 0
        for i in range(zval.size(dim=0)):
            p_pred = odeint(func, zval[i].unsqueeze(dim=0), batch_t).to(device)
            p_pred = torch.reshape(p_pred[-1, :, :], (1, N))
            l_val += loss_bc(p_pred.unsqueeze(dim=0), pval[i].unsqueeze(dim=0))
        val_loss = l_val.item() / zval.size(dim=0)
        loss_val.append(val_loss)
        print(f"[train_reptile] Epoch {e+1}/{max_epochs} done. "
              f"Train loss={loss_train[-1]:.4f}, Val loss={loss_val[-1]:.4f}")

        if val_loss < Loss_opt:
            Loss_opt = val_loss
            best_model = copy.deepcopy(func)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping!")
            break

        func.zero_grad()
        loss.backward()
        optimizer.step()

    func = copy.deepcopy(best_model)
    
    if len(ztst.size()) == 2:
        for i in range(ztst.size(dim=0)):
            pred_test = odeint(func, ztst[i].unsqueeze(dim=0), batch_t).to(device)
            pred_test = pred_test[-1, :, :]
            pred_test = torch.reshape(pred_test, (1, N))
            qtst[i, :] = pred_test.detach().numpy()
        for i in range(zall.size(dim=0)):
            pred_test = odeint(func, zall[i].unsqueeze(dim=0), batch_t).to(device)
            pred_test = pred_test[-1, :, :]
            pred_test = torch.reshape(pred_test, (1, N))
            qtrn[i, :] = pred_test.detach().numpy()
    elif len(ztst.size()) == 1:
        for i in range(ztst.size(dim=0)):
            pred_test = odeint(func, ztst.unsqueeze(dim=0), batch_t).to(device)
            pred_test = pred_test[-1, :, :]
            pred_test = torch.reshape(pred_test, (1, N))
            qtst = pred_test.detach().numpy()
        for i in range(zall.size(dim=0)):
            pred_test = odeint(func, zall[i].unsqueeze(dim=0), batch_t).to(device)
            pred_test = pred_test[-1, :, :]
            pred_test = torch.reshape(pred_test, (1, N))
            qtrn[i, :] = pred_test.detach().numpy()

    return loss_train, loss_val, qtst, qtrn, Loss_opt

device = 'cpu'
batch_time = 100
t = torch.arange(0.0, batch_time, 0.01)

P = pd.read_csv(filepath_train, delimiter=',', header=None).values

number_of_cols = P.shape[1]
random_indices = np.random.choice(number_of_cols, size=int(0.2 * number_of_cols), replace=False)
P_val = P[:, random_indices]
P_train = P[:, np.setdiff1d(range(0, number_of_cols), random_indices)]
ptrn, ztrn = process_data(P_train)
pval, zval = process_data(P_val)
pall, zall = process_data(P)
M, N = ptrn.shape

P_test = pd.read_csv(filepath_test, delimiter=',', header=None).values
ptst, ztst = process_data(P_test)

def objective(trial):
    LR = trial.suggest_float('lr', 1e-4, 1e-1, log=True)  
    mb = trial.suggest_categorical('mb', [16, 32, 64]) 
    max_epochs = trial.suggest_int('max_epochs', 500, 1500)  
    try:
        loss_train, loss_val, _, _, Loss_opt = train_reptile(max_epochs, mb, LR, ztrn, ptrn, ztst, ptst, zval, pval, zall, pall)
        return Loss_opt
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.inf

study = optuna.create_study(direction='minimize')  
try:
    study.optimize(objective, n_trials=20)  
except KeyboardInterrupt:
    print("Optimization was interrupted manually.")

print("Optimal hyperparameters:", study.best_params)

LR = study.best_params['lr']
mb = study.best_params['mb']
max_epochs = study.best_params['max_epochs']

print('Optimal hyperparameters:')
print('learning rate:', LR)
print('mb:', mb)
print('max_epochs:', max_epochs)

loss_train, loss_val, qtst, qtrn, _ = train_reptile(max_epochs, mb, LR, ztrn, ptrn, ztst, ptst, zval, pval, zall, pall)

plt.figure()
plt.plot(loss_train, label='Training loss')
plt.plot(loss_val, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss & Validation Loss')
plt.savefig('loss_curves.png')
plt.show()

np.savetxt('qtst.csv', qtst, delimiter=',')
np.savetxt('qtrn.csv', qtrn, delimiter=',')
