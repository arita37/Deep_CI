"""
Implementation of Dragonnet from: https://arxiv.org/pdf/1906.02120.pdf
"""


from torch import nn
import torch.nn.functional as F
import torch
import ipdb
import pandas as pd
from torch.utils.data import Dataset

class IHDP_Dataset(Dataset):
    def __init__(self):
        dfs = []
        for i in range(1,11,1):
            file_name = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_"+str(i)+".csv"
            data= pd.read_csv(file_name, header = None)
            dfs.append(data)
            print(i)
        self.data = pd.concat(dfs)
        col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]

        for i in range(1,26):
            col.append("x"+str(i))
        self.data.columns = col


    def __getitem__(self, index):
        return self.data[[c for c in self.data.columns if 'x' in c]].iloc[index].values, self.data[['y_factual', 'treatment','y_cfactual']].iloc[index].values

    def __len__(self):
        return self.data.shape[0]

def treatment_loss(labels, predictions):
    t_true = labels[:, 1]
    t_pred = predictions[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    loss = torch.sum(F.binary_cross_entropy(t_pred, t_true))

    return loss

def outcome_loss(labels, predictions):

    y_true = labels[:, 0]
    t_true = labels[:, 1]

    y0_pred = predictions[:, 0]
    y1_pred = predictions[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.pow((y_true - y0_pred), 2))
    loss1 = torch.sum(t_true * torch.pow((y_true - y1_pred),2))

    return loss0 + loss1

def overall_loss(labels, predictions, ratio=1.):
    vanilla_loss = outcome_loss(labels, predictions) + treatment_loss(labels, predictions)

    y_true = labels[:, 0]
    t_true = labels[:, 1]

    y0_pred = predictions[:, 0]
    y1_pred = predictions[:, 1]
    t_pred = predictions[:, 2]

    epsilons = predictions[:, 3]
    t_pred = (t_pred + 0.01) / 1.02

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + epsilons * h
    targeted_regularization = torch.sum(torch.pow((y_true - y_pert), 2))
    loss = vanilla_loss + ratio * targeted_regularization

    return loss

class EpsilonLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)
    def forward(self, x):
        eps = self.weight * x
        return eps 

class DragonNet(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.hidden_size = params.hidden_size
        self.input_size = params.input_size
        self.batch_size = params.batch_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)

        self.treatment_predictor = nn.Linear(self.hidden_size, 1)

        self.y0_fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.y1_fc = nn.Linear(self.hidden_size, self.hidden_size)

        self.y0_out = nn.Linear(self.hidden_size, 1)
        self.y1_out = nn.Linear(self.hidden_size, 1)
        self.epsilons = EpsilonLayer()

    def forward(self, x):

        rep = F.elu(self.fc1(x))
        rep = F.elu(self.fc2(rep))
        rep = F.elu(self.fc3(rep))

        treatment_prediction = F.sigmoid(self.treatment_predictor(rep))

        y0 = F.elu(self.y0_fc(rep))
        y0_prediction = self.y0_out(y0)

        y1 = F.elu(self.y1_fc(rep))
        y1_prediction = self.y1_out(y1)

        eps = self.epsilons(torch.ones_like(treatment_prediction))
        return torch.cat([y0_prediction, y1_prediction,treatment_prediction, eps], axis=-1)




