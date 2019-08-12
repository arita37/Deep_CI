import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

class BalancedNeuralNetwork(nn.Module):
  def __init__(self, input_size, representation_size):
      super(BalancedNeuralNetwork, self).__init__()
      self.fu1 = nn.Linear(input_size, input_size*2)
      self.bn = nn.BatchNorm1d(input_size*2)
      self.fu2 = nn.Linear(input_size*2, input_size)
      self.bn2 = nn.BatchNorm1d(input_size)
      self.fu3 = nn.Linear(input_size, representation_size)
      self.fu4 = nn.Linear(representation_size+1, representation_size*2)
      self.fu5 = nn.Linear(representation_size*2, 1)
      
  def forward(self,x,t):
      x = F.relu(self.bn(self.fu1(x)))
      x = F.relu(self.bn2(self.fu2(x)))
      x_representation = F.sigmoid(self.fu3(x))
      x_representation_t = torch.cat([x_representation, t], dim=1)
      x = F.relu(self.fu4(x_representation_t))
      x = self.fu5(x)
      return x, x_representation

def mmd2_lin(X,t):
    ''' Linear MMD '''

    Xc = X[data['treatment'].values==0]
    Xt = X[data['treatment'].values==1]

    mean_control = torch.mean(Xc,1)
    mean_treated = torch.mean(Xt,1)
    mmd = torch.sum(torch.sqrt((2.0*mean_treated - 2.0*mean_control).pow(2)))

    return mmd

def pehe(Y_t, Y_c, y_hat_t, y_hat_c):
  return torch.sqrt(torch.mean(((y_hat_t - y_hat_c) - (Y_t-Y_c)).pow(2)))
  
def train(x, t, y_true, representation_size, epochs, alpha, batch_data=False):
  learning_rate = 1e-3
  bnn = BalancedNeuralNetwork(x.shape[1], representation_size)
  optimizer = torch.optim.Adam(bnn.parameters(), lr=learning_rate,weight_decay=1e-3)
  loss_fn = torch.nn.MSELoss(reduction='mean')
  T = torch.as_tensor(t, dtype=torch.float).unsqueeze(1)
  losses = np.zeros(epochs)
  for i in range(epochs):

      y_pred, rep = bnn(x,T)

      loss = loss_fn(y_pred, y_true)
      loss_mmd = mmd2_lin(rep, t)
      loss = loss+alpha*loss_mmd
      losses[i] = loss.item()
      if i%200==0:
        print(i, loss.item())
        print(loss_mmd)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
  return bnn, losses