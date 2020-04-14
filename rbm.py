# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#u1 is the first split.....we have five sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
#making a list of lists...one list for every user.
#each matrix will contain all users and all movies...0 rating which hasn't been rated.
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked) so that output and input have same format
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
# or operqator doesn't work with pytorch
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)#matrix of nh*nv dim, mean of zero and var of 1
        self.a = torch.randn(1, nh)#this is the bias for probab of hidden node given visible node
        self.b = torch.randn(1, nv)#bias for visible nodes given the hidden node.
    def sample_h(self, x):# x is visible neurons v in the prob phgivenv
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)#so that bias is applied to each line
        p_h_given_v = torch.sigmoid(activation)#prob of activation of the hidden node, vector of nh elements
        return p_h_given_v, torch.bernoulli(p_h_given_v)#to get vectors of 0 and 1, by random sampling, if random number<p_H_givenv it will be one
    def sample_v(self, y):#we nwwd this to apply gibb's sampling
        wy = torch.mm(y, self.W)#no transpose of W here
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):#k step contrastive divergence (one cycle of sampling V again)- approx the max log likelihood fn. gibb's sampling
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()#vo is the input layer, vk is the input layer after k sampling
        self.b += torch.sum((v0 - vk), 0)#0 coz of dimensionality 2
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100#no. of features we want to detect
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.#s has type float.
    for id_user in range(0, nb_users - batch_size, batch_size):#batch learning
        vk = training_set[id_user:id_user+batch_size]#initially equal to input at start of gibb's chain
        v0 = training_set[id_user:id_user+batch_size]#this is gonna remain constant, will be compared to
        ph0,_ = rbm.sample_h(v0)#,_ to only get the first variable
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))#loss is the distance, not rmse here
       # train_loss += np.sqrt(torch.mean((v0[v0>=0] - vk[v0>=0])**2)) # RMSE here
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))#normalising train loss

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]#training set will be used to activate the hidden nodes.
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:#existent ratings.
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += np.sqrt(torch.mean((vt[vt>=0] - v[vt>=0])**2)) # RMSE here
        s += 1.
print('test loss: '+str(test_loss/s))
