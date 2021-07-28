#Nick Cerone
#7.27.21

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as fn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math

# ======================================== SETUP AND VARS ========================================

#convert each action param to fall within range [0,1] to pass through BCE
def action_to_cvae(action):
  # Input range:
  # Column 0: Velocity, --[-3250, 3250]-- #EDIT: [-1750, 0]
  # Column 1: Steering Angle, [.15, .85]
  ret = np.zeros(action.shape)
  #ret[0] = action[0] / -1750.0
  ret[0] = (action[0] - 0.15) / 0.7
  return ret


#convert each cVAE output on range [0,1] to interpretable actions
def cvae_to_action(cvae):
  #make it 1D
  cvae = cvae.detach().numpy().flatten()

  ret = np.zeros(cvae.shape)
  #ret[0] = cvae[0] * -1750.0
  ret[0] = (cvae[0] * 0.7) + 0.15
  return ret


grid_size = 50 #50 by 50 grid, and 50 wedges coincidentally (2pi / 0.04pi = 50)
def to_one_hot(state):
    ret = torch.zeros((1, grid_size*3))

    #set the x (half are negative, some number within 0 and 50)
    ret[0][int(state[0][0] + 25)] = 1

    #set the y (assumption that none are negative)
    ret[0][int(state[0][1] + 50)] = 1

    #set what wedge of the orien
    ret[0][int(state[0][2] + 125)] = 1

    return ret


#Floor function to discretize the state into the grid
pose_step = 0.08 #estimate average variance in position
orien_step = 0.04 #estimate average variance in orientation
def state_discretize(state):
    #Input State:
    #X PF
    #Y PF
    #Z Orientation
    ret = torch.zeros(state.shape)
    ret[0][0] = math.floor(state[0][0] / pose_step + 0.5) #Floor function based on pose_step - accounts for negative pose values
    ret[0][1] = math.floor(state[0][1] / pose_step + 0.5)
    ret[0][2] = math.floor(state[0][2] / (orien_step * math.pi) + 0.5) #convert to radians
    
    ret = to_one_hot(ret)

    return ret





# initialize some variables
batch_size = 64
input_dim = 1 #Velocity, Wheel Angle
state_dim = 150 #X Driver, Y Driver, Z Orien Driver - to one hot --> 50, 50, 50 = 150
hidden_dim1 = 100
hidden_dim2 = 20
z_dim = 1



# Xavier Initialization - initialize weights with smaller numbers (less variance) based on the batch size to prevent gradient from exploding
def xav_init(bs, out_dim):
    xav_std = 1.0 / np.sqrt(bs / 2.0)

    return Parameter(torch.randn(bs, out_dim) * xav_std, requires_grad=True)


# ======================================== SETUP AND VARS ========================================


# ============================================= CVAE =============================================

class cVAE(nn.Module):
    weights = []
    biases = []

    # =============== Constructor ================

    # initializes random weights and 0 biases
    def __init__(self):
        super(cVAE, self).__init__()

        #Setting weights and biases as module parameters()
        #Encoder
        self.weights_xh1 = xav_init(input_dim + state_dim, hidden_dim1)
        self.weights_h1h2 = xav_init(hidden_dim1, hidden_dim2)
        self.weights_h2Zmu = xav_init(hidden_dim2, z_dim)
        self.weights_h2Zsig = xav_init(hidden_dim2, z_dim)

        self.biases_xh1 = Parameter(torch.zeros(hidden_dim1), requires_grad=True)
        self.biases_h1h2 = Parameter(torch.zeros(hidden_dim2), requires_grad=True)
        self.biases_h2Zmu = Parameter(torch.zeros(z_dim), requires_grad=True)
        self.biases_h2Zsig = Parameter(torch.zeros(z_dim), requires_grad=True)


        #Decoder
        self.weights_zh2 = xav_init(z_dim + state_dim, hidden_dim2)
        self.weights_h2h1 = xav_init(hidden_dim2, hidden_dim1)
        self.weights_h1xhat = xav_init(hidden_dim1, input_dim)

        self.biases_zh2 = Parameter(torch.zeros(hidden_dim2), requires_grad=True)
        self.biases_h2h1 = Parameter(torch.zeros(hidden_dim1), requires_grad=True)
        self.biases_h1xhat = Parameter(torch.zeros(input_dim), requires_grad=True)

    # =============== Constructor ================


   

    # ================== Decoder ==================

    # Decoder function
    
    # Decoder function
    def P(self, z, s):
        # concatonate over the labels
        z_prime = torch.cat([z, s], 1)

        # do more dot products to reconstruct the matrix now
        p_hidden2 = torch.relu(torch.matmul(z_prime, self.weights_zh2) + self.biases_zh2.repeat(z_prime.size(0), 1))
        p_hidden1 = torch.relu(torch.matmul(p_hidden2, self.weights_h2h1) + self.biases_h2h1.repeat(p_hidden2.size(0), 1))


        X_hat = torch.sigmoid(torch.matmul(p_hidden1, self.weights_h1xhat) + self.biases_h1xhat.repeat(p_hidden1.size(0), 1))

        return X_hat

    # ================== Decoder ==================

# ============================================= CVAE =============================================
