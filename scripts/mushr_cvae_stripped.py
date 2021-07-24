#Nick Cerone
#7.21.21

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as fn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

# ======================================== SETUP AND VARS ========================================

#convert each action param to fall within range [0,1] to pass through BCE
def action_to_cvae(action):
  # Input range:
  # Column 0: Velocity, [-3250, 3250]
  # Column 1: Steering Angle, [.15, .85]
  ret = np.zeros(action.shape)
  ret[0] = (action[0] + 3250.0) / 6500.0
  ret[1] = (action[1] - 0.15) / 0.7
  return ret


#convert each cVAE output on range [0,1] to interpretable actions
def cvae_to_action(cvae):
  #make it 1D
  cvae = cvae.detach().numpy().flatten()

  ret = np.zeros(cvae.shape)
  ret[0] = (cvae[0] * 6500.0) - 3250.0
  ret[1] = (cvae[1] * 0.7) + 0.15
  return ret



#initialize some variables
batch_size = 64
input_dim = 2 #Velocity, Wheel Angle
hidden_dim = 4
state_dim = 3 #X Driver, Y Driver, Z Orien Driver
z_dim = 2



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
        self.weights_xh = xav_init(input_dim + state_dim, hidden_dim)
        self.weights_hZmu = xav_init(hidden_dim, z_dim)
        self.weights_hZsig = xav_init(hidden_dim, z_dim)

        self.biases_xh = Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.biases_hZmu = Parameter(torch.zeros(z_dim), requires_grad=True)
        self.biases_hZsig = Parameter(torch.zeros(z_dim), requires_grad=True)


        #Decoder
        self.weights_zh = xav_init(z_dim + state_dim, hidden_dim)
        self.weights_hxhat = xav_init(hidden_dim, input_dim)

        self.biases_zh = Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.biases_hxhat = Parameter(torch.zeros(input_dim), requires_grad=True)

    # =============== Constructor ================


   

    # ================== Decoder ==================

    # Decoder function
    def P(self, z, s):
        # concatonate over the labels
        z_prime = torch.cat([z, s], 1)

        # do more dot products to reconstruct the matrix now
        p_hidden = torch.relu(torch.matmul(z_prime, self.weights_zh) + self.biases_zh.repeat(z_prime.size(0), 1))

        X_hat = torch.sigmoid(torch.matmul(p_hidden, self.weights_hxhat) + self.biases_hxhat.repeat(p_hidden.size(0), 1))

        return X_hat

    # ================== Decoder ==================

    # ============== Loss Functions ===============

    # find KL divergence with mean and standard deviation of latent space
    def kl_div(self, z_mu, z_sig):
        # these equations both give the same thing, but kl_2 has less going on

        # kl_1 = -0.5 * torch.sum(1 + torch.log(z_sig ** 2) - torch.square(z_mu) - torch.exp(torch.log(z_sig ** 2)), axis=1)
        # kl_2 = -0.5 * torch.sum((1 + tgorch.lo(z_sig**2) - z_mu**2 - z_sig**2), axis=1)
        kl_3 = 0.5 * torch.sum(torch.exp(z_sig) + z_mu ** 2 - 1. - z_sig, 1)
        return torch.mean(kl_3, axis=0)

    # calculate the total loss - want to minimize
    def calc_tot_loss(self, orig, recon, z_mu, z_sig):
        # MSE = nn.MSELoss()(recon, orig) #extra loss function to try

        # binary cross entropy loss instead of MSE
        BCE = fn.binary_cross_entropy(recon, orig, size_average=False) / batch_size

        # KL Divergece
        KL = self.kl_div(z_mu, z_sig)

        # hyperparameter lambda
        lam = 1

        return BCE + lam * KL

    # ============== Loss Functions ===============



# ============================================= CVAE =============================================

