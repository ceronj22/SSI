#Nick Cerone
#7.20.21

from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as fn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# ======================================== SETUP AND VARS ========================================

# Load MNIST
mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

# VARIABLES, WEIGHTS, AND FUNCTIONS

# initialize some variables variables

batch_size = 48
input_dim = 28 * 28
hidden_dim = 128
state_dim = 10
z_dim = 2

app_folder = "/Users/ceronj22/Desktop/"
name = "weights"


# Xavier Initialization - initialize weights with smaller numbers (less variance) based on the batch size to prevent gradient from exploding
def xav_init(bs, out_dim):
    xav_std = 1.0 / np.sqrt(bs / 2.0)

    return Parameter(torch.randn(bs, out_dim) * xav_std, requires_grad=True)


# ================== Storage ==================

# original images
origs = []

# reconstructed images
recons = []

# labels
labels = []

# save the losses to plot later
losses = []

# save latent variables to plot later
latents = []

# Means
mus = []

# Stds
sigs = []


# ================== Storage ==================

# ======================================== SETUP AND VARS ========================================


# ============================================= CVAE =============================================

class cVAE(nn.Module):
    weights = []
    biases = []

    # =============== Constructor ================

    # initializes random weights and 0 biases
    def __init__(self):
        super().__init__()

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


    # ================== Encoder ==================

    # the encoder takes in both some input and the state s
    def Q(self, X, s):
        # concatonate the two tensors over a given dimension
        X_prime = torch.cat([X, s], 1)

        # @ is matrix multiplication (dot product) - can only be done if col len of first matrix is the same as row len
        # of 2nd (if m1 has size n x p and n2 has size p x m, then the output matrix will have size n x m)

        # grab a hidden layer and relu it
        hidden = torch.relu(X_prime @ self.weights_xh + self.biases_xh.repeat(X_prime.size(0), 1))

        # each row in biases should be the same, so to make it a 2D matrix we just copy it vertically batch_size times
        # we can't just initialize biases as 2D because when backpropping, different rows would get different bias vals

        # grab tensors containing our mean and variance of the input data by reducing the size and taking the dot product
        Zmu = hidden @ self.weights_hZmu + self.biases_hZmu.repeat(hidden.size(0), 1)
        Zsig = hidden @ self.weights_hZsig + self.biases_hZsig.repeat(hidden.size(0), 1)

        mus.append(Zmu)
        sigs.append(Zsig)

        return Zmu, Zsig

    # ================== Encoder ==================

    # ================== Decoder ==================

    # samples a point from the normal distribution N(0,1) and augments it based on latent information to give differentiable values
    def reparameterize(self, mu, log_sig):
        # Random sampling - randn returns a tensor with random numbers sampled from a Normal Distribution N(0,1)
        epsilon = Variable(torch.randn(batch_size, z_dim))

        # adding the mean we pulled from the encoder and multiplying by standard deviation gives a differentiable sample which relates to our Z
        # want to take the sqrt of sig, but that's a very computationally heavy operation. instead, we take the log (we now assume the encoder
        # will output log(sigma)), divide by 2 (log rules), and then raise to the e
        sample = mu + torch.exp(log_sig / 2) * epsilon

        latents.append(sample)  # save to our latent array to plot later

        return sample

    # Decoder function
    def P(self, z, s):
        # concatonate over the labels
        z_prime = torch.cat([z, s], 1)

        # do more dot products to reconstruct the matrix now
        p_hidden = torch.relu(z_prime @ self.weights_zh + self.biases_zh.repeat(z_prime.size(0), 1))

        X_hat = torch.sigmoid(p_hidden @ self.weights_hxhat + self.biases_hxhat.repeat(p_hidden.size(0), 1))

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

    # ================== Forward ==================

    # run one step of the encoder
    def forward(self, X, s):
        Zmu, Zsig = self.Q(X, s)
        z = self.reparameterize(Zmu, Zsig)
        X_hat = self.P(z, s)

        loss = self.calc_tot_loss(X, X_hat, Zmu, Zsig)

        return X_hat, loss

    # ================== Forward ==================


# ============================================= CVAE =============================================


# =================== Other ===================

# converts mnist input data to tensor to pass through nn
def num_to_inp(n):
    return Variable(torch.zeros(1, input_dim), requires_grad=True) + n.reshape(1, input_dim)


# converts ouput tensor to mnist data
def out_to_num(o):
    return o.detach().numpy().reshape(28, 28)


# makes a batch of onehot tensors given a batch of labels
def to_one_hot(t):
    to_ret = torch.empty(batch_size, 10)

    for i in range(len(t)):
        l = t[i]

        one_hot = int(l[0])
        l = torch.zeros(10)

        l[one_hot] = 1

        to_ret[i] = l

    return to_ret


# makes a one hot tensor based on input numbers - for image generation
def num_to_one_hot(n=None):
    l = torch.zeros(10)

    # if you did not define a number...
    if n == None:
        return l
    elif isinstance(n, int):  # if you set n to be a number
        l[n] = 1
    else:  # this assumes n is a list-like, and thus will iterate through n and set any value in n to be 1 in the onehot
        for i in n:
            l[i] = 1

    return l


# =================== Other ===================


# Create a new cVAE object
net = cVAE()


# =========================================== TRAINING ===========================================
def train():
    # make load the data randomly into an interator
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)
    data_iterator = iter(data_loader)

    # define parameters for our torch optimizer
    op_parameters = net.parameters()

    learn_rate = 0.0001
    optimizer = optim.Adam(op_parameters, lr=learn_rate)

    # we can print out the total number of batches if we check the current length of arrays we've already saved values to
    prev_trained = 0 #len(origs)

    #number of times we want it to iterate
    batches = 1250 #1250 batches = 1 Epoch (1250*48 = 60,000)

    # this would all get looped quite a bit
    for i in range(batches):

        # make sure we don't run out of images
        #if i * batch_size >= 60000 - batch_size:
        #    data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=batch_size, shuffle=True)
        #    data_iterator = iter(data_loader)

        # pull input images
        images, labs = data_iterator.next()  # pulls the next batch of data from the dataset (mnist)

        X = images.reshape(batch_size,
                           input_dim)  # making images a tensor of dimension (batch_size, input_dim) (batch_size, 28*28)
        s = to_one_hot(labs.reshape(batch_size, 1))  # reshape the labels
        # print(s[0])

        # run encoder and decoder using X and s and generate an X_hat
        X_hat, loss = net.forward(X, s)

        origs.append(X)
        recons.append(X_hat)
        labels.append(labs)

        print("Batch: {}\tLoss: {}".format(i + 1 + prev_trained, loss))
        losses.append(float(loss))  # append it as a float so that we can put into matplotlib later

        # back prop weights
        loss.backward()
        optimizer.step()

        # Housekeeping
        for p in op_parameters:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


# =========================================== TRAINING ===========================================












best_loss = 9999999 #safe bet for a really high starting loss
curr_epoch = 0

if os.path.exists(str(app_folder + name)):
    #net.load_state_dict(torch.load(str(app_folder + name)))

    #load the old checkpoint
    checkpoint = torch.load(str(app_folder + name))

    net.load_state_dict(checkpoint['model_state_dict'])
    curr_epoch = checkpoint['epoch'] + 1 #the previous epoch number is saved, so we have to increment
    best_loss = checkpoint['loss']

    print("Loading checkpoint from file...")
else:
    print("No checkpoint file detected.")


#how many times do we want to iterate through the whole dataset
num_epochs = 5

#train over many epochs and save the best models
for i in range(curr_epoch, num_epochs):
    print("CURR EPOCH: {} -=- PREV BEST LOSS: {}".format(i, best_loss))

    #train another epoch
    train()

    #if the result of this training is that the model is better than it was before, save the state
    if losses[-1] < best_loss:
        print("Saving checkpoint...")
        #torch.save(net.state_dict(), str(app_folder + name))

        #save a checkpoit
        torch.save({
            'model_state_dict': net.state_dict(),
            'epoch': i,
            'loss': losses[-1]
        }, str(app_folder + name))

        print("Saved!")

        best_loss = losses[-1]
    else:
        print("Model not saved; Loss of {} greater than Best Loss {}".format(losses[-1], best_loss))

#check with validation set
