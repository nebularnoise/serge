import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from sys import stdout

def determinist_forward_pass(model,minibatch):
    z = model.encode(minibatch)
    reconstruction = model.decode(z)
    return z,reconstruction

def stochastic_forward_pass(model,minibatch):
    logvar, mu = model.encode(minibatch)
    z          = model.sample(logvar,mu)
    reconstruction = model.decode(z)
    return logvar,mu,z,reconstruction

def Vloss(logvar,mu,output,minibatch,rec_loss_f):
    kl_loss  = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    #rec_loss = rec_loss_f(output, minibatch.unsqueeze(1),reduction="sum")
    rec_loss = rec_loss_f(output, minibatch.unsqueeze(1), size_average=False)
    loss = kl_loss + rec_loss
    return loss

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def Wloss(z, output, minibatch, rec_loss_f):
    return compute_mmd(z,torch.randn_like(z)) +\
     rec_loss_f(output, minibatch.unsqueeze(1), size_average=False)

def train_model(model, device, train_loader, test_loader, epoch,
                rec_loss_f, name, mode="V", lr=1e-3):
    statut = "     EPOCH %d, TRAIN_LOSS: %f, TEST_LOSS: %f "
    train_loss_log = np.zeros(epoch)
    test_loss_log  = np.zeros(epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_size = len(train_loader)
    for e in range(epoch):
        model.train()

        for batch_idx, minibatch in enumerate(train_loader):

            print("Training on minibatch %d...        " % (batch_idx), end="\r")

            minibatch = minibatch[0].to(device).squeeze(1)
            optimizer.zero_grad()

            # FORWARD PASS
            if model.determinist_encoder:
                z,rec = determinist_forward_pass(model,minibatch)
            else:
                logvar,mu,z,rec = stochastic_forward_pass(model,minibatch)

            # LOSS PASS
            if mode=="V":
                loss = Vloss(logvar,mu,rec,minibatch,rec_loss_f)
            elif mode=="W":
                loss = Wloss(z,rec,minibatch,rec_loss_f)

            # BACKWARD PASS
            train_loss_log[e] += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(test_loader):
                print("Testing on minibatch %d...     " % (batch_idx), end="\r")
                minibatch = minibatch[0].to(device).squeeze(1)
                optimizer.zero_grad()

                # FORWARD PASS
                if model.determinist_encoder:
                    z,rec = determinist_forward_pass(model,minibatch)
                else:
                    logvar,mu,z,rec = stochastic_forward_pass(model,minibatch)

                # LOSS PASS
                if mode=="V":
                    loss = Vloss(logvar,mu,rec,minibatch,rec_loss_f)
                elif mode=="W":
                    loss = Wloss(z,rec,minibatch,rec_loss_f)

                # LOG

                test_loss_log[e] += loss.item()

        print(statut % (e,train_loss_log[e],test_loss_log[e]))

        np.save("log",{"train":train_loss_log,"test":test_loss_log})

        if ((e+1)%3==0):
            if (test_loss_log[e] < np.min(test_loss_log[:e])):
                torch.save(model,'model_{}.torch'.format(name))

        if ((e+1)%20==0):
            lr /= 2
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
