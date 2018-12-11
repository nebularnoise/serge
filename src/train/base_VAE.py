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
    rec_loss = rec_loss_f(output, minibatch.unsqueeze(1))
    loss = kl_loss + rec_loss
    return loss

def Wloss():
    pass

def train_model(model, device, train_loader, test_loader, epoch,
                rec_loss_f, mode="V", lr=1e-3):
    train_loss_log = np.zeros(epoch)
    test_loss_log  = np.zeros(epoch)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_size = len(train_loader)
    for e in range(epoch):
        model.train()

        for batch_idx, minibatch in enumerate(train_loader):

            print("     Working on minibatch %d..." % (batch_idx), end="\r")

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
                pass

            # BACKWARD PASS
            train_loss_log[e] += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch_idx, minibatch in enumerate(train_loader):
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
                    pass

                # LOG

                test_loss_log[e] += loss.item()
