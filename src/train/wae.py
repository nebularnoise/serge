#author: Antoine CAILLON
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


def train_wae(model, train_loader, device, number_of_epochs=300, learning_rate = 1e-3, lr_decay=0.2, lambda_factor=1):

    train_loss_log = np.zeros(n_ep)
    test_loss_log  = np.zeros(n_ep)
    prior_loss_log = np.zeros(n_ep)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(number_of_epochs):
        model.train()

        for batch_idx, minibatches in enumerate(train_loader):
            process_minibatch(model, optimizer, minibatches, train_loss_log[epoch], prior_loss_log[epoch])

        model.eval()
        with torch.no_grad():
            for batch_idx,minibatch in enumerate(test_loader):
              process_minibatch(model, optimizer, minibatches, test_loss_log[epoch])

            if (epoch)%20==0:
                print("EPOCH #{} DONE, TRAIN_ERROR {}, TEST_ERROR {}".format(epoch,\
                                                train_loss_log[epoch],\
                                                test_loss_log[epoch]))

            if test_loss_log[epoch] < test_loss_log[epoch-1]:
                torch.save(model,'model.torch')

            learning_rate *= lr_decay
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)




def compute_losses(output, minibatch, lambda_factor=1):
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

    def W_loss(z):
      return compute_mmd(z,torch.randn_like(z))

    rec_loss = F.binary_cross_entropy(output,minibatch.unsqueeze(1))
    prior_loss = lambda_factor*W_loss(z)
    return (rec_loss, prior_loss)

def process_minibatch(model, optimizer, minibatches, loss_log_epoch, prior_loss_log="MISSING"):
    minibatch = minibatches[0].to(device).squeeze(1)

    optimizer.zero_grad()

    logvar,mu = model.encode(minibatch)
    z         = model.sample(logvar,mu)

    output    = model.decode(z)

    (rec_loss, prior_loss) = compute_losses(output, minibatch, lambda_factor)

    loss     = rec_loss + prior_loss

    train_loss_log += loss.item()
    if prior_loss_log is not "MISSING":
        prior_loss_log += prior_loss.item()

    loss.backward()
    optimizer.step()
