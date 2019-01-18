#author: Antoine CAILLON
import torch
import torch.nn as nn
import torch.functional as F


def train_vae(model, train_loader, device, number_of_epochs=100, learning_rate = 1e-1, lr_decay=0.2):

    train_loss_log = np.zeros(n_ep)
    test_loss_log  = np.zeros(n_ep)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    for epoch in range(n_ep):
        model.train()

        for batch_idx, minibatches in enumerate(train_loader):
            process_minibatch(model, optimizer, minibatches, train_loss_log[epoch])

        model.eval()
        with torch.no_grad():
            for batch_idx,minibatch in enumerate(test_loader):
              process_minibatch(model, optimizer, minibatches, test_loss_log[epoch])

            if (epoch)%20==0:
                print("EPOCH #{} DONE, TRAIN_ERROR {}, TEST_ERROR {}".format(epoch,\
                                                train_loss_log[epoch],\
                                                test_loss_log[epoch]))

            learning_rate *= lr_decay
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)


def compute_loss(minibatch, output, mu, logvar):
    rec_loss = F.binary_cross_entropy(output,minibatch.unsqueeze(1),\
                                   reduction="sum")
    kl_loss  = 0.25 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    loss     = rec_loss + kl_loss
    return loss

def process_minibatch(model, optimizer, minibatches, loss_log_epoch):
    minibatch = minibatches[0].to(device).squeeze(1)

    optimizer.zero_grad()

    logvar,mu = model.encode(minibatch)
    z         = model.sample(logvar,mu)
    output    = model.decode(z)

    loss     = compute_loss(minibatch, output, mu, logvar)

    loss.backward()
    optimizer.step()
    loss_log_epoch += loss.item()
