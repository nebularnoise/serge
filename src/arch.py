import torch
import torch.nn as nn
import torch.functionnal as F


class VariationnalAutoEncoder(nn.Module):
    """Defines the architecure composing a Variationnal Auto-Encoder

    Parameters
    ----------


    """
    def __init__(self):
        super(AE,self).__init__()

        self.act   = nn.LeakyReLU()
        self.tanh  = nn.Tanh()

        self.enc_dim = [1,128,256,7*7*256,1024,128]

        self.z_dim = 32

        self.enc1  = nn.Conv2d(1,self.enc_dim[1],5,padding=2,stride=2)
        self.enc1s = nn.Sequential(nn.BatchNorm2d(self.enc_dim[1]),\
                               self.act)
        self.enc2  = nn.Conv2d(self.enc_dim[1],self.enc_dim[2],
                            5,padding=2,stride=2)
        self.enc2s = nn.Sequential(nn.BatchNorm2d(self.enc_dim[2]),\
                               self.act)

        self.enc3  = nn.Linear(7*7*256,1024)
        self.enc3s = nn.Sequential(nn.BatchNorm1d(1024),\
                               self.act)

        self.enc4  = nn.Linear(1024,128)
        self.enc4s = nn.Sequential(nn.BatchNorm1d(128),\
                               self.act)

        self.logvar = nn.Linear(128,self.z_dim)

        self.mu    = nn.Linear(128, self.z_dim)

        self.dec1  = nn.Linear(self.z_dim,128)
        self.dec1s = nn.Sequential(nn.BatchNorm1d(128),\
                               self.act)

        self.dec2  = nn.Linear(128,1024)
        self.dec2s = nn.Sequential(nn.BatchNorm1d(1024),\
                               self.act)

        self.dec3  = nn.Linear(1024,7*7*256)
        self.dec3s = nn.Sequential(nn.BatchNorm1d(7*7*256),\
                               self.act)

        self.dec4  = nn.ConvTranspose2d(256,128,5,stride=2,padding=2)
        self.dec4s = nn.Sequential(nn.BatchNorm2d(128),\
                               self.act)

        self.dec5  = nn.ConvTranspose2d(128,1,5,stride=2,padding=2)
        self.dec5s = nn.Sequential(nn.BatchNorm2d(1),\
                               self.act)

        self.dec6  = nn.ConvTranspose2d(1,1,4,stride=1,padding=0)
        self.dec6s = nn.Sequential(nn.BatchNorm2d(1),\
                               nn.Sigmoid())

    def encode(self,x):
        x = x.unsqueeze(1)
        #ENCODE
        x = self.enc1s(self.enc1(x))
        x = self.enc2s(self.enc2(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.enc3s(self.enc3(x))
        x = self.enc4s(self.enc4(x))

        logvar = self.logvar(x)
        mu     = self.mu(x)

        return logvar,mu

    def sample(self,logvar,mu):
        return mu + torch.randn_like(mu)*torch.exp(.5*logvar)

    def decode(self,x):
        #DECODE
        x = self.dec1s(self.dec1(x))
        x = self.dec2s(self.dec2(x))
        x = self.dec3s(self.dec3(x))
        x = x.view(-1,256,7,7)
        x = self.dec4s(self.dec4(x))
        x = self.dec5s(self.dec5(x))
        x = self.dec6s(self.dec6(x))
        return x

    def forward(self,x):
        logvar,mu = self.encode(x)
        z         = self.sample(logvar,mu)
        y         = self.decode(z)
        return y

def num_flat_features(self,x):
        size = x.size()[1:]
        num  = 1
        for s in size:
            num *= s
        return num
