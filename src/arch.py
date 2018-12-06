import torch
import torch.nn as nn
import torch.functional as F


class VariationnalAutoEncoder(nn.Module):
    def __init__(self, determinist_encoder=False):
        """Defines the inner layers of a variationnal auto encoderself.

        Parameters
        ----------


        """
        super(VariationnalAutoEncoder,self).__init__()

        self.determinist_encoder = determinist_encoder

        self.act   = nn.LeakyReLU()

        self.enc_dim = [1,128,256,7*7*256,1024,128]
        self.dec_dim = [128,1024,7*7*256,256,128,1,1]

        self.z_dim = 32

        #-----------------DEFINITION OF THE ENCODER-----------------------------

        self.enc1  = nn.Conv2d(1,self.enc_dim[1],5,padding=2,stride=2)
        self.enc1s = nn.Sequential(nn.BatchNorm2d(self.enc_dim[1]),\
                               self.act)
        self.enc2  = nn.Conv2d(self.enc_dim[1],self.enc_dim[2],
                            5,padding=2,stride=2)
        self.enc2s = nn.Sequential(nn.BatchNorm2d(self.enc_dim[2]),\
                               self.act)

        self.enc3  = nn.Linear(self.enc_dim[3],self.enc_dim[4])
        self.enc3s = nn.Sequential(nn.BatchNorm1d(self.enc_dim[4]),\
                               self.act)

        self.enc4  = nn.Linear(self.enc_dim[4],self.enc_dim[5])
        self.enc4s = nn.Sequential(nn.BatchNorm1d(self.enc_dim[5]),\
                               self.act)

        if self.determinist_encoder:
            self.enc5 = nn.Linear(self.enc_dim[5],z_dim)
        else:
            self.logvar = nn.Linear(self.enc_dim[5],self.z_dim)
            self.mu    = nn.Linear(self.enc_dim[5], self.z_dim)

        #-----------------DEFINITION OF THE DECODER-----------------------------
        
        self.dec1  = nn.Linear(self.z_dim,self.dec_dim[0])
        self.dec1s = nn.Sequential(nn.BatchNorm1d(self.dec_dim[0]),\
                               self.act)

        self.dec2  = nn.Linear(self.dec_dim[0],self.dec_dim[1])
        self.dec2s = nn.Sequential(nn.BatchNorm1d(self.dec_dim[1]),\
                               self.act)

        self.dec3  = nn.Linear(self.dec_dim[1],self.dec_dim[2])
        self.dec3s = nn.Sequential(nn.BatchNorm1d(self.dec_dim[2]),\
                               self.act)

        self.dec4  = nn.ConvTranspose2d(self.dec_dim[3],self.dec_dim[4]
                                        ,5,stride=2,padding=2)
        self.dec4s = nn.Sequential(nn.BatchNorm2d(self.dec_dim[4]),\
                               self.act)

        self.dec5  = nn.ConvTranspose2d(self.dec_dim[4],self.dec_dim[5],
                                        5,stride=2,padding=2)
        self.dec5s = nn.Sequential(nn.BatchNorm2d(1),\
                               self.act)

        self.dec6  = nn.ConvTranspose2d(self.dec_dim[5],self.dec_dim[6],
                                        4,stride=1,padding=0)
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
        if self.determinist_encoder:
            z = self.enc_5(x)
            return z
        else:
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

if __name__ == "__main__":
    model = VariationnalAutoEncoder()
