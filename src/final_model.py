import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import librosa as li
from glob import glob

class AudioDataset(data.Dataset):
    def __init__(self, files, process):
        self.liste = glob(files)
        if process:
            print("Preprocessing stuff... ", end="")
            for elm in self.liste:
                [x,fs] = li.load(elm, sr=22050)
                x = self.pad(x,34560)
                mel = li.filters.mel(fs,2048,500)
                S = torch.from_numpy(mel.dot(abs(li.stft(x,n_fft=2048,win_length=2048, hop_length=256, center=False)))).float()
                S = S/torch.max(S)
                torch.save(S,elm.replace(".wav",".pt"))
            print("Done!")

    def __getitem__(self,i):
        return torch.load(self.liste[i].replace(".wav",".pt"))

    def __len__(self):
        return len(self.liste)

    def pad(self,x,n):
        m = len(x)
        if m<n:
            return np.concatenate([x,np.zeros(n-m)])
        else:
            return x[:n]

class WAE(nn.Module):
    def __init__(self):
        super(WAE,self).__init__()
        size = [1, 16, 32, 64, 128, 256]
        zdim = 32

        self.act = nn.LeakyReLU()

        self.enc1 = nn.Conv2d(size[0],size[1],stride=2, kernel_size=5, padding=2)
        self.enc2 = nn.Conv2d(size[1],size[2],stride=2, kernel_size=5, padding=2)
        self.enc3 = nn.Conv2d(size[2],size[3],stride=2, kernel_size=5, padding=2)
        self.enc4 = nn.Conv2d(size[3],size[4],stride=2, kernel_size=5, padding=2)
        self.enc5 = nn.Conv2d(size[4],size[5],stride=2, kernel_size=5, padding=2)

        self.lin1 = nn.Linear(256*16*4, 1024)
        self.lin2 = nn.Linear(1024, 256)
        self.lin3 = nn.Linear(256, zdim)

        self.decf = nn.Conv2d(1,1,stride=1, kernel_size=3, padding=1)
        self.dec0 = nn.ConvTranspose2d(size[0],size[0],stride=2, kernel_size=5, padding=2)
        self.dec1 = nn.ConvTranspose2d(size[1],size[0],stride=2, kernel_size=5, padding=2)
        self.dec2 = nn.ConvTranspose2d(size[2],size[1],stride=2, kernel_size=5, padding=2)
        self.dec3 = nn.ConvTranspose2d(size[3],size[2],stride=2, kernel_size=5, padding=2)
        self.dec4 = nn.ConvTranspose2d(size[4],size[3],stride=2, kernel_size=5, padding=2)
        self.dec5 = nn.ConvTranspose2d(size[5],size[4],stride=2, kernel_size=5, padding=2)

        self.dlin1 = nn.Linear(1024,256*16*4)
        self.dlin2 = nn.Linear(256,1024)
        self.dlin3 = nn.Linear(zdim,256)

        self.f1   = nn.Sequential(self.enc1,
                                nn.BatchNorm2d(num_features=size[1]),self.act,
                                self.enc2,
                                nn.BatchNorm2d(num_features=size[2]),self.act,
                                self.enc3,
                                nn.BatchNorm2d(num_features=size[3]),self.act,
                                self.enc4,
                                nn.BatchNorm2d(num_features=size[4]),self.act,
                                self.enc5,
                                nn.BatchNorm2d(num_features=size[5]),self.act)

        self.f2   = nn.Sequential(self.lin1,
                                 nn.BatchNorm1d(num_features=1024),self.act,
                                 self.lin2,
                                 nn.BatchNorm1d(num_features=256),self.act,
                                 self.lin3)

        self.f3   = nn.Sequential(self.dlin3,
                                  nn.BatchNorm1d(num_features=256), self.act,
                                  self.dlin2,
                                  nn.BatchNorm1d(num_features=1024), self.act,
                                  self.dlin1,
                                  nn.BatchNorm1d(num_features=256*16*4), self.act)

        self.f4   = nn.Sequential(self.dec5,
                                 nn.BatchNorm2d(num_features=size[4]), self.act,
                                 self.dec4,
                                 nn.BatchNorm2d(num_features=size[3]), self.act,
                                 self.dec3,
                                 nn.BatchNorm2d(num_features=size[2]), self.act,
                                 self.dec2,
                                 nn.BatchNorm2d(num_features=size[1]), self.act,
                                 self.dec1,
                                 nn.BatchNorm2d(num_features=size[0]), self.act,
                                 self.dec0, nn.Sigmoid())




    def flatten(self, inp):
        dim = 1
        for i,elm in enumerate(inp.size()):
            if i!=0:
                dim *= elm
        return inp.view(-1,dim)

    def encode(self, inp):
        inp = inp.unsqueeze(1)
        inp = self.f1(inp)
        #print(inp.size())
        inp = self.flatten(inp)
        inp = self.f2(inp)
        return inp

    def decode(self, inp):
        #print(inp.size())
        inp = self.f3(inp)
        inp = inp.view(-1, 256, 16, 4)
        inp = self.f4(inp)
        inp = nn.functional.interpolate(inp, size=(500,128))
        inp = self.decf(inp)
        inp = torch.sigmoid(inp)
        return inp.squeeze(1)


    def forward(self,inp):
        return self.decode(self.encode(inp))

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

def train():
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    epoch = 1000
    loss = torch.nn.modules.BCELoss()
    for e in range(epoch):
        for idx, minibatch in enumerate(GCloader):

            minibatch = minibatch.to(device)

            optimizer.zero_grad()

            z = model.encode(minibatch)

            rec = model.decode(z)

            error = loss(rec,minibatch) + compute_mmd(z,torch.randn_like(z))

            error.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        if e%(epoch//10)==0:
            print("EPOCH %d, ERROR %f" % (e,error))

        if (e+1)%(epoch//3)==0:
            lr /= 2.
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)

if __name__=="__main__":
    GC = GuitarChord(files="ukulele_sample_pack/*.wav", process=True)
    GCloader = data.DataLoader(GC, batch_size=8, shuffle=True, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    torch.cuda.empty_cache()

    #model = WAE().to(device)
    model = torch.load("model_1000_epoch.pt")

    #
