import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import librosa as li
from glob import glob
from os import system
import argparse

class AudioDataset(data.Dataset):
    def __init__(self, files, process, slice_size):
        self.slice_size = slice_size
        self.division = 128 // slice_size
        self.liste = glob(files)
        if process:
            print("Preprocessing audio dataset... ")
            print("                      ]", end="\r")
            print("[", end="")


            list_freq = 27.5*(2**(np.arange(88)/12))

            get_idx = lambda f: int(np.argmin(abs(list_freq - f)))

            for i,elm in enumerate(self.liste):
                if i%(len(self.liste)//20)==0:
                    print("=", end="", flush=True)

                # LOADING AUDIO
                [x,fs] = li.load(elm, sr=22050)
                x = self.pad(x,34560)
                mel = li.filters.mel(fs,2048,500)

                # AMPLITUDE MEL-SOECTROGRAM
                S = mel.dot(abs(li.stft(x,n_fft=2048,
                win_length=2048, hop_length=256, center=False)))
                S[S<1e-3] = 0
                S = np.log(1 + S)
                S = torch.from_numpy(2 * S / np.max(S) - 1).float()

                # fundamental frequency estimation by autocorrelation method
                xx = np.correlate(x,x, mode="full")[len(x):]
                fmin = 100
                fmax = 2000

                tmin = fs//fmax
                tmax = fs//fmin

                f0 = fs/(np.argmax(xx[tmin:tmax])+tmin)

                idx = get_idx(f0)

                oct = torch.zeros([7])
                semitone = torch.zeros([12])

                oct[idx//12] = 1
                semitone[idx%12] = 1

                torch.save((S, oct, semitone),elm.replace(".wav",".pt"))

            print("]\nDone!")

    def __getitem__(self,i):
        stft,o,s = torch.load(self.liste[i//self.division].replace(".wav",".pt"))
        slice = stft[:, (i%self.division)*self.slice_size:(i%self.division+1)*self.slice_size]
        return slice,o,s

    def __len__(self):
        return len(self.liste)*self.division

    def pad(self,x,n):
        m = len(x)
        if m<n:
            return np.concatenate([x,np.zeros(n-m)])
        else:
            return x[:n]

    def shift(self,S,f0,fs,n_bin):
        mel        = np.linspace(li.core.hz_to_mel(0),li.core.hz_to_mel(fs/2), n_bin)
        freq       = li.core.mel_to_hz(mel)
        bin_shift  = -int(np.argmin(abs(freq-f0)) - 100)

        S = torch.roll(S, bin_shift, 0)

        S[n_bin+bin_shift:,:] = 0

        return S


class WAE(nn.Module):
    def __init__(self, zdim, n_trames):
        super(WAE,self).__init__()
        size = [1, 16, 32, 64, 128, 256]

        self.flat_number = int(256*16*n_trames/32)
        self.n_trames = n_trames
        #print(self.flat_number)

        act = nn.LeakyReLU()

        self.act = act

        enc1 = nn.Conv2d(size[0],size[1],stride=2, kernel_size=5, padding=2)
        enc2 = nn.Conv2d(size[1],size[2],stride=2, kernel_size=5, padding=2)
        enc3 = nn.Conv2d(size[2],size[3],stride=2, kernel_size=5, padding=2)
        enc4 = nn.Conv2d(size[3],size[4],stride=2, kernel_size=5, padding=2)
        enc5 = nn.Conv2d(size[4],size[5],stride=2, kernel_size=5, padding=2)

        lin1 = nn.Linear(self.flat_number, 1024)
        lin2 = nn.Linear(1024, 256)
        lin3 = nn.Linear(256, zdim)


        dlin1 = nn.Linear(zdim+7+12,256)
        dlin2 = nn.Linear(256,1024)
        dlin3 = nn.Linear(1024, self.flat_number)

        self.dconv1 = nn.Conv2d(256, 128, stride=1, padding=2, kernel_size=5)
        self.dconv2 = nn.Conv2d(128, 64,  stride=1, padding=2, kernel_size=5)
        self.dconv3 = nn.Conv2d(64, 32,  stride=1, padding=3, kernel_size=7)
        self.dconv4 = nn.Conv2d(32, 4,  stride=1, padding=4, kernel_size=9)

        self.dconv_logvar = nn.Conv2d(4, 1, stride=1, padding=3, kernel_size=7)
        self.dconv_mean   = nn.Conv2d(4, 1, stride=1, padding=3, kernel_size=7)

        self.e1   = nn.Sequential(enc1,
                                nn.BatchNorm2d(num_features=size[1]),act,
                                enc2,
                                nn.BatchNorm2d(num_features=size[2]),act,
                                enc3,
                                nn.BatchNorm2d(num_features=size[3]),act,
                                enc4,
                                nn.BatchNorm2d(num_features=size[4]),act,
                                enc5,
                                nn.BatchNorm2d(num_features=size[5]),act)

        self.e2   = nn.Sequential(lin1,
                                 nn.BatchNorm1d(num_features=1024),act,
                                 lin2,
                                 nn.BatchNorm1d(num_features=256),act,
                                 lin3)

        self.d1   = nn.Sequential(dlin1,
                                 act,
                                 dlin2,
                                 act,
                                 dlin3,
                                 act)



    def flatten(self, inp):
        dim = 1
        for i,elm in enumerate(inp.size()):
            if i!=0:
                dim *= elm
        return inp.view(-1,dim)

    def encode(self, inp):
        inp = inp.unsqueeze(1)
        inp = self.e1(inp)
        #print(inp.size())
        inp = self.flatten(inp)
        #print(inp.size())
        inp = self.e2(inp)
        return inp

    def decode(self, inp, oct, semitone):
        inp = torch.cat([inp, oct, semitone], 1)
        #print(inp.size())
        inp = self.d1(inp).view(-1,256,16,4)
        #print(inp.size())
        z_2 = self.act(self.dconv1(F.upsample(inp, scale_factor=3)))
        z_3 = self.act(self.dconv2(F.upsample(z_2, scale_factor=3)))
        z_4 = self.act(self.dconv3(F.upsample(z_3, scale_factor=3)))
        z_5 = self.act(self.dconv4(F.upsample(z_4, size=[500,128])))

        logvar = self.dconv_logvar(z_5).squeeze(1)
        mean   = torch.tanh(self.dconv_mean(z_5)).squeeze(1)


        return mean, logvar

    def sample(self, inp, oct, semitone):
        mel = torch.from_numpy(li.filters.mel(22050, 2048, n_mels=500)).float()
        mel_specto, logvar = self.decode(inp, f)
        return torch.transpose(mel, 0, 1).mm(mel_specto.squeeze(0))

    def forward(self,inp, oct, semitone):
        return self.decode(self.encode(inp), oct, semitone)

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

def train(model, GCloader, epoch, savefig=False, lr_rate=3, nb_update=10):
    model.train()
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #loss = torch.nn.modules.BCELoss()
    for e in range(epoch):
        for idx, (minibatch,octave,semitone) in enumerate(GCloader):



            minibatch = minibatch.to(device)
            octave    = octave.to(device)
            semitone  = semitone.to(device)

            optimizer.zero_grad()

            z = model.encode(minibatch)

            rec, logvar = model.decode(z,octave, semitone)

            nll = (logvar+(minibatch - rec).pow(2).div(torch.exp(logvar).add(1e-7))+np.log(2*np.pi))
            loss = torch.sum(torch.sum(torch.sum(0.5*nll,2),1))

            error = loss + .2*compute_mmd(z,torch.randn_like(z))

            error.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        print("     epoch {} done".format(e), end="\r")
        if e%(epoch//nb_update)==0:
            torch.save(model, "output/model_%d_epoch.pt" % e)
            print("EPOCH %d, ERROR %f" % (e,error))
            if savefig:
                show_me_how_good_model_is_learning(model, GC, 4)
                plt.savefig("output/epoch_%d.png"%e)

        if (e+1)%(epoch//lr_rate)==0:
            lr /= 2.
            optimizer = torch.optim.Adam(model.parameters(),lr=lr)

def show_me_how_good_model_is_learning(model, GC, n):
    model.eval()
    N = len(GC)

    dataset = data.DataLoader(GC, shuffle=True, batch_size=n)

    with torch.no_grad():
        plt.figure(figsize=(20,25))

        spectrogram, oct, semitone = next(iter(dataset))

        oct = oct.to(device)
        semitone = semitone.to(device)
        spectrogram = spectrogram.to(device)

        rec, logvar = model(spectrogram, oct, semitone)
        rec = rec.cpu().numpy()
        spectrogram = spectrogram.cpu().numpy()

    for i in range(n):
        plt.subplot(n,2,2*i + 1)
        plt.imshow(spectrogram[i,:,:], origin="lower", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Original")

        plt.subplot(n,2,2*i+ 2)
        plt.imshow(rec[i,:,:], origin="lower", aspect="auto", vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Reconstruction")
    model.train()


if __name__=="__main__":
    system("mkdir output")

    parser = argparse.ArgumentParser(description="Final model training")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--dataset", type=str, default=None, help="Location of dataset")
    parser.add_argument("--lr-step", type=int, default=3, help="Number of division of lr over epoch")
    parser.add_argument("--zdim", type=int, default=32, help="Dimension of latent space")
    parser.add_argument("--n-trames", type=int, default="128", help="Processes n_trames. Must be a power of 2 >= 32")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device to be used")
    parser.add_argument("--nb-update", type=int, default=10, help="Number of update / backup to do")
    parser.add_argument("--process-dataset", type=int, default=0, help="1/0 if Preprocessing needed")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    args = parser.parse_args()

    GC = AudioDataset(files="%s/*.wav" % args.dataset, process=args.process_dataset, slice_size=args.n_trames)
    GCloader = data.DataLoader(GC, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    torch.cuda.empty_cache()

    model = WAE(args.zdim, args.n_trames).to(device)

    train(model, GCloader, args.epoch, savefig=True, lr_rate=args.lr_step,
        nb_update=args.nb_update)
    # show_me_how_good_model_is_learning(model, GC, 4)
    # plt.show()
    torch.save(model, "model_%d_epoch.pt"%args.epoch)
