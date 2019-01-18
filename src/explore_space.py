import argparse
import mido
import torch
import sounddevice as sd
from util import audio_utilities as au
import librosa as li
from scipy.signal import fftconvolve
import numpy as np
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="TorchScript to load")
parser.add_argument("--gl-iteration", type=int, default=30, help="Number of GL iterations")
parser.add_argument("--reverb", type=int, default=0, help="Reverb or not (0/1)")
args = parser.parse_args()

print("Ready...", end="", flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = torch.jit.load(args.model, map_location=device)

print("Steady...", end="", flush=True)

key = mido.get_input_names()[0]

x = torch.Tensor([0,0,0,0]).unsqueeze(0).to(device)
o = torch.zeros([1,7]).to(device)
s = torch.zeros([1,12]).to(device)

#ri,fs = li.load("ri.wav", sr=22050)
fs = 22050

with mido.open_input(key) as inport:
    print("Go!")
    for msg in inport:
        if msg.channel:
            if msg.velocity:
                if not msg.note:
                    break
                # create and inverse spectrogram
                s *= 0
                o *= 0

                idx = msg.note
                o[:, idx//12] = 1
                s[:, idx%12]  = 1

                #print("audio generation...", end="")
                #rt = time()
                with torch.no_grad():
                    S = model(x,o,s).detach().cpu().numpy().T
                #print(int(1000*(time()-rt)))
                #rt = time()
                sig = au.reconstruct_signal_griffin_lim(S, 2048, 256, args.gl_iteration, verbose=False)
                if args.reverb:
                    sig = fftconvolve(sig, ri)

                sig /= np.max(abs(sig))
                #print(int(1000*(time()-rt)))
                #print("done! playing...")
                sd.play(sig[3*fs//10:],22050)
        elif (msg.control-48<4):
            k = msg.control - 48
            v = 10*(msg.value/127-.5)
            x[:,k] = v
            print("{}             ".format(x), end="\r", flush=True)
