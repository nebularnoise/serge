#autor Antoine CAILLON
import argparse
import mido
import torch
import sounddevice as sd
from pyglim import pyglim as gl
import librosa as li
from scipy.signal import fftconvolve
import numpy as np
import matplotlib.pyplot as plt
from time import time
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="TorchScript to load")
parser.add_argument("--gl-iteration", type=int, default=30, help="Number of GL iterations")
parser.add_argument("--reverb", type=int, default=0, help="Reverb or not (0/1)")
args = parser.parse_args()

print("Ready...", end="", flush=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

model = torch.jit.load(args.model, map_location=device)

print("Steady...", end="", flush=True)

key = mido.get_input_names()[0]

x = torch.Tensor([0,0,0,0]).unsqueeze(0).to(device)
o = torch.zeros([1,7]).to(device)
s = torch.zeros([1,12]).to(device)

#ri,fs = li.load("ri.wav", sr=22050)
fs = 22050
window = np.hanning(2048)

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
                    S = 256*model(x,o,s).detach().cpu().numpy().T

                # plt.figure(figsize=(10,5))
                # plt.subplot(121)

                S = np.ascontiguousarray(S)

                sig = gl.griffin_lim_reconstruct_single_precision(args.gl_iteration, S, window, 3, 256)

                if args.reverb:
                    sig = fftconvolve(sig, ri)

                # plt.imshow(S, aspect="auto", cmap="Greys")
                # plt.title("Model output")
                # plt.subplot(122)
                # plt.imshow(abs(li.stft(sig)).T, aspect="auto", cmap="Greys")
                # plt.title("PyGlim output")
                # plt.show()
                #sig /= np.max(abs(sig))
                #print(int(1000*(time()-rt)))
                #print("done! playing...")
                sd.play(sig[3*fs//10:],22050)

        elif (msg.control-48<4):
            k = msg.control - 48
            v = 10*(msg.value/127-.5)
            x[:,k] = v
            print("{}             ".format(x), end="\r", flush=True)
