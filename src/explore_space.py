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
from glob import glob

class RI(object):
    def __init__(self):
        self.ri_list = glob("ri/*.wav")
        self.ri_files = []

        for elm in self.ri_list:
            self.ri_files.append(li.load(elm, sr=22050)[0])

    def __len__(self):
        return len(self.ri_list)

    def __getitem__(self,i):
        return self.ri_files[i]

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

ri = RI()
current = 0

key = mido.get_input_names()[0]

x = torch.Tensor([0,0,0,0]).unsqueeze(0).to(device)
o = torch.zeros([1,7]).to(device)
s = torch.zeros([1,12]).to(device)



fs = 22050
window = np.hanning(2048)

reverb = args.reverb

with mido.open_input(key) as inport:
    print("Go!")
    for msg in inport:
        if msg.type == "note_on" and msg.channel:
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



            S = np.ascontiguousarray(S)

            sig = gl.griffin_lim_reconstruct_single_precision(args.gl_iteration, S, window, 3, 256)

            if reverb:
                sig = fftconvolve(sig, ri[current])

            sig = sig/ np.max(abs(sig))
            sd.play(sig[3*fs//10:],22050)

        elif msg.type == "control_change":
            k = msg.control % 4
            v = 10*(msg.value/127-.5)
            x[:,k] = v
            print("{}             ".format(x), end="\r", flush=True)

        elif msg.type == "note_on":
            if msg.note == 93:
                reverb = (reverb + 1) % 2
                print(("reverb = %d" % reverb) + "        ", end="\r", flush=True)
            elif msg.note == 66:
                current = (current - 1) % len(ri)
                print(ri.ri_list[current] + "        ", end="\r", flush=True)
            elif msg.note == 67:
                current = (current + 1) % len(ri)
                print(ri.ri_list[current] + "        ", end="\r", flush=True)
            else:
                pass
                # print(msg.note)
