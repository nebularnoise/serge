#author: Antoine CAILLON
import librosa as li
import torch
import numpy as np
import matplotlib.pyplot as plt
from final_model import AudioDataset

motu = AudioDataset("../notebooks/motu/motu_dataset/*.wav", process=1, slice_size=128)

print("%d samples found." % len(motu))

for i in range(len(motu)):
    s,ss,f = motu[i]
    f = f.numpy()
    [x,fs] = li.load(motu.liste[i].replace(".pt", ".wav"))
    x_ = np.fft.fft(x)
    f_ = np.linspace(0,fs,len(x_))
    #plt.plot(f_, np.log(abs(x_)))
    plt.figure(figsize=(15,7))
    plt.subplot(121)
    plt.imshow(s.numpy(), origin="lower", aspect="auto")
    plt.subplot(122)
    plt.imshow(ss.numpy(), origin="lower", aspect="auto")
    plt.title("%d over 102, f0= %f" % (i,f))
    #plt.xlim([100,1000])
    plt.show()
