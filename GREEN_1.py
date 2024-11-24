from statistics import *
from scipy import signal
import numpy as np
from scipy.signal import butter, filtfilt


def process_video(signal_RGB):
  return np.array(signal_RGB['G']) # to obtain the desidered structure

def GREEN(signal_RGB,FS):

    G_signal = process_video(signal_RGB)
    G_signal.shape

    avg = mean(G_signal)
    G_det = G_signal - [avg]*G_signal.shape[0]

    # if we look at the Green algorith implemenation in the TOOLBOX, they don't filter the signal, since an additional filtering is inserted during the hear rate (HR or BPM) extraction

    return G_det