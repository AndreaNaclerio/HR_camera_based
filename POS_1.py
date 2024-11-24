import math

import numpy as np
from scipy import signal

import math

import cv2
from scipy import io as scio
from scipy import linalg
from scipy import signal
from scipy import sparse
#from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error



def _process_video(signal_RGB):
  RGB = [0]*3
  count = 0
  for channel in signal_RGB.keys():
    RGB[count] = signal_RGB[channel]
    count += 1
  return np.transpose(np.array(RGB)) # to obtain the desidered structure

def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal



'''
POS algorithm:
- signal is divided in windows, and for each one the PPG is extracted and then concatenated (to recreate the entire signal)
- extracted the 1D signals from the video
- apply the pseudo-code present in the paper
- extract the rPPG signal
- detrend the signal (DA VEDERE BENE IN COSA CONSISTE)
- only then the filter is applied


NB. una volta estratto il segnale BVP/rPPG, anche in questo caso sarà applicata lo stesso post-processing che è stato utilizzato nel caso del CHROM.
Più in generale possiamo dire che il post-processing viene applicato a tutti i segnali estratti, almeno quelli con metodi unsupervised
-> ATTENZIONE: è per questo motivo che quando andiamo a vedere l'algoritmo GREEN, questo contiene solamente estrazione del canale verde senza nessun 
filtraggio al suo interno. Questo perchè il filtraggio verrà fatto durante il post-processing

-> ATTENZIONE: anche nel caso degli altri algoritmi in cui nel codice troviamo in filtraggio, viene applicato il postprocessing, e quindi un ulteriore 
filtraggio per ripulire ulteriormente il segnale ed estrarre le metriche di nostro interesse (heart rate, SNR ecc) 
'''
def POS(signal_RGB, fs):
    WinSec = 1.6
    RGB = _process_video(signal_RGB) #andiamo ad estrarre i segnali 1D dal video (1 valore per frame mantenendo separati i diversi canali)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N): #pseudo-codice inserito nel papar (lo stesso che ho implementato) 
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP