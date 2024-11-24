import numpy as np
import math
from scipy import signal


def CHROME(signal_RGB,FS):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6 #durata della finestra in sec (finestra per andare a dividere il segnale in più parti)

    RGB = _process_video(signal_RGB) #questa funzione va a creare i 3 segnali 1D a partire dal video (quindi estrae un valore da ogni frame, mantenendo la divisione tra R G e B channel)
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass') #prepariamo il filtro che andrà utlizzato per filtrare solamente le frequenze di nostro interesse

    '''
    l'idea è quella di dividere il segnale in diversi intervalli e per ciascuno estrarre il segnale PPG/BVP signal. Le diverse porzioni andranno concatenate tra di loro
    per formare il segnale PPG completo
    '''
    WinL = math.ceil(WinSec*FS) #moltiplicando lunghezza nel tempo per FS, andiamo a calcolare la lunghezza in samples della singola finestra (quanti samples dobbiamo andare a considerare per estrarre una finestra)
    if(WinL % 2): #
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        temp = SWin[:int(WinL//2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S
    return BVP

def _process_video(signal_RGB):
  RGB = [0]*3
  count = 0
  for channel in signal_RGB.keys():
    RGB[count] = signal_RGB[channel]
    count += 1
  return np.transpose(np.array(RGB)) # to obtain the desidered structure