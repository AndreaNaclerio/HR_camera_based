"""LGI
Local group invariance for heart rate estimation from face videos.
Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V.
In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 1254–1262
(2018).
"""

import math

import numpy as np
from scipy import linalg
from scipy import signal


def LGI(signall):
    precessed_data = process_signal(signall)
    print(precessed_data.shape)
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp

def process_signal(signall):
    # Extract the R, G, B channels from the signal dictionary
    R = np.array(signall['R'])
    G = np.array(signall['G'])
    B = np.array(signall['B'])
    
    # Stack the R, G, B channels vertically into a 2D array of shape (3, N)
    RGB = np.vstack([R, G, B])
    
    # Reshape the array to have shape (1, 3, N)
    RGB = RGB.reshape(1, 3, -1)
    
    return RGB
