import numpy as np
from numpy import linalg as LA
import math

def calc_sampleParam(micPos, w_0, c):

    #computes sampling period and number of samples within TDOA interval.

    # IN:
    # micPos         microphone positions - channels x coordinates
    # w_0            band limit
    # c              speed of sound
    #
    # OUT:
    # T              sampling period
    # N_mm           TDOAs - microphone pairs

    M = len(micPos);
    # microphone pairs
    P = M*(M-1)/2;
    # sampling period
    T = math.pi/w_0;

    dist = np.zeros((int(P),1));
    p = 0;
    for mprime in range (1,M+1):
        for m in range (mprime+1,M+1):
            p = p+1;
            # distance between microphones
            dist[p-1] = LA.norm(micPos[m-1,:] - micPos[mprime-1,:]);


    # distance between samples
    Delta_t_0 = dist/c;

    # samples inside TDOA interval
    N_mm = np.floor(Delta_t_0/T);

    return T, N_mm
