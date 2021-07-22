import numpy as np
import math
from scipy.fft import fft, ifft
def calc_STFT(x, fs, win, N_STFT, R_STFT, sides):
    N_STFT_half = N_STFT/2 + 1;

    #get frequency vector
    f = np.linspace(0,fs/2,int(N_STFT_half));
    if (sides=='twosided'):
        f = [f, np.take(-f,(range(len(f)-2,0,-1)))];

    #init
    L = math.floor((len(x) - N_STFT + R_STFT)/R_STFT);
    M = len(np.transpose(x));
    if (sides == 'onesided'):
        X = np.zeros((int(N_STFT_half), L, len(np.transpose(x))));
    if (sides == 'twosided'):
        X = np.zeros(N_STFT, L, M);

    X=np.complex256(X);

    for m in range (1,M+1):
        for l in range (1,L+1): # Frame index
            x_frame = x[(l-1)*int(R_STFT)+1-1:(l-1)*int(R_STFT)+int(N_STFT), m-1];
            X_frame = fft(np.multiply(win,x_frame));
            if (sides == 'onesided'):
                X[:,l-1,m-1] = X_frame[0:int(N_STFT_half)];
            if (sides == 'twosided'):
                X[:,l-1,m-1] = X_frame;
    return X, f
