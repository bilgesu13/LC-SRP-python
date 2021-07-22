#############################################################################################################
# Copyright 2021 Bilgesu Çakmak
#
# This software is distributed under the terms of the GNU Public License
# version 3 (http://www.gnu.org/licenses/gpl.txt).
#
# A Matlab version of this code is available at
# https://github.com/tdietzen/LC-SRP.
#
# If you find it useful, please cite:
#
# [1] T. Dietzen, E. De Sena, and T. van Waterschoot, "Low-Complexity
# Steered Response Power Mapping based on Nyquist-Shannon Sampling," in
# Proc. 2021 IEEE Workshop Appl. Signal Process. Audio, Acoust. (WASPAA 2021), New Paltz, NY, USA, Oct. 2021.
#############################################################################################################




import time
### ACOUSTIC SETUP
from numpy import linalg as LA

import numpy as np


import scipy.io as spio

#speed of sound
c = 340;
#sample rate
fs = 16000;
#bandlimit
import math
pi=math.pi
w_0 = pi*fs;
#SNR in dB
SNR = 6;

## MICROPHONE ARRAY
# circular array, 10cm radius, six microphones
import scipy.io
#import matplotlib.pyplot as plt
tmp = scipy.io.loadmat('coord_mic_array.mat')
# array center
arrayCenterPos = tmp.get('arrayCenterPos')
# microphone positions
micPos = tmp.get('micPos');
# number of microphones
M = len(micPos)

### SOURCE LOCATIONS
# 8 different locations
tmp = scipy.io.loadmat('coord_loc_1_8.mat');
true_loc = tmp.get('true_loc');
# compute ground truth DOA vectors for source locations

from DOA import DOA
true_DOAvec, DOAang=DOA(true_loc,arrayCenterPos)
L = 32;


# STFT PARAMETERS
# window size
N_STFT = 2048;
# shift
R_STFT = N_STFT/2;
# window
win = np.sqrt(np.hanning(N_STFT));
N_STFT_half = math.floor(N_STFT/2)+1;
# frequency vector
omega = 2*pi*np.transpose(np.linspace(0,fs/2,N_STFT_half));


# CANDIDATE LOCATIONS
# polar angles of candidate locations
ang_pol= np.arange(90, 181, 2).tolist();
# azimuth angles of candidate locations 
ang_az = np.arange(0,359,2).tolist();
# compute candidate DOA vectors and TDOAs
from gen_searchGrid import gen_searchGrid
DOAvec_i, Delta_t_i = gen_searchGrid(micPos, ang_pol, ang_az, 'spherical', c);

DOAvec_i_tmp= DOAvec_i.copy();
DOAvec_i_tmp2= DOAvec_i.copy();


# SRP APPROXIMATION PARAMETERS
# compute sampling period and number of samples within TDOA interval
from calc_sampleParam import calc_sampleParam
T, N_mm  =calc_sampleParam(micPos, w_0, c);
#number of auxilary samples (approximation will be computed for all values in vector)
N_aux = range(0,3);

## PROCESSING

# init results (per source location, frame, number of auxilary samples)
# approximation error in dB


approxErr_dB=np.zeros((len(true_loc), L, len(N_aux)))
locErr=np.zeros((len(true_loc), L, len(N_aux)+1))

res = {'field1': approxErr_dB, 'field2': locErr};



import soundfile as sf

for true_loc_idx in range (1,len(true_loc)+1):
#for true_loc_idx in range (1,2):

    print(['PROCESSING SOURCE LOCATION'+str(true_loc_idx)])
    #GENERATE MICROPHONE SIGNALS
    #speech componentfor selected source
    x_TD,samplerate = sf.read('x_loc' +str(true_loc_idx)+ '.wav');
    #noise component
    v_TD,sr = sf.read('v.wav');
    #scale noise component
    from set_SNR import set_SNR
    v_TD = set_SNR(x_TD, v_TD, SNR);

    # transform to STFT domain
    from calc_STFT import calc_STFT
    x_STFT,f_x = calc_STFT(x_TD, fs, win, N_STFT, R_STFT, 'onesided');
    v_STFT,f_v = calc_STFT(v_TD, fs, win, N_STFT, R_STFT, 'onesided');

    # discard frames that do not contain speech energy(local SNR 15 dB below average)
    l = 1;
    useframe_idx = np.array([]);
    while len(useframe_idx) < L:
        SNR_local = 10*math.log(((sum(np.power(abs(x_STFT[:, l-1, 1-1]), 2)) / sum(np.power(abs(v_STFT[:, l-1, 1-1]), 2)))),10);
        if SNR_local > SNR - 15:
            useframe_idx=np.append(useframe_idx, l, axis=None)
        l = l + 1;


    # final microphone signal in STFT domain
    len_frame=len(useframe_idx)
    y_STFT = x_STFT[:, range(0, len_frame), :] + v_STFT[:, range(0, len_frame), :];
    for i in range(1, len_frame + 1):
        y_STFT[:, i - 1, :] = x_STFT[:, int(useframe_idx[i - 1]) - 1, :] + v_STFT[:, int(useframe_idx[i - 1]) - 1,
                                                                                :];



    ## PROCESSING
    from calc_FD_GCC import calc_FD_GCC
    psi_STFT = calc_FD_GCC(y_STFT); #sorun yok

    #conventional SRP

    print('* compute conventional SRP (stay tuned, this will take a few minutes)...')
    t = time.time();
    print (t)
    from calc_SRPconv import calc_SRPconv
    SRP_conv = calc_SRPconv(psi_STFT, omega, Delta_t_i);
    elapsed = time.time() - t;
    print(elapsed)
    print('done')


    # SRP approximation based on shannon nyquist sampes
    print('* compute SRP approximation...')
    t = time.time();
    from calc_SRP import calc_SRP
    SRP_appr = calc_SRP(psi_STFT, omega, T, N_mm, N_aux, Delta_t_i);
    elapsed = time.time() - t;
    print(elapsed)
    print('done')

    ####

    approxErr_dB = np.zeros([L, len(N_aux)]);
    locErr = np.zeros([L, len(N_aux) + 1]);

    maxIdx_conv = np.argmax(SRP_conv, 1);
    maxIdx_conv = maxIdx_conv.reshape(-1,1);
    estim_DOAvec = DOAvec_i_tmp[0:len(maxIdx_conv),:];
    for i in range (1,len(maxIdx_conv)+1):
        estim_DOAvec[i-1] = DOAvec_i_tmp[maxIdx_conv[i-1],:];
    locErr[:, 0] = np.rad2deg(np.arccos(np.dot(estim_DOAvec , np.transpose(true_DOAvec[true_loc_idx-1,:]))/(np.sqrt(np.sum(np.power(estim_DOAvec,2), axis=1)) * LA.norm(true_DOAvec[true_loc_idx-1,:]))));


    for N_aux_ind in range (1,len(N_aux)+1):
        approxErr = np.sum(np.power(SRP_conv - SRP_appr[:,:, N_aux_ind-1], 2), axis=1) / np.sum(np.power(SRP_conv, 2), axis=1);
        approxErr_dB[:, N_aux_ind-1] = 10*np.log10(approxErr);

        maxIdx_conv = np.argmax(SRP_appr[:,:, N_aux_ind-1], 1);
        estim_DOAvec = DOAvec_i_tmp2[0:len(maxIdx_conv),:];
        for i in range(1, len(maxIdx_conv) + 1):
            estim_DOAvec[i - 1] = DOAvec_i_tmp2[maxIdx_conv[i - 1], :];
        locErr[:, N_aux_ind] = np.rad2deg(np.arccos(np.dot(estim_DOAvec , np.transpose(true_DOAvec[true_loc_idx-1,:]))/(np.sqrt(np.sum(np.power(estim_DOAvec,2), axis=1)) * LA.norm(true_DOAvec[true_loc_idx-1,:]))));


    res['field1'][true_loc_idx-1,:,:] = approxErr_dB;
    res['field2'][true_loc_idx-1,:,:] = locErr;



print('DONE.')

