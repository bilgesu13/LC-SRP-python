import numpy as np
import math

def calc_SRPconv(Psi_STFT, omega, Delta_t_i):

    J = np.size(Delta_t_i,0);
    P = np.size(Delta_t_i,1);
    K = np.size(Psi_STFT,0);
    L = np.size(Psi_STFT,1);
    P = np.size(Psi_STFT,2);

    SRP_stack = np.zeros([L,J]);
    for l in range (1, L+1):
        print(l)
        SRP_FD = np.zeros([K,J]);
        SRP_FD = np.complex64(SRP_FD);
        for k in range (2, K+1):
            psi = np.squeeze(Psi_STFT[k-1,l-1,:]);
            i=np.arange(len(SRP_FD[k-1]))+1;
            SRP_FD[k-1,:] = np.dot((np.exp(1j*Delta_t_i[i-1,:]*omega[k-1])),psi);

        SRP_stack[l-1,:] = 2*np.sum(np.real(SRP_FD),axis=0);
    return SRP_stack

