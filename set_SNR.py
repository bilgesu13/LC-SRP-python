import numpy as np
from numpy import linalg as LA
import math
import control

def set_SNR(x, v, SNR):

    # [v_scaled, scaling] = set_SNR(x, v, SNR)
    # scales the noise signal v relative to speech signal x in order to obtain specified SNR.

    # IN:
    # x           speech signal
    # v           noise signal
    # SNR         desired SNR

    # OUT:
    # v_scaled    scaled noise signal
    # scaling     scaling factor


    #scale signals
    power_x = LA.norm(x[:,0]);
    power_v  = LA.norm(v[:,0]);
    if math.isinf(SNR):
        scaling = 0;
    else:
        scaling = ((power_x/power_v)/control.db2mag(SNR));
    v_scaled = scaling*v;
    return v_scaled #,scaling