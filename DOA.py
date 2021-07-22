import numpy as np
from numpy import linalg as LA

def DOA (loc, arrayCenterPos):
    J = len(loc);
    DOAvec = loc - np.tile(arrayCenterPos, (J, 1));

    ang_pol = np.zeros((J, 1))
    ang_az = np.zeros((J, 1));


    for i in range(1, len(DOAvec)+1):
        ang_pol = np.arccos(DOAvec[i - 1, 2] / LA.norm(DOAvec[i - 1, :]));
        ang_az = np.arctan2(DOAvec[i - 1, 1], DOAvec[i - 1, 0]);

    
    DOAang = np.rad2deg([ang_pol, ang_az]);
    print(ang_pol)
    print(ang_az)
    print('b')
    return (DOAvec, DOAang)
