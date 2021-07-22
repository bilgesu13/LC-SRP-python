# LC-SRP-python
Low-Complexity Steered Response Power Mapping based on Nyquist-Shannon Sampling. 

calc_FD_GCC.py computes frequency domain GCC.
calc_SRP.py computes SRP approximation. 
calc_SRPconv.py computes conventional SRP.
calc_STFT.py computes the STFT.
calc_sampleParam.py computes sampling period and number of samples within TDOA interval.
coord_loc_1_8.mat location of the sources.
coord_mic_array.mat coordinates of the microphone array.
gen_searchGrid.py generates search grid.
main.py defines the configuration, loads microphone signals, performs conventional and low-complexity SRP and computes approximation and localization errors. This is done for eight different source locations.
