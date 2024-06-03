import copy
import sio
import numpy as np
import matplotlib.pyplot as plt


# Import data
emgdata = sio.loadmat('Your file')

# Extract needed variables
emgtime = emgdata['emgtime'][0]
emg = emgdata['emg'][0]

# Initialise filtered signal
emgf = copy.deepcopy(emg)

# The loop version for interpretability
for i in range(1, len(emgf) - 1):
    emgf[i] = emg[i] ** 2 - emg[i - 1] * emg[i + 1]

# The vectorised version for speed and elegance
emgf2 = copy.deepcopy(emg)
emgf2[1:-1] = emg[1:-1] ** 2 - emg[0:-2] * emg[2:]

# Convert both signals to zscore

# Find timepoint zero
time0 = np.argmin(emgtime ** 2)

# Convert original EMG to z-score from time-zero
emgZ = (emg-np.mean(emg[0:time0])) / np.std(emg[0:time0])

# Same for filtered EMG energy
emgZf = (emgf-np.mean(emgf[0:time0])) / np.std(emgf[0:time0])


# Plot "raw" (normalised to max.1)
plt.plot(emgtime, emg / np.max(emg), 'b', label='EMG')
plt.plot(emgtime, emgf / np.max(emgf), 'm', label='TKEO energy')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude or energy')
plt.legend()

plt.show()

# Plot zscored
plt.plot(emgtime, emgZ, 'b', label='EMG')
plt.plot(emgtime, emgZf, 'm', label='TKEO energy')

plt.xlabel('Time (ms)')
plt.ylabel('Zscore relative to pre-stimulus')
plt.legend()
plt.show()
