import copy
import numpy as np
import matplotlib.pyplot as plt


# Create signal
srate = 1000  # Hz
time = np.arange(0, 3, 1/srate)
n = len(time)
p = 15  # Poles for random interpolation

# Noise level measured in standard deviations
noiseamp = 5

# Amplitude modulator and noise level
ampl = np.interp(np.linspace(1, p, n), np.arange(0, p), np.random.rand(p) * 30)
noise = noiseamp * np.random.randn(n)
signal = ampl + noise

# Create Gaussian kernel
# Full-width half-maximum: the key Gaussian parameter
fwhm = 25  # in ms

# Normalised time vector in ms
k = 40
gtime = 1000*np.arange(-k, k+1)/srate

# Create Gaussian window
gauswin = np.exp(-(4 * np.log(2) * gtime ** 2) / fwhm ** 2)

# Compute empirical FWHM
pstPeakHalf = k + np.argmin((gauswin[k:] - .5) ** 2)
prePeakHalf = np.argmin((gauswin - .5) ** 2)

empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]

# Show the Gaussian
plt.plot(gtime, gauswin, 'ko-')
plt.plot([gtime[prePeakHalf], gtime[pstPeakHalf]], [gauswin[prePeakHalf], gauswin[pstPeakHalf]], 'm')

# Then normalise Gaussian to unit energy
gauswin = gauswin / np.sum(gauswin)

plt.xlabel('Time (ms)')
plt.ylabel('Gain')

plt.show()


# Implement the filter
# Initialise filtered signal vector
filtsigG = copy.deepcopy(signal)

# Implement the running mean filter
for i in range(k + 1, n - k):
    # Each point is the weighted average of k surrounding points
    filtsigG[i] = np.sum(signal[i - k: i + k + 1] * gauswin)

# Plot
plt.plot(time, signal, 'r', label='Original')
plt.plot(time, filtsigG, 'k', label='Gaussian-filtered')

plt.xlabel('Time (s)')
plt.ylabel('Amp. (a.u.)')
plt.legend()
plt.title('Gaussian smoothing filter')


# For comparison, plot mean smoothing filter
# Initialise filtered signal vector
filtsigMean = copy.deepcopy(signal)

# Implement the running mean filter
# Note: using mk instead of k to avoid confusion with k above
mk = 20  # Filter window is actually mk*2+1
for i in range(mk + 1, n - mk - 1):
    # Each point is the average of k surrounding points
    filtsigMean[i] = np.mean(signal[i-mk:i+mk+1])

plt.plot(time, filtsigMean, 'b', label='Running mean')
plt.legend()
plt.show()
