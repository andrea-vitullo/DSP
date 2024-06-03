import numpy as np
import matplotlib.pyplot as plt


# Create signal
srate = 1000
time = np.arange(0, 3, 1/srate)
n = len(time)
p = 15  # poles for random interpolation

# Noise level measured in standard deviations
noiseamp = 5

# Amplitude modulator and noise level
ampl = np.interp(np.linspace(0, p, n), np.arange(0, p), np.random.rand(p) * 30)
noise = noiseamp * np.random.randn(n)
signal = ampl + noise

# Initialise filtered signal vector
filtsig = np.zeros(n)  # filtsig = signal ---> For non-zero edge case

# Implement running mean filter
k = 20  # The filter window is actually k * 2 * 1

for i in range(k, n - k):
    # Each point is the average of k surrounding points
    filtsig[i] = np.mean(signal[i - k: i + k])

# Compute window size in ms
windowsize = 1000 * (k * 2 + 1)

# Plot the noisy abd filtered signals
plt.plot(time, signal, label='Original signal')
plt.plot(time, filtsig, label='Filtered signal')

plt.legend()
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Running mean filter')

plt.show()
