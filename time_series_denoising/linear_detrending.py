import numpy as np
import matplotlib.pyplot as plt
import scipy


# Create signal with linear trend imposed
n = 2000
signal = np.cumsum(np.random.randn(n)) + np.linspace(-30, 30, n)

# Linear detrending
detsignal = scipy.signal.detrend(signal)

# Get means
omean = np.mean(signal)  # Original mean
dmean = np.mean(detsignal)  # Detrended mean

# Plot signal and detrended signal
plt.plot(range(0, n), signal, label='Original, mean=%d' % omean)
plt.plot(range(0, n), detsignal, label='Detrended, mean=%d' % dmean)

plt.legend()
plt.show()
