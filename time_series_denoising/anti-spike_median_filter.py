import numpy as np
import matplotlib.pyplot as plt
import copy


# Create signal
n = 2000
signal = np.cumsum(np.random.randn(n))


# Proportion of time points to replace with noise
propnoise = .05

# Find noise points
noisepnts = np.random.permutation(n)
noisepnts = noisepnts[0:int(n * propnoise)]

# Generate signal and replace points with noise
signal[noisepnts] = 50 + np.random.rand(len(noisepnts)) * 100


# Use hist to pick threshold
plt.hist(signal, 100)
plt.show()

# Visual-picked threshold
threshold = 40  # Threshold to be set based on the randomly generated noise


# Find data values above the threshold
suprathresh = np.where(signal > threshold)[0]

# Initialise filtered signal
filtsig = copy.deepcopy(signal)

# Loop through suprathreshold points and set to median of k
k = 20  # Actual window is k * 2 + 1
for ti in range(len(suprathresh)):

    # Lower and upper bounds
    lowbnd = np.max((0, suprathresh[ti] - k))
    uppbnd = np.min((suprathresh[ti] + k + 1, n))

    # Compute median of surrounding points
    filtsig[suprathresh[ti]] = np.median(signal[lowbnd:uppbnd])

# Plot
plt.plot(range(0, n), signal, range(0, n), filtsig)
plt.show()
