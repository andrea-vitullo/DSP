import numpy as np
import matplotlib.pyplot as plt


# Polynomial intuition =============================================

order = 2
x = np.linspace(-15, 15, 100)

y = np.zeros(len(x))

for i in range(order + 1):
    y = y + np.random.randn(1) * x ** i

plt.plot(x, y)
plt.title('Order-%d polynomial' % order)
plt.show()


# Generate signal with slow polynomial artifact ====================

n = 10000
t = range(n)
k = 10  # Number of poles for random amplitudes

slowdrift = np.interp(np.linspace(1, k, n), np.arange(0, k), 100 * np.random.randn(k))
signal = slowdrift + 20 * np.random.randn(n)

# plot
plt.plot(t, signal)
plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.show()


# Fit a 3-order polynomial =========================================

# Polynomial fit (returns coefficients)
p = np.polyfit(t, signal, 3)

# Predicted data is evaluation of polynomial
yHat = np.polyval(p, t)

# Compute residual (the cleaned signal)
residual = signal - yHat

# Now plot the fit (the function that will be removed)
plt.plot(t, signal, 'b', label='Original')
plt.plot(t, yHat, 'r', label='Polyfit')
plt.plot(t, residual, 'k', label='Filtered signal')

plt.legend()
plt.show()


# Bayes information criterion to find optimal order ================

# Possible orders
orders = range(5, 40)

# Sum of squared errors (sse is reserved!)
sse1 = np.zeros(len(orders))

# Loop through orders
for ri in range(len(orders)):

    # Compute polynomial (fitting time series)
    yHat = np.polyval(np.polyfit(t, signal, orders[ri]), t)

    # Compute fit of model to data (sum of squared errors)
    sse1[ri] = np.sum((yHat - signal) ** 2) / n


# Bayes information criterion
bic = n * np.log(sse1) + orders * np.log(n)

# Best parameter has the lowest BIC
bestP = min(bic)
idx = np.argmin(bic)

# Plot the BIC
plt.plot(orders, bic, 'ks-')
plt.plot(orders[idx], bestP, 'ro')
plt.xlabel('Polynomial order')
plt.ylabel('Bayes information criterion')
plt.show()


# Now repeat filter for best (smallest) BIC ========================

# Polynomial fit
polycoefs = np.polyfit(t, signal, orders[idx])

# Estimated data based on the coefficients
yHat = np.polyval(polycoefs, t)

# Filtered signal is residual
filtsig = signal - yHat

# Plotting
plt.plot(t, signal, 'b', label='Original')
plt.plot(t, yHat, 'r', label='Polynomial fit')
plt.plot(t, filtsig, 'k', label='Filtered')

plt.xlabel('Time (a.u.)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
