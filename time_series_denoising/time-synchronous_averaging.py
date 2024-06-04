import numpy as np
import matplotlib.pyplot as plt


# Simulate data
# Create event (Gaussian's derivative)
k = 100  # duration of event in time points
event = np.diff(np.exp(-np.linspace(-2, 2, k + 1) ** 2))
event = event / np.max(event)  # Normalise to max = 1

# Event onset times
Nevents = 30
onsettimes = np.random.permutation(10000 - k)
onsettimes = onsettimes[0:Nevents]

# Put event into data
data = np.zeros(10000)
for ei in range(Nevents):
    data[onsettimes[ei]:onsettimes[ei] + k] = event

# Add noise
data = data + .5 * np.random.randn(len(data))

# Plot data
plt.subplot(211)
plt.plot(data)

# Plot one event
plt.subplot(212)
plt.plot(range(k), data[onsettimes[3]:onsettimes[3] + k])
plt.plot(range(k), event)
plt.show()

# Extract all events into a matrix

datamatrix = np.zeros((Nevents, k))

for ei in range(Nevents):
    datamatrix[ei, :] = data[onsettimes[ei]:onsettimes[ei] + k]

plt.imshow(datamatrix)
plt.xlabel('Time')
plt.ylabel('Event number')
plt.title('All events')
plt.show()

plt.plot(range(0, k), np.mean(datamatrix, axis=0), label='Averaged')
plt.plot(range(0, k), event, label='Ground-truth')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Average events')
plt.show()
