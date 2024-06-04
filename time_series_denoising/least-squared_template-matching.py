import numpy as np
import sio
import matplotlib.pyplot as plt


# Load dataset
matdat = sio.loadmat('Your File')
EEGdat = matdat['EEGdat']
eyedat = matdat['eyedat']
timevec = matdat['timevec'][0]
MN = np.shape(EEGdat)  # Matrix sizes

# Initialise residual data
resdat = np.zeros(np.shape(EEGdat))

# Loop over trials
for triali in range(MN[1]):

    # Build the least-squares model as intercept and EOG from this trial
    X = np.column_stack((np.ones(MN[0]), eyedat[:, triali]))

    # Compute regression coefficients for EEG channel
    b = np.linalg.solve(X.T@X, X.T@EEGdat[:, triali])

    # predicted data
    yHat = X@b

    # New data are the residuals after projecting out the best EKG fit
    resdat[:, triali] = EEGdat[:, triali] - yHat

# Plotting
# Trial averages
plt.plot(timevec, np.mean(eyedat, axis=1), label='EOG')
plt.plot(timevec, np.mean(EEGdat, axis=1), label='EEG')
plt.plot(timevec, np.mean(resdat, 1), label='Residual')

plt.xlabel('Time (ms)')
plt.legend()
plt.show()

# Show all trials in a map
clim = [-1, 1]*20

plt.subplot(131)
plt.imshow(eyedat.T)
plt.title('EOG')

plt.subplot(132)
plt.imshow(EEGdat.T)
plt.title('EOG')

plt.subplot(133)
plt.imshow(resdat.T)
plt.title('Residual')

plt.tight_layout()
plt.show()
