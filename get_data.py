# -*- coding: utf-8 -*-
"""
This code can be used to load the h5 file with the measurements from HBTA into python
for it to run, the measurements file "hbta_SDproject.hdf5" and this script must be in the same directory
"""
import koma.oma, koma.plot
from koma.signal import xwelch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, welch, resample

import matplotlib.pyplot as plt
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

# %% opening file
acceleration = []
sensor_names = []
position = []
direction = []
with h5py.File('hbta_SDproject.hdf5', 'r') as h5f:
    for sensor in h5f.keys():
        for channel in h5f[sensor].keys():
            acceleration.append(np.array(h5f[sensor][channel])) # use signal.detrend here
            sensor_names.append(str(sensor)+str(channel))
            position.append(h5f[sensor].attrs['position'])
            if channel == 'x':
                direction.append(h5f[sensor].attrs['transformation_matrix'][0])
            elif channel == 'y':
                direction.append(h5f[sensor].attrs['transformation_matrix'][1])
            elif channel == 'z':
                direction.append(h5f[sensor].attrs['transformation_matrix'][2])
        
    fs = h5f[sensor].attrs['samplerate']
acceleration = np.array(acceleration)
position = np.array(position)
direction = np.array(direction)

W = fs/2
N = len(acceleration[1])

dw = W/N
T = 1/dw

t = np.linspace(0,T,N)

"""plt.figure(figsize = (12,3))
plt.plot(t,acceleration[0,:])
plt.plot(t,acceleration[1,:])
plt.plot(t,acceleration[2,:])
plt.grid()
plt.show()"""

"""plt.figure(figsize = (12,3))
plt.plot(t,acceleration[3,:], linewidth=.5)
plt.plot(t,acceleration[4,:], linewidth=.5)
plt.plot(t,acceleration[5,:], linewidth=.5)
plt.grid()
plt.show()"""

v = cumtrapz(acceleration[0, :], t, initial=0)
x = cumtrapz(v, t, initial=0)

"""plt.figure(figsize = (12,3))
plt.plot(t,acceleration[0,:])
plt.grid()
plt.show()"""
"""
# Create a 3x9 grid for the 27 subplots
fig, axes = plt.subplots(3, 9, figsize=(22, 6))  # Adjust figsize as needed

# Iterate over each of the 27 acceleration arrays and plot them
for i in range(27):
    row = i % 3  # Determine the row (integer division)
    col = i // 3   # Determine the column (modulus)
    
    axes[row, col].plot(t, acceleration[i, :])  # Plot each acceleration array
    axes[row, col].set_title(f"Plot {i+1}")     # Optional: Set title for each subplot
    axes[row, col].grid(True)
    axes[row, col].set_xlabel(r"$t$")                   # Add grid to each subplot

# Adjust layout to prevent overlap
plt.ylabel(r"$a(t)$")
plt.tight_layout()"""

"""
# Show the plot
plt.show()"""

# Create a 3x9 grid for the 27 subplots
fig, axes = plt.subplots(3, 9, figsize=(18, 6))  # Adjust figsize as needed

# Iterate over each of the 27 acceleration arrays and plot them
for i in range(27):
    f, psd = welch(acceleration[i,:], fs=fs, nfft=2048, nperseg=1024)
    row = i % 3  # Determine the row (integer division)
    col = i // 3   # Determine the column (modulus)
    
    axes[row, col].plot(f, psd)  # Plot each acceleration array
    axes[row, col].set_title(f"Plot {i+1}")     # Optional: Set title for each subplot
    axes[row, col].grid(True)
    axes[row, col].set_xlabel(r"$\omega$")                   # Add grid to each subplot

# Adjust layout to prevent overlap
fig.text(0.0, 0.5, r'$a(t)$', va='center', rotation='vertical')
plt.tight_layout()


# Show the plot
plt.show()


"""fig, axs = plt.subplots(3, 1, figsize=(14, 3))

for i in range(27, 3):
    axs[0, i/3].plot(t, acceleration[i, :])
    axs[0, i/3].set_xlabel(r"$t$", fontsize=14)
    axs[0, i/3].set_ylabel(r"$a(t)$", fontsize=14)
    axs[0, i/3].grid()

    axs[1, i/3].plot(t, acceleration[1+i, :])
    axs[1, i/3].set_xlabel(r"$t$", fontsize=14)
    axs[1, i/3].grid()
    
    axs[2, i/3].plot(t, acceleration[2+i, :])
    axs[2, i/3].set_xlabel(r"$t$", fontsize=14)
    axs[2, i/3].grid()

plt.tight_layout()
plt.suptitle("Acceleration of every sensor", fontsize=16)
plt.show()"""

"""f, psd = welch(acceleration[27,:], fs=fs, nfft=2048, nperseg=1024)
plt.plot(f, psd)
plt.show()"""


#USIKKER PÅ HVA XWELCH BRUKES TIL??
"""f, cpsd = xwelch(acceleration[0,:], fs=fs, nfft=2048, nperseg=1024)
plt.plot(f, cpsd)
plt.show()"""




# %%
