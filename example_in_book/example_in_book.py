import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

N_f = 4 #frames
w_s = 6000 #Hz
R = 60 #Hz
N_s = 400 #samples

f_c = 0.1

X_sample_index = np.arange(0, N_s)

Y_wave = np.zeros(N_s)
for i in range(N_s):
    I = i % 100
    if I <= 50:
        Y_wave[i] = 200 * (1 - math.exp(-I*f_c))
    else:
        Y_wave[i] = 200 * math.exp(-(I-50)*f_c)

plt.plot(X_sample_index, Y_wave)
plt.title('Luminance_Curve')
plt.xlabel('Sample Index')
plt.ylabel('Luminance cd/m2')
plt.savefig("Luminance_Curve.png")
plt.close()

X_frequency_index = np.arange(0, N_s) * w_s / N_s
K_DFT = np.abs(np.fft.fft(Y_wave, n=N_s, axis=-1, norm=None)) / np.sqrt(N_s)
plt.plot(X_frequency_index, K_DFT)
plt.title('K_DFT_original') #
plt.xlabel('Frequency')
plt.ylabel('Luminance cd/m2')
plt.savefig('K_DFT_original.png')
plt.close()

X_frequency_index = np.arange(0, N_s) * w_s / N_s
K_DFT = np.abs(np.fft.fft(Y_wave, n=N_s, axis=-1, norm=None)) / N_s
plt.plot(X_frequency_index, K_DFT)
plt.title('K_DFT_improve')
plt.xlabel('Frequency')
plt.ylabel('Luminance cd/m2')
plt.savefig('K_DFT_improve.png')
plt.close()

K_DCT = np.abs(cv2.dct(Y_wave)) / np.sqrt(N_s)
plt.plot(X_frequency_index, K_DCT)
plt.title('K_DCT')
plt.xlabel('Frequency')
plt.ylabel('Luminance cd/m2')
plt.savefig('K_DCT.png')
plt.close()