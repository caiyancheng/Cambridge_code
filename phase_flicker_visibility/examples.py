import numpy as np
import matplotlib.pyplot as plt
import math

A_set = [1, 1] #amplitude set
f_set = [1, 1.1] #frequency set
p_set = [0, -3.1415926] #phase set


if len(A_set) != len(p_set) or len(A_set) != len(f_set):
    raise AttributeError("amplitude set & phase set & angular frequency set don't have the same length")

x_time = np.linspace(0,1,1000)

y_luminance = np.zeros(len(x_time))

for x_index in range(len(y_luminance)):
    t = x_time[x_index]
    y_luminance_index = 0
    for f_index in range(len(A_set)):
        y_luminance_index += A_set[f_index] * np.sin(2 * math.pi * f_set[f_index] * t + p_set[f_index])
    y_luminance[x_index] = y_luminance_index

N_s = len(y_luminance)

t_s = (x_time[1:] - x_time[:-1]).mean()
w_s = 1 / t_s
x_frequency = np.arange(0, N_s) * w_s / N_s
y_luminance_DFT = np.fft.fft(y_luminance)
y_luminance_DFT_Amplitude = np.abs(y_luminance_DFT) / len(y_luminance)
y_luminance_DFT_phase = np.angle(y_luminance_DFT)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(x_time, y_luminance)
plt.title('Signal Waveform')
plt.xlabel('Time')
plt.ylabel('Luminance cd/m2')

Frequency_Limit = 2
x_frequency_sub = x_frequency[x_frequency<Frequency_Limit]
N_s_sub = len(x_frequency_sub)

plt.subplot(1,3,2)
plt.plot(x_frequency_sub, y_luminance_DFT_Amplitude[0:N_s_sub])
plt.title('Amplitude-Frequency plot of DFT (FFT)')
plt.xlabel('Frequency')
plt.ylabel('Luminance cd/m2')

plt.subplot(1,3,3)
plt.plot(x_frequency_sub, y_luminance_DFT_phase[0:N_s_sub])
plt.title('Phase-Frequency plot of DFT (FFT)')
plt.xlabel('Frequency')
plt.ylabel('Phase')

plt.savefig('Example.png')
plt.close()

# plt.plot(x_time, y_luminance)
# plt.title('Example_different_phase_and_frequency')
# plt.xlabel('Time')
# plt.ylabel('Luminance cd/m2')
# plt.savefig('Example_different_phase_and_frequency.png')
# plt.close()




