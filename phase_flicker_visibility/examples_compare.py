import numpy as np
import matplotlib.pyplot as plt
import math

A_set_1 = [1, 1] #amplitude set
f_set_1 = [1, 1.01] #frequency set
p_set_1 = [0, 0] #phase set

A_set_2 = [1, 1] #amplitude set
f_set_2 = [1, 1.01] #frequency set
p_set_2 = [0, -3.1415926] #phase set


if len(A_set_1) != len(p_set_1) or len(A_set_1) != len(f_set_1) or len(A_set_2) != len(p_set_2) or len(A_set_2) != len(f_set_2):
    raise AttributeError("amplitude set & phase set & angular frequency set don't have the same length")

x_time = np.linspace(0,1000,1000)

y_luminance_1 = np.zeros(len(x_time))
y_luminance_2 = np.zeros(len(x_time))

for x_index in range(len(x_time)):
    t = x_time[x_index]
    y_luminance_1_index = 0
    y_luminance_2_index = 0
    for f_index in range(len(A_set_1)):
        y_luminance_1_index += A_set_1[f_index] * np.sin(2 * math.pi * f_set_1[f_index] * t + p_set_1[f_index])
    for f_index in range(len(A_set_2)):
        y_luminance_2_index += A_set_2[f_index] * np.sin(2 * math.pi * f_set_2[f_index] * t + p_set_2[f_index])
    y_luminance_1[x_index] = y_luminance_1_index
    y_luminance_2[x_index] = y_luminance_2_index

N_s = len(y_luminance_1)

t_s = (x_time[1:] - x_time[:-1]).mean()
w_s = 1 / t_s
x_frequency = np.arange(0, N_s) * w_s / N_s
y_luminance_DFT_1 = np.fft.fft(y_luminance_1)
y_luminance_DFT_2 = np.fft.fft(y_luminance_2)
y_luminance_DFT_1_Amplitude = np.abs(y_luminance_DFT_1) / len(y_luminance_1)
y_luminance_DFT_2_Amplitude = np.abs(y_luminance_DFT_2) / len(y_luminance_2)
y_luminance_DFT_1_phase = np.angle(y_luminance_DFT_1)
y_luminance_DFT_2_phase = np.angle(y_luminance_DFT_2)

# plt.figure(figsize=(6,3.3))
# plt.figure(figsize=(15,5))
# plt.subplot(1,3,1)
# plt.plot(x_time, y_luminance_1)
# plt.plot(x_time, y_luminance_2)
# plt.title('Signal Waveform')
# plt.xlabel('Time')
# plt.ylabel('Luminance cd/m2')

Frequency_Limit = 2
x_frequency_sub = x_frequency[x_frequency<Frequency_Limit]
N_s_sub = len(x_frequency_sub)

plt.subplot(1,3,2)
plt.plot(x_frequency_sub, y_luminance_DFT_1_Amplitude[0:N_s_sub])
plt.plot(x_frequency_sub, y_luminance_DFT_2_Amplitude[0:N_s_sub])
plt.title('Amplitude-Frequency plot of DFT (FFT)')
plt.xlabel('Frequency')
plt.ylabel('Luminance cd/m2')
#
# # plt.subplot(1,3,3)
# plt.plot(x_frequency_sub, y_luminance_DFT_1_phase[0:N_s_sub])
# plt.plot(x_frequency_sub, y_luminance_DFT_2_phase[0:N_s_sub])
# plt.title('Phase-Frequency plot of DFT (FFT)')
# plt.xlabel('Frequency')
# plt.ylabel('Phase')

plt.savefig('Example_Compare_Amplitude.png')
plt.close()

# plt.plot(x_time, y_luminance)
# plt.title('Example_different_phase_and_frequency')
# plt.xlabel('Time')
# plt.ylabel('Luminance cd/m2')
# plt.savefig('Example_different_phase_and_frequency.png')
# plt.close()




