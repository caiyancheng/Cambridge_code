import csv
import math

import matplotlib.pyplot as plt
import numpy as np
from TCSF.TCSF import TCSF

class VESA_flicker_visibility:
    '''
    VESA method
    '''
    def __int__(self, data_path = '../LuminanceData.csv'):
        self.data_path = data_path
        self.x_time = []
        self.y_luminance = []

    def read_data(self):
        try:
            fp = open(self.data_path, 'r')
        except:
            raise AttributeError("Can't load data")
        Luminance_data = csv.reader(fp)

        for i in Luminance_data:
            if Luminance_data.line_num == 1:
                continue
            self.x_time.append(float(i[0]))
            self.y_luminance.append(float(i[1]))
        self.x_time = np.array(self.x_time)
        self.y_luminance = np.array(self.y_luminance)

    def plot_pict(self, x_array, y_array, x_label, y_label, title, fig_size=False, save=False, save_fig_name='no name'):
        if fig_size:
            plt.figure(figsize=fig_size)
        plt.plot(x_array, y_array)
        plt.title(label=title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if save:
            plt.savefig(f'{save_fig_name}.png')
            plt.close()
        else:
            plt.show()

    def run_dft(self):
        t_s = (self.x_time[1:] - self.x_time[:-1]).mean()
        w_s = 1 / t_s

        self.N_s = len(self.x_time)
        self.K_DFT = np.abs(np.fft.fft(self.y_luminance)) / self.N_s  # (N_s ** 0.5) The algorithm in the book conflicts with the example.
        # We believe that the example conforms to the standard DFT, so we change the algorithm in the book. See the report for details.
        self.x_freq = np.arange(0, self.N_s) * w_s / self.N_s
        self.plot_pict(x_array=self.x_freq[:int(self.N_s/2)], y_array=self.K_DFT[:int(self.N_s/2)], x_label='Frequency', y_label='Luminance ($cd / m^2$)',
                       title='spectrum overall', save=True, save_fig_name='spectrum_overall')
        # x_freq_sub = self.x_freq[self.x_freq <= 120]
        # N_s_sub = len(x_freq_sub)
        # self.plot_pict(x_array=self.x_freq[:N_s_sub], y_array=self.K_DFT[:N_s_sub],
        #                x_label='Frequency', y_label='Luminance ($cd / m^2$)',
        #                title='spectrum overall', save=True, save_fig_name='spectrum_overall_sub')
        self.K_mean = self.y_luminance.mean()

    def compute(self):
        X_frequency = []
        Y_P = []
        for frequency_index in range(self.N_s):
            frequency = self.x_freq[frequency_index]
            if frequency == 0:
                continue
            if frequency > 120:
                break
            S = TCSF(frequency)
            P = S * self.K_DFT[frequency_index]
            X_frequency.append(frequency)
            Y_P.append(P)
        X_frequency = np.array(X_frequency)
        Y_P = np.array(Y_P)
        self.plot_pict(x_array=X_frequency, y_array=Y_P, x_label='Frequency', y_label='P',
                       title='P_Frequency', save=True, save_fig_name='P_Frequency')

        P_max = Y_P.max()
        X_frequency_P_max = X_frequency[Y_P.argmax()]
        VESA = 20 * math.log10((2 * P_max / (self.K_DFT[0])))
        print('VESA', VESA)




if __name__ == "__main__":
    VESA_Model = VESA_flicker_visibility()
    VESA_Model.__int__()
    VESA_Model.read_data()
    VESA_Model.run_dft()
    VESA_Model.compute()
