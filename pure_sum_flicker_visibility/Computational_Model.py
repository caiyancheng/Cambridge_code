import csv
import matplotlib.pyplot as plt
import numpy as np
from TCSF.TCSF import TCSF

class Pure_sum_flicker_visibility:
    '''
    This is the first version of the improvement. We consider that the waveform is similar to a square wave with
    a duty cycle of more than 90%, and there are many peaks in the DFT, so we cannot just take one peak. We use
    the method of weighted sum to calculate JND.
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
        # self.plot_pict(x_array=self.x_freq[:int(self.N_s/2)], y_array=self.K_DFT[:int(self.N_s/2)], x_label='Frequency', y_label='Luminance ($cd / m^2$)',
        #                title='spectrum overall', save=True, fig_size=(6,6), save_fig_name='spectrum_overall')
        x_freq_sub = self.x_freq[self.x_freq <= 120]
        N_s_sub = len(x_freq_sub)
        self.plot_pict(x_array=self.x_freq[1:N_s_sub], y_array=self.K_DFT[1:N_s_sub],
                       x_label='Frequency', y_label='Luminance ($cd / m^2$)',
                       title='spectrum overall', save=True, save_fig_name='spectrum_overall_sub_np_0')
        self.K_mean = self.y_luminance.mean()

    def compute_JND(self):
        X_JND_frequency = []
        Y_JND = []
        for frequency_index in range(self.N_s):
            frequency = self.x_freq[frequency_index]
            if frequency == 0:
                continue
            if frequency > 120:
                break
            S = TCSF(frequency)
            C_R = 2 * self.K_DFT[frequency_index] / self.K_mean
            JND = S * C_R
            X_JND_frequency.append(frequency)
            Y_JND.append(JND)
        X_JND_frequency = np.array(X_JND_frequency)
        Y_JND = np.array(Y_JND)
        self.plot_pict(x_array=X_JND_frequency, y_array=Y_JND, x_label='Frequency', y_label='JND',
                       title='JND_Frequency', save=True, fig_size=(6,3), save_fig_name='JND_Frequency')

        JND_max = Y_JND.max()
        JND_sum = Y_JND.sum()
        JND_combine = np.linalg.norm(Y_JND)
        print('JND Max', JND_max)
        print('JND Sum', JND_sum)
        print('JND_combine', JND_combine)




if __name__ == "__main__":
    Pure_sum_Model = Pure_sum_flicker_visibility()
    Pure_sum_Model.__int__()
    Pure_sum_Model.read_data()
    Pure_sum_Model.run_dft()
    Pure_sum_Model.compute_JND()
