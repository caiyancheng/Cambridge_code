import csv
import matplotlib.pyplot as plt
import numpy as np

class basic_flicker_visibility:
    '''
    In it, I divide the entire function into three sections for review, which are the 120Hz section from 0 to 0.09s,
    the transition section from 120Hz to 24Hz from 0.09 to 0.19s, and the 24Hz section from 0.19 to 0.28s.
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

    def plot_pict(self, x_array, y_array, x_label, y_label, title, save=False, save_fig_name='no name'):
        plt.plot(x_array, y_array)
        plt.title(label=title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if save:
            plt.savefig(f'{save_fig_name}.png')
            plt.close()
        else:
            plt.show()

    def run(self):
        t_s = (self.x_time[1:] - self.x_time[:-1]).mean()
        w_s = 1 / t_s

        ## 0 ~ 0.9s ---- 120Hz
        x_times_1 = self.x_time[self.x_time < 0.09]
        N_s_1 = len(x_times_1)
        K_DFT_1 = np.abs(np.fft.fft(self.y_luminance[0:N_s_1], n=N_s_1, axis=-1, norm=None)) / N_s_1  # (N_s ** 0.5) The algorithm in the book conflicts with the example.
        # We believe that the example conforms to the standard DFT, so we change the algorithm in the book. See the report for details.
        x_freq_1 = np.arange(0, N_s_1) * w_s / N_s_1
        self.plot_pict(x_array=x_freq_1, y_array=K_DFT_1, x_label='Frequency', y_label='Luminance ($cd / m^2$)',
                       title='spectrum 1 ---- 120 Hz', save=True, save_fig_name='spectrum_1_120Hz')

        ## 0.9 ~ 1.9s ---- 120->24 Hz
        x_times_2 = self.x_time[(self.x_time >= 0.09) & (self.x_time < 0.19)] # 100ms specifically marked in the studentship document
        N_s_2 = len(x_times_2)
        K_DFT_2 = np.abs(np.fft.fft(self.y_luminance[N_s_1:N_s_1+N_s_2], n=N_s_2, axis=-1, norm=None)) / N_s_2
        x_freq_2 = np.arange(0, N_s_2) * w_s / N_s_2
        self.plot_pict(x_array=x_freq_2, y_array=K_DFT_2, x_label='Frequency', y_label='Luminance ($cd / m^2$)',
                       title='spectrum 2 ---- 120 $\longrightarrow $ 24 Hz', save=True, save_fig_name='spectrum_2_120_24Hz')

        ## 1.9 ~ end ---- 24 Hz
        x_times_3 = self.x_time[self.x_time >= 0.19]
        N_s_3 = len(x_times_3)
        K_DFT_3 = np.abs(np.fft.fft(self.y_luminance[N_s_1+N_s_2:], n=N_s_3, axis=-1, norm=None)) / N_s_3
        x_freq_3 = np.arange(0, N_s_3) * w_s / N_s_3
        self.plot_pict(x_array=x_freq_3, y_array=K_DFT_3, x_label='Frequency', y_label='Luminance ($cd / m^2$)',
                       title='spectrum 3 ---- 24 Hz', save=True, save_fig_name='spectrum_3_24Hz')


if __name__ == "__main__":
    Basic_Model = basic_flicker_visibility()
    Basic_Model.__int__()
    Basic_Model.read_data()
    Basic_Model.run()

    #There are 2 disadvantages:
    # (1) I don't know which section represents 120Hz and which section represents 24Hz
    # (2) There are a lot of clutter, and the intensity is not small, the method in the book is not suitable