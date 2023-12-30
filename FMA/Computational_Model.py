import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

fp = open('../LuminanceData.csv', 'r')
Luminance_data = csv.reader(fp)

x_time = []
y_luminance = []

for i in Luminance_data:
    if Luminance_data.line_num == 1:
        continue
    x_time.append(float(i[0]))
    y_luminance.append(float(i[1]))

x_time = np.array(x_time)
y_luminance = np.array(y_luminance)
L_max = y_luminance.max()
L_min = y_luminance.min()
FMA = 2*(L_max-L_min) / (L_max+L_min)
print('FMA', FMA)