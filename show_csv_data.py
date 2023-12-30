import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

fp = open('LuminanceData.csv', 'r')
Luminance_data = csv.reader(fp)

x_time = []
y_luminance = []

for i in tqdm(Luminance_data):
    if Luminance_data.line_num == 1:
        continue
    x_time.append(float(i[0]))
    y_luminance.append(float(i[1]))

plt.figure(figsize=(6,6))
plt.plot(x_time, y_luminance)
plt.title('Luminance_Curve')
plt.xlabel('Times in seconds')
plt.ylabel('Luminance cd/m2')
plt.show()