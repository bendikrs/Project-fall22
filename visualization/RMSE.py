# Read values from csv file and plot them

import matplotlib.pyplot as plt
import numpy as np

def read_RMSE_data(path):
    data = np.genfromtxt(path, delimiter=',', names=['time', 'pose', 'heading', '0'])
    heading = data['heading']
    heading[heading == 0] = np.nan

    data['heading'] = heading
    data['time'] = data['time'] - data['time'][1]
    return data

path1 = 'output_data\RMSE_pose0_heading1_LIVE_DEMO.csv'

data1 = read_RMSE_data(path1)

# Plot RMSE
ax, fig = plt.subplots()
plt.plot(data1['time'], data1['pose'], label='Pose RMSE')
plt.plot(data1['time'], data1['heading'], label='Heading RMSE')
plt.xlabel('Time [s]')
plt.title('RMSE')
plt.legend()
plt.show()

