import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# Read in the data
path = r'output_data/'
# make a dataframe for each file
file_names = os.listdir(path)
dataframes = {}
for file_name in file_names:
    if file_name.endswith(".csv"):
        dataframes[file_name[:-4]] = pd.read_csv(path + file_name)

# print(dataframes["RMSE_pose0_heading1_LIVE_DEMO"])

# Plot the data
# Pose RMSE
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["__time"]
   - dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["__time"][0], dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["/RMSE/data.0"])
ax.set_title("Pose RMSE")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RMSE (m)")
ax.grid()
fig.savefig("output_data/pose_RMSE.eps", format="eps")
plt.show()

# Heading RMSE
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["__time"] 
    - dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["__time"][0], dataframes["RMSE_pose0_heading1_LIVE_DEMO"]["/RMSE/data.1"])
ax.set_title("Heading RMSE")
ax.set_xlabel("Time (s)")
ax.set_ylabel("RMSE (rad)")
ax.grid()
fig.savefig("output_data/heading_RMSE.eps", format="eps")
plt.show()

