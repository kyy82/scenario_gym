import numpy as np

from scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario
from scenario_gym.metrics import EgoSpeedHistory
import os
from key_performance_indicators import vehicle_specific_power
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))
scenario_name = 'd9726503-e04a-4e8b-b487-8805ef790c93.xosc'
file_path = os.path.join(current_path, '..', 'tests', 'input_files', 'Scenarios', scenario_name)

scenario = import_scenario(file_path)
plt.figure(1)
scenario.plot()

gym = ScenarioGym(metrics=[EgoSpeedHistory()])
gym.load_scenario(file_path)
gym.rollout()

speed_and_acceleration = gym.get_metrics()['ego_speed_history']
vsp, _ = vehicle_specific_power(speed_and_acceleration)

plt.figure(2)
fig, ax1 = plt.subplots()

# Plot speed and acceleration on the left y-axis
color = 'tab:red'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Speed (m/s)', color=color)
ax1.plot(speed_and_acceleration[0, :], speed_and_acceleration[1, :], label='Speed (m/s)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot VSP on the right y-axis
color = 'tab:blue'
ax2.set_ylabel('VSP (kw/tonne)', color=color)
ax2.plot(speed_and_acceleration[0, :], speed_and_acceleration[2,:], label='Acceleration (m/s^-2)', color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Add a legend
# Since we are using two different axes, we need to manually handle the legend
# Create handles and labels for each line plot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine the handles and labels
handles = handles1 + handles2
labels = labels1 + labels2
# Place the combined legend on the plot
ax1.legend(handles, labels, loc='upper left')

# Show the plot
plt.show()

plt.figure(3)
plt.plot(speed_and_acceleration[0,:], vsp)
plt.xlabel('Time (s)')
plt.ylabel('VSP (kW/tonne)')
plt.show()

# mean = np.mean(vsp)
# std_dev = np.std(vsp)
# threshold = 3
# outliers = [x for x in vsp if x < mean - threshold * std_dev or x > mean + threshold * std_dev]
# indices = np.where(np.isin(vsp, outliers))[0]
# indices = [indices[0]-1, indices[0], indices[1], indices[1]+1]
# print(speed_and_acceleration[1,indices], speed_and_acceleration[2,indices])