from scenario_gym import ScenarioGym
from scenario_gym.metrics import AllHistory
from scenario_gym.xosc_interface import import_scenario
import os
import numpy as np
import matplotlib.pyplot as plt



current_path = os.path.dirname(os.path.abspath(__file__))
scenario_name = '9c324146-be03-4d4e-8112-eaf36af15c17.xosc'
file_path = os.path.join(current_path, '../..', 'tests', 'input_files', 'Scenarios', scenario_name)

scenario = import_scenario(file_path)
scenario.plot()

gym = ScenarioGym(metrics=[AllHistory()])
gym.load_scenario(file_path)
gym.rollout()

speed, acceleration, t = gym.get_metrics()['all_history']

# print(speed)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create two subplots: one for speed and one for acceleration


for vehicle, speeds in speed.items():
    axs[0].plot(t[:len(speeds)], speeds, label=f'Vehicle {vehicle}')
axs[0].set_title('Speed History')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Speed (m/s)')
# axs[0].legend()

for vehicle, accelerations in acceleration.items():
    axs[1].plot(t[:len(accelerations)], accelerations, label=f'Vehicle {vehicle}')
axs[1].set_title('Acceleration History')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Acceleration (m/s²)')
# axs[1].legend()

plt.tight_layout()
plt.show()