from scenario_gym import ScenarioGym
from scenario_gym.metrics import AllHistory
from scenario_gym.xosc_interface import import_scenario
import os
import numpy as np



current_path = os.path.dirname(os.path.abspath(__file__))
scenario_name = '9c324146-be03-4d4e-8112-eaf36af15c17.xosc'
file_path = os.path.join(current_path, '../..', 'tests', 'input_files', 'Scenarios', scenario_name)

scenario = import_scenario(file_path)
scenario.plot()

gym = ScenarioGym(metrics=[AllHistory()])
gym.load_scenario(file_path)
gym.rollout()

speed, acceleration, dt = gym.get_metrics()['all_history']

print(speed)
