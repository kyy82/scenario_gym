from scenario_gym import ScenarioGym
from scenario_gym.xosc_interface import import_scenario
from scenario_gym.metrics import EgoSpeedHistory
import os

current_path = os.path.dirname(os.path.abspath(__file__))
scenario_name = '5c5188e0-715a-4dd2-a6b2-b3c96b52d608.xosc'
file_path = os.path.join(current_path, '..', 'tests', 'input_files', 'Scenarios', scenario_name)



scenario = import_scenario(file_path)
scenario.plot()

gym = ScenarioGym(metrics=[EgoSpeedHistory()])
gym.load_scenario(file_path)
gym.rollout()


print(gym.get_metrics()['ego_speed_history'])