from typing import Any

import numpy as np

from scenario_gym.state import State

from .base import Metric


class Delay(Metric):
    """Record total delay of all vehicles in the scenario"""

    def __init__(self, name="delay"):
        super().__init__(name)
        self.max_speed = 20

    def _reset(self, state: State) -> None:
        self.vehicles = set(state.scenario.vehicles)
        self.delay = {vehicle: 0.0 for vehicle in self.vehicles}
        self.distances = None
        self.ego = state.scenario.ego
        self.prev_dist = 0.0

    def _step(self, state: State) -> None:
        dist = state.distances[self.ego]
        t = state.t
        if dist > 1e-6 and dist - self.prev_dist > 1e-6:
            self.delay = t - dist/self.max_speed
        self.prev_dist = dist

    def get_state(self) -> Any:
        return self.delay

    def set_max_speed(self, speed: float) -> None:
        self.max_speed = speed


class AllHistory(Metric):
    """Record speed and acceleration history of all agents."""
    name = "all_history"

    def _reset(self, state: State) -> None:
        """Reset history"""
        self.vehicles = set(state.scenario.vehicles)
        self.speed_history = {vehicle: [] for vehicle in self.vehicles}
        self.acceleration_history = {vehicle: [] for vehicle in self.vehicles}
        self.last_speed = {
            vehicle: np.linalg.norm(state.velocities[vehicle][:3]) if vehicle in state.velocities.keys() else None for
            vehicle in self.vehicles}
        self.t_history = []

    def _step(self, state: State) -> None:
        """Record speed and acceleration of each vehicle"""

        current_vehicles = set(state.velocities.keys())
        unprocessed_vehicles = self.vehicles - current_vehicles

        for vehicle in current_vehicles:
            speed = np.linalg.norm(state.velocities[vehicle][:3])
            previous_speed = self.last_speed.get(vehicle, None)

            # default acceleration to 0 if a vehicle just appeared in the scenario
            if previous_speed is not None:
                acceleration = (speed - previous_speed) / state.dt
            else:
                acceleration = 0

            self.speed_history.setdefault(vehicle, []).append(speed)
            self.acceleration_history.setdefault(vehicle, []).append(acceleration)

            self.last_speed[vehicle] = speed
            if vehicle in unprocessed_vehicles:
                unprocessed_vehicles.remove(vehicle)

        # for vehicles in the scenario but not in the current time step, set speed and acceleration to None
        for vehicle in unprocessed_vehicles:
            self.speed_history.setdefault(vehicle, []).append(None)
            self.acceleration_history.setdefault(vehicle, []).append(None)

        self.t_history.append(state.t)

    def get_state(self) -> Any:
        # print("Expected time steps:", len(self.t_history))
        #
        # for vehicle in self.vehicles:
        #     print(
        #         f"Vehicle {vehicle}:
        #         Speed Entries: {len(self.speed_history[vehicle])},
        #         Acceleration Entries: {len(self.acceleration_history[vehicle])}")
        return self.speed_history, self.acceleration_history, self.t_history


class EgoSpeedAccelerationHistory(Metric):
    """Record speed and acceleration history of the ego."""

    name = "ego_speed_acceleration_history"

    def _reset(self, state: State) -> None:
        """Reset speed history."""
        self.ego = state.scenario.ego
        self.ego_speed_history = [np.linalg.norm(state.velocities[self.ego][:3])]
        self.t_history = [0.0]
        self.ego_acceleration_history = []

    def _step(self, state: State) -> None:
        """Append new speed and time."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.ego_acceleration_history.append((speed - self.ego_speed_history[-1]) / state.dt)
        self.ego_speed_history.append(speed)
        self.t_history.append(state.t)

    def get_state(self) -> np.ndarray:
        """Return speed history"""
        self.ego_acceleration_history.append(0.0)
        return np.array([self.t_history, self.ego_speed_history, self.ego_acceleration_history])
