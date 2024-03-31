from typing import Any

import numpy as np

from scenario_gym.state import State

from .base import Metric


class AllHistory(Metric):
    """Record speed and acceleration history of all agents."""
    name = "all_history"

    def _reset(self, state: State) -> None:
        """Reset history"""
        self.vehicles = set(state.scenario.vehicles)
        self.speed_history = {vehicle: [] for vehicle in self.vehicles}
        self.acceleration_history = {vehicle: [] for vehicle in self.vehicles}
        self.last_velocity = {vehicle: state.velocities[vehicle][:3] if vehicle in state.velocities else None for
                              vehicle in self.vehicles}
        self.dt_history = []

    def _step(self, state: State) -> None:
        """Record speed and acceleration of each agent"""
        for vehicle in state.scenario.vehicles:
            current_velocity = state.velocities[vehicle][:3] if vehicle in state.velocities else np.zeros(3)
            previous_velocity = self.last_velocity.get(vehicle, None)

            speed = np.linalg.norm(current_velocity)
            # default acceleration to 0 if a vehicle just appeared in the scenario
            if previous_velocity is not None:
                acceleration = np.linalg.norm(current_velocity - previous_velocity) / max(state.dt, 1e-5)
            else:
                acceleration = 0

            self.speed_history.setdefault(vehicle, []).append(speed)
            self.acceleration_history.setdefault(vehicle, []).append(acceleration)

            self.last_velocity[vehicle] = current_velocity

        # for vehicles in the scenario but not in the current time step, set speed and acceleration to 0
        for vehicle in self.vehicles:
            if vehicle not in state.velocities.keys():
                self.speed_history.setdefault(vehicle, []).append(0)
                self.acceleration_history.setdefault(vehicle, []).append(0)

        self.dt_history.append(state.dt)

    def get_state(self) -> Any:
        # return self.velocities, self.vehicles
        return self.speed_history, self.acceleration_history, self.dt_history


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


class EgoAvgSpeed(Metric):
    """Measure the average speed of the ego."""

    name = "ego_avg_speed"

    def _reset(self, state: State) -> None:
        """Reset the average speed."""
        self.ego = state.scenario.ego
        self.ego_avg_speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.t = 0.0

    def _step(self, state: State) -> None:
        """Update the average speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        w = self.t / state.t
        self.ego_avg_speed += (1.0 - w) * (speed - self.ego_avg_speed)
        self.t = state.t

    def get_state(self) -> float:
        """Return the current average speed."""
        return self.ego_avg_speed


class EgoMaxSpeed(Metric):
    """Measure the maximum speed of the ego."""

    name = "ego_max_speed"

    def _reset(self, state: State) -> None:
        """Reset the maximum speed."""
        self.ego = state.scenario.ego
        self.ego_max_speed = np.linalg.norm(state.velocities[self.ego][:3])

    def _step(self, state: State) -> None:
        """Update the maximum speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.ego_max_speed = np.maximum(speed, self.ego_max_speed)

    def get_state(self) -> float:
        """Return the current max speed."""
        return self.ego_max_speed


class EgoDistanceTravelled(Metric):
    """Measure the distance travelled by the ego."""

    name = "ego_distance_travelled"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.ego

    def _step(self, state: State) -> None:
        """Pass as entity will update its distance."""
        self.dist = state.distances[self.ego]

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return self.dist
