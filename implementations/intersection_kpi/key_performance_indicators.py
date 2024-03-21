import numpy as np


def vehicle_specific_power(speed_acceleration_history: np.ndarray):
    """
    Computes vehicle specific power (VSP) in kW/tonne of one vehicle for each time step using speed and
    acceleration
    v_and_a_history: 2D 3xN array
    """

    time = speed_acceleration_history[0,:]
    speed = speed_acceleration_history[1,:]
    acceleration = speed_acceleration_history[2,:]

    vsp = 1.1*speed*acceleration + 0.213*speed + 0.000305*speed**3
    return vsp, time


