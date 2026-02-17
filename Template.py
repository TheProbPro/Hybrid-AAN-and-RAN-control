from PATERAAN import PATE_RAAN
from DIARAAN import DIA_RAAN
import time

###############################################
#           Template of control loop          #
###############################################

if __name__ == "__main__":
    # Initialize you motors here

    # Initialize the controller
    trials = 10
    duration = 10.0
    control_loop_frequency = 166.7
    controller = DIA_RAAN(ILC_trials=trials, trial_duration=duration, operational_frequency=control_loop_frequency, DOF=1, Motor_torque=4.1, Hysteresis_thresholds=(-0.1, 0.1), Hysteresis_dwell_time=0.01)
    #controller = PATE_RAAN(ILC_trials=trials, trial_duration=duration, operational_frequency=control_loop_frequency, DOF=1, Motor_torque=4.1, Hysteresis_thresholds=(-0.1, 0.1), Hysteresis_dwell_time=0.01)

    # Teach ILC
    for trial in range(trials):
        tracking_errors = []
        start_time = time.time()
        while time.time() - start_time < duration:
            # Read sensors to get current state
            current_angle = 0.0 # Replace with actual sensor reading
            desired_angle = 0.0 # Replace with trajectory generation logic
            current_velocity = 0.0 # Replace with actual sensor reading
            desired_velocity = 0.0 # Replace with trajectory generation logic
            current_time = time.time() - start_time

            # Update control mode and get torque command
            torque_command = controller.update_control_mode(last_torque=0.0, current_angle=current_angle, desired_angle=desired_angle, current_velocity=current_velocity, desired_velocity=desired_velocity, current_time=current_time)

            # Apply torque command to motors here

            # Calculate tracking error for analysis
            tracking_error = desired_angle - current_angle
            tracking_errors.append(tracking_error)
        # teach ILC at the end of each trial
        controller.teach_ILC(tracking_errors)

    # After ILC is taught, you can run the controller in a loop without teaching ILC to see the performance of the AAN/RAN switching
    start_time = time.time()
    while time.time() - start_time < duration:
        # Read sensors to get current state
        current_angle = 0.0 # Replace with actual sensor reading
        desired_angle = 0.0 # Replace with trajectory generation logic
        current_velocity = 0.0 # Replace with actual sensor reading
        desired_velocity = 0.0 # Replace with trajectory generation logic
        current_time = time.time() - start_time

        # Update control mode and get torque command
        torque_command = controller.update_control_mode(last_torque=0.0, current_angle=current_angle, desired_angle=desired_angle, current_velocity=current_velocity, desired_velocity=desired_velocity, current_time=current_time)

        # Apply torque command to motors here