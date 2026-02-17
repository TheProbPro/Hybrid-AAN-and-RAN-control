from HysteresisThreshold import Hysteresis
from OIAC import ada_imp_con
from ILC import ILC
from ControlMode import ControlMode
import numpy as np
import math

class PATE_RAAN():
    """
    PATE-RAAN – Implements AAN/RAN switching logic based on Section 2.5.1 of the paper

    Switching conditions:
    1. AAN -> RAN: The user can stably track the target (error < threshold for N consecutive seconds)
    2. RAN -> AAN: The user performs poorly in RAN mode (insufficient motion amplitude or excessive error)

    Functionality:
    - update_mode(): Called at each control loop iteration to determine current mode and compute control output. Returns the torque to apply based on the current mode and tracking performance. This Function will not
    change contolmode before the required number of ILC trials have been completed. During the ILC trial period, it will only update the OIAC and ILC controllers without switching modes.
    - manual_switch_mode(): Allows manual switching between AAN and RAN modes for testing purposes.
    - reset(): Resets the controller state to initial conditions.
    - teach_ILC(): After each ILC trial, this function can be called with the trial's error history to update the ILC feedforward for the next trial.
    """
    def __init__(self, ILC_trials = 10, trial_duration = 10.0, operational_frequency = 166.7, DOF = 1, Motor_torque = 4.1, Hysteresis_thresholds = (math.radians(5.6), math.radians(5.7)), Hysteresis_dwell_time = 0.01):
        """
        :param ILC_trials: Number of ILC trials to perform before relying solely on AAN/RAN switching
        :param trial_duration: Duration of each ILC trial in seconds
        :param operational_frequency: Control loop frequency in Hz
        :param DOF: Degrees of freedom for the controller (assumed 1 for simplicity, but depends on hardware)
        :param Motor_torque: Maximum torque the motor can apply in Nm
        :param Hysteresis_thresholds: Tuple of (low_threshold, high_threshold) in radians for switching conditions. Either the high or low threshold should be the 5.7 degrees as explained in the paper, the other can be tuned to adjust the hysteresis width.
        :param Hysteresis_dwell_time: Description
        """
        self.current_mode = ControlMode.AAN  # Start in AAN mode by default
        self.mode_history = []
        self.Max_Motor_Torque = Motor_torque
        self.Required_ILC_Trials = ILC_trials
        self.current_ILC_trial = 0
        self.ILC_trail_duration = trial_duration
        self.Operational_Frequency = operational_frequency
        
        # Switching condition parameters
        self.aan_to_ran_error_threshold = Hysteresis_thresholds[1]  # 5.7-degree error threshold
        self.aan_to_ran_stable_time = 1.0  # Require 1 second of stable performance
        self.ran_to_aan_motion_threshold = math.radians(10.0)  # Minimum motion amplitude in RAN mode
        self.ran_to_aan_error_threshold = Hysteresis_thresholds[1]  # Maximum allowable error in RAN mode
        
        # State tracking
        self.stable_tracking_start_time = None
        self.ran_motion_range_history = []
        self.ran_error_history = []

        # Initialize hysteresis and controllers
        self.hysteresis = Hysteresis(low_thresh=Hysteresis_thresholds[0], high_thresh=Hysteresis_thresholds[1], on_dwell_s=Hysteresis_dwell_time, off_dwell_s=Hysteresis_dwell_time, initial_state=False)
        self.OIAC = ada_imp_con(dof=DOF) # Assuming DOF for simplicity
        self.ILC = ILC()
        
    def update_mode(self, position_error: float, current_angle: float, desired_angle: float, current_velocity: float, desired_velocity: float, current_time: float):
        """
        Updates control mode based on tracking error threshold. If ILC isnt taugt, the controller will not change control modes. After ILC is taugt, 
        the controller will switch between AAN and RAN based on the switching conditions, 
        and return the total torque that should be applied to the motors.

        Parameters:
            position_error: current position error (rad)
            current_angle: current joint angle (rad)
            desired_angle: desired joint angle (rad)
            current_velocity: current joint velocity (rad/s)
            desired_velocity: desired joint velocity (rad/s)
            current_time: current time (s)

        Returns:
            total_torque: torque to apply to the motors based on current mode and tracking performance (Nm)
        """
        if self.current_ILC_trial < self.Required_ILC_Trials:
            self.OIAC.update_impedance(current_angle, desired_angle, current_velocity, desired_velocity)
            fb_torque = self.OIAC.calc_tau_fb()[0,0] # Assuming single DOF for simplicity
            
            ff_torque = 0.0
            if self.current_ILC_trial > 0:
                ff_torque = self.ILC.get_feedforward()
            
            total_torque = fb_torque + ff_torque
            # Clip total torque to max motor torque
            total_torque = np.clip(total_torque, -self.Max_Motor_Torque, self.Max_Motor_Torque)

            return total_torque

        old_mode = self.current_mode
        
        if self.current_mode == ControlMode.AAN:
            # Check AAN -> RAN condition
            if not self.hysteresis.update(abs(position_error), current_time=None):#abs(position_error) < self.aan_to_ran_error_threshold:
                if self.stable_tracking_start_time is None:
                    self.stable_tracking_start_time = current_time
                elif (current_time - self.stable_tracking_start_time) > self.aan_to_ran_stable_time:
                    self.current_mode = ControlMode.RAN
                    self.stable_tracking_start_time = None
                    print(f"\n{'='*60}")
                    print("MODE SWITCH: AAN -> RAN")
                    print("User has demonstrated stable tracking ability")
                    print(f"{'='*60}\n")
            else:
                self.stable_tracking_start_time = None
                
        elif self.current_mode == ControlMode.RAN:
            # Check RAN -> AAN condition
            motion_range = abs(current_angle - math.radians(55.0))  # Relative to neutral position
            self.ran_motion_range_history.append(motion_range)
            self.ran_error_history.append(abs(position_error))
            
            # Keep only the most recent 5 seconds of history
            if len(self.ran_motion_range_history) > 100:  # Assuming 50 Hz control frequency
                self.ran_motion_range_history.pop(0)
                self.ran_error_history.pop(0)
            
            # Check whether to switch back to AAN
            if len(self.ran_motion_range_history) > 50:
                avg_motion = np.mean(self.ran_motion_range_history[-50:])
                avg_error = np.mean(self.ran_error_history[-50:])
                
                # Insufficient motion amplitude or excessive error
                if (avg_motion < self.ran_to_aan_motion_threshold or 
                    self.hysteresis.update(avg_error, current_time=None)):#avg_error > self.ran_to_aan_error_threshold):
                    self.current_mode = ControlMode.AAN
                    self.ran_motion_range_history.clear()
                    self.ran_error_history.clear()
                    print(f"\n{'='*60}")
                    print("MODE SWITCH: RAN -> AAN")
                    print(f"Avg motion: {math.degrees(avg_motion):.1f}°, "
                          f"Avg error: {math.degrees(avg_error):.1f}°, "
                          f"Current position error: {math.degrees(position_error):.1f}°")
                    print("User needs more assistance")
                    print(f"{'='*60}\n")
        
        mode_changed = (old_mode != self.current_mode)
        if mode_changed:
            self.mode_history.append({
                'time': current_time,
                'from': old_mode,
                'to': self.current_mode
            })

        if self.current_mode == ControlMode.AAN:
            self.OIAC.update_impedance(current_angle, desired_angle, current_velocity, desired_velocity)
            fb_torque = self.OIAC.calc_tau_fb()[0,0] # Assuming single DOF for simplicity
            
            ff_torque = 0.0
            if self.current_ILC_trial > 0:
                ff_torque = self.ILC.get_feedforward()
            
            total_torque = fb_torque + ff_torque
            # Clip total torque to max motor torque
            total_torque = np.clip(total_torque, -self.Max_Motor_Torque, self.Max_Motor_Torque)

            return total_torque
        elif self.current_mode == ControlMode.RAN:
            self.OIAC.update_impedance(current_angle, desired_angle, current_velocity, desired_velocity)
            fb_torque = self.OIAC.calc_tau_fb()[0,0] # Assuming single DOF for simplicity
            fb_torque = np.clip(fb_torque, -self.Max_Motor_Torque, self.Max_Motor_Torque)

            return fb_torque
        else:
            print("Error: Unknown control mode")
            return 0.0
    
    def manual_switch_mode(self):
        """Manually switch control mode"""
        if self.current_mode == ControlMode.AAN:
            self.current_mode = ControlMode.RAN
            print("\nManually switched to RAN mode")
        else:
            self.current_mode = ControlMode.AAN
            print("\nManually switched to AAN mode")
        
        self.stable_tracking_start_time = None
        self.ran_motion_range_history.clear()
        self.ran_error_history.clear()

    def reset(self):
        """Reset controller state"""
        self.current_mode = ControlMode.AAN
        self.mode_history.clear()
        self.stable_tracking_start_time = None
        self.ran_motion_range_history.clear()
        self.ran_error_history.clear()
        print("Reset to AAN mode")

    def teach_ILC(self, trial_error_history: list):
        """
        Once an ILC trial of duration self.ILC_trail_duration is completed, this function can be called with the trial's error history to update the ILC feedforward for the next trial.
        
        :param trial_error_history: A list containing the tracking error at each time step from the previous control loop.
        :type trial_error_history: list
        """
        if len(trial_error_history) < self.ILC_trail_duration * self.Operational_Frequency:
            print("Not enough data to teach ILC for this trial")
            return
        if len(trial_error_history) > self.ILC_trail_duration * self.Operational_Frequency:
            trial_error_history = trial_error_history[:int(self.ILC_trail_duration * self.Operational_Frequency)]
            print("Truncated trial error history to fit ILC trial duration")
        if self.current_ILC_trial < self.Required_ILC_Trials:
            self.ILC.update_learning(trial_error_history)
            self.current_ILC_trial += 1