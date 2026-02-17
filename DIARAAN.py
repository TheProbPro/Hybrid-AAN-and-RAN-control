from HysteresisThreshold import Hysteresis
from OIAC import ada_imp_con
from ILC import ILC
from ControlMode import ControlMode
import numpy as np

#################################################################################
#                       DIA-RAAN controller implementation                      #
# This class implements the DIA-RAAN controller, which is a torque-sign-based   #
# AAN/RAN switching controller as described in Section 2.5.2 of the paper:      #
# "Myoelectric and Hybrid AAN - RAN control ofexoskeleton arm".                 #
# Written by Zi C. Wang, Victor B. Nielsen, and Xiao F. Xiong.                  # 
#################################################################################

class DIA_RAAN():
    """
    DIA-RAAN - Implements AAN/RAN switching logic based on torque sign, as described in Section 2.5.2 of the paper. 
    This controller uses the sign of the applied torque to determine whether to be in AAN or RAN mode, with hysteresis to prevent rapid switching.
    Torque-sign-based AAN/RAN controller
    - Positive torque: AAN mode (assistance)
    - Negative torque: RAN mode (resistance)
    Functionality:'
    - update_control_mode(): Called at each control loop iteration to determine current mode based on the sign of the applied torque and hysteresis. Returns the current control mode.
    - manual_switch_mode(): Allows manual switching between AAN and RAN modes for testing purposes.
    - reset(): Resets the controller state to initial conditions.
    - teach_ILC(): After each ILC trial, this function can be called with the trial's error history to update the ILC feedforward for the next trial.
    """
    def __init__(self, ILC_trials = 10, trial_duration = 10.0, operational_frequency = 166.7, DOF = 1, Motor_torque = 4.1, Hysteresis_thresholds = (-0.1, 0.1), Hysteresis_dwell_time = 0.01):
        """
        :param ILC_trials: Number of ILC trials to perform before relying solely on AAN/RAN switching
        :param trial_duration: Duration of each ILC trial in seconds
        :param operational_frequency: Control loop frequency in Hz
        :param DOF: Degrees of freedom for the controller (assumed 1 for simplicity, but depends on hardware)
        :param Motor_torque: Maximum torque the motor can apply in Nm
        :param Hysteresis_thresholds: Tuple of (low_threshold, high_threshold) for switching conditions. can be tuned to adjust the hysteresis width.
        :param Hysteresis_dwell_time: Description
        """
        # Mode switching parameters
        self.current_mode = ControlMode.AAN  # Initial mode
        self.mode_history = []
        self.Max_Motor_Torque = Motor_torque
        self.Required_ILC_Trials = ILC_trials
        self.current_ILC_trial = 0
        self.ILC_trail_duration = trial_duration
        self.Operational_Frequency = operational_frequency
        
        # RAN resistance parameters
        self.ran_resistance_level = 2.5  # Base resistance level
        self.ran_velocity_factor = 1.5   # Velocity-dependent resistance
        
        # Switching timing parameters
        self.last_switch_time = 0
        self.min_switch_interval = 0.6   # Minimum switching interval

        # Initialize hysteresis and controllers
        self.hysteresis = Hysteresis(low_thresh=Hysteresis_thresholds[0], high_thresh=Hysteresis_thresholds[1], on_dwell_s=Hysteresis_dwell_time, off_dwell_s=Hysteresis_dwell_time, initial_state=False)
        self.OIAC = ada_imp_con(dof=DOF) # Assuming DOF for simplicity
        self.ILC = ILC()


    def update_control_mode(self, last_torque: float, current_angle: float, desired_angle: float, current_velocity: float, desired_velocity: float, current_time: float):
        """
        Updates control mode based on tracking error threshold. If ILC isnt taugt, the controller will not change control modes. After ILC is taugt, 
        the controller will switch between AAN and RAN based on the switching conditions, 
        and return the total torque that should be applied to the motors.

        Parameters:
            last_torque: the torque applied in the last control loop iteration (used for switching conditions)
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

        can_switch = (current_time - self.last_switch_time) >= self.min_switch_interval
        old_mode = self.current_mode
        if not self.hysteresis.update(last_torque, current_time=None):#torque < 0:
            if self.current_mode != ControlMode.AAN and can_switch:
                self.current_mode = ControlMode.AAN
                self.last_switch_time = current_time
                print(f"Switched to AAN mode at t={current_time:.2f}s")
        else:
            if self.current_mode != ControlMode.RAN and can_switch:
                self.current_mode = ControlMode.RAN
                self.last_switch_time = current_time
                print(f"Switched to RAN mode at t={current_time:.2f}s")

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

    def reset(self):
        """Reset controller state"""
        self.current_mode = ControlMode.AAN
        self.mode_history.clear()
        self.last_switch_time = 0
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