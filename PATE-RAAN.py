from HysteresisThreshold import Hysteresis
from OIAC import ada_imp_con
from ILC import ILC
from ControlMode import ControlMode
import numpy as np
import math

class PATE_RAAN():
    """
    Mode manager – implements AAN/RAN switching logic based on Figure 9 of the paper

    Switching conditions:
    1. AAN -> RAN: The user can stably track the target (error < threshold for N consecutive seconds)
    2. RAN -> AAN: The user performs poorly in RAN mode (insufficient motion amplitude or excessive error)
    """
    def __init__(self):
        self.current_mode = ControlMode.AAN  # Start in AAN mode by default
        self.mode_history = []
        
        # Switching condition parameters
        self.aan_to_ran_error_threshold = math.radians(5.7)  # 5.7-degree error threshold
        self.aan_to_ran_stable_time = 1.0  # Require 1 second of stable performance
        self.ran_to_aan_motion_threshold = math.radians(10.0)  # Minimum motion amplitude in RAN mode
        self.ran_to_aan_error_threshold = math.radians(5.7)  # Maximum allowable error in RAN mode
        
        # State tracking
        self.stable_tracking_start_time = None
        self.ran_motion_range_history = []
        self.ran_error_history = []

        # Initialize hysteresis and controllers
        self.hysteresis = Hysteresis(low_thresh=math.radians(5.6), high_thresh=math.radians(5.7), on_dwell_s=0.01, off_dwell_s=0.01, initial_state=False)
        self.OIAC = ada_imp_con(dof=1) # Assuming 1 DOF for simplicity
        self.ILC = ILC()
        
    def update_mode(self, position_error, current_angle, desired_angle, current_time):
        """
        Update control mode

        Parameters:
            position_error: current position error (radians)
            current_angle: current joint angle
            desired_angle: desired joint angle
            current_time: current time

        Returns:
            mode_changed: whether a mode switch occurred
        """
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
        
        return mode_changed
    
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