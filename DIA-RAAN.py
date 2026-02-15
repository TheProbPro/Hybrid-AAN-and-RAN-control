from HysteresisThreshold import Hysteresis
from OIAC import ada_imp_con
from ILC import ILC
from ControlMode import ControlMode
import time

class DIA_RAAN():
    """
    Torque-sign-based AAN/RAN controller
    - Positive torque: AAN mode (assistance)
    - Negative torque: RAN mode (resistance)
    """
    def __init__(self):
        # Mode switching parameters
        self.current_mode = ControlMode.AAN  # Initial mode
        self.last_torque = 0.0
        self.mode_history = []
        
        # RAN resistance parameters
        self.ran_resistance_level = 2.5  # Base resistance level
        self.ran_velocity_factor = 1.5   # Velocity-dependent resistance
        
        # Switching timing parameters
        self.last_switch_time = 0
        self.min_switch_interval = 0.6   # Minimum switching interval

        # Initialize hysteresis and controllers
        self.hysteresis = Hysteresis(low_thresh=-0.1, high_thresh=0.1, on_dwell_s=0.01, off_dwell_s=0.01, initial_state=False)
        self.OIAC = ada_imp_con(dof=1) # Assuming 1 DOF for simplicity
        self.ILC = ILC()


    def update_control_mode(self, torque, t):
        current_time = t
        can_switch = (current_time - self.last_switch_time) >= self.min_switch_interval
        old_mode = self.current_mode
        if not self.hysteresis.update(torque, current_time=None):#torque < 0:
            if self.current_mode != ControlMode.AAN and can_switch:
                self.current_mode = ControlMode.AAN
                self.last_switch_time = current_time
                print(f"Switched to AAN mode at t={t:.2f}s")
        else:
            if self.current_mode != ControlMode.RAN and can_switch:
                self.current_mode = ControlMode.RAN
                self.last_switch_time = current_time
                print(f"Switched to RAN mode at t={t:.2f}s")

        mode_changed = (old_mode != self.current_mode)
        if mode_changed:
            self.mode_history.append({
                'time': current_time,
                'from': old_mode,
                'to': self.current_mode
            })
        
        return self.current_mode
    
    def get_mode_statistics(self, recent_seconds=5):
        """Get mode statistics over a recent time window"""
        if not self.mode_history:
            return 0.0, 0.0
            
        current_time = time.time() if self.mode_history else 0
        cutoff_time = current_time - recent_seconds
        
        recent_history = [mode for (t, mode, _) in self.mode_history if t >= cutoff_time]
        
        if not recent_history:
            return 0.0, 0.0
            
        aan_count = recent_history.count(ControlMode.AAN)
        ran_count = recent_history.count(ControlMode.RAN)
        total_count = len(recent_history)
        
        aan_ratio = aan_count / total_count * 100
        ran_ratio = ran_count / total_count * 100
        
        return aan_ratio, ran_ratio
    
    def reset(self):
        """Reset controller state"""
        self.current_mode = ControlMode.AAN
        self.mode_history.clear()
        self.last_switch_time = 0
        self.last_torque = 0.0
        print("[TorqueBased Controller] Reset to AAN mode")