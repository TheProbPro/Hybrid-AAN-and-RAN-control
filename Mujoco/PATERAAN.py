import mujoco, mujoco.viewer
import time, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy import interpolate
import pandas as pd
from pathlib import Path

# ---------------------------
# Configuration and System Diagnostics
# ---------------------------
MODEL_PATH = "mergedCopy.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Detailed system diagnostics
print("=== System Detailed Diagnostics ===")
print(f"Model joint count: {model.njnt}")
print(f"Model degrees of freedom: {model.nv}")
print(f"Actuator count: {model.nu}")

# Find all joint information
for i in range(model.njnt):
    jnt_type = model.jnt_type[i]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_adr = model.jnt_qposadr[i]
    dof_adr = model.jnt_dofadr[i] if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE else -1
    print(f"Joint {i} ({jnt_name}): type={jnt_type}, qpos_addr={qpos_adr}, dof_addr={dof_adr}")

# Find elbow joint
joint_name = "el_x"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
if joint_id == -1:
    print(f"Error: Joint '{joint_name}' not found")
    print("Available joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            print(f"  {name}")
    exit()
    
qpos_adr = model.jnt_qposadr[joint_id]
dof_adr = model.jnt_dofadr[joint_id]

print(f"\nTarget joint '{joint_name}':")
print(f"  ID: {joint_id}, qpos address: {qpos_adr}, dof address: {dof_adr}")
print(f"  Joint range: {model.jnt_range[joint_id]} radians")
print(f"  Joint range: {np.degrees(model.jnt_range[joint_id])} degrees")

# Find actuators
print(f"\nActuator configuration:")
actuator_found = False
for i in range(model.nu):
    jnt_id = model.actuator_trnid[i, 0]
    jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  Actuator {i} ({act_name}): controlled joint={jnt_name}, ctrlrange={model.actuator_ctrlrange[i]}, gear={model.actuator_gear[i]}")
    
    if jnt_name == joint_name:
        act_idx = i
        actuator_found = True
        print(f"  -> Found elbow joint actuator: index={i}")

if not actuator_found:
    print("Warning: No dedicated elbow joint actuator found, using first actuator")
    act_idx = 0

# ==================== Physical System Test ====================

def physical_system_test():
    """Test physical system response"""
    print("\n=== Physical System Response Test ===")
    
    data.qpos[qpos_adr] = math.radians(55.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    initial_pos = data.qpos[qpos_adr]
    print(f"Initial position: {math.degrees(initial_pos):.1f}°")
    
    print("\nGravity effect test:")
    for i in range(50):
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            pos_deg = math.degrees(data.qpos[qpos_adr])
            print(f"  Step {i}: position={pos_deg:6.1f}°")
    
    data.qpos[qpos_adr] = math.radians(55.0)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    print("\nTorque response test:")
    test_torques = [-20, -10, -5, 0, 5, 10, 20]
    for torque in test_torques:
        data.ctrl[act_idx] = torque
        positions = []
        for step in range(20):
            mujoco.mj_step(model, data)
            if step % 4 == 0:
                positions.append(math.degrees(data.qpos[qpos_adr]))
        
        print(f"  Torque={torque:3}Nm: position changes {positions}")
        
        data.qpos[qpos_adr] = math.radians(55.0)
        data.qvel[dof_adr] = 0.0
        mujoco.mj_forward(model, data)

physical_system_test()

# ==================== True RAN-Optimized OIAC Controller ====================

class TrueRANOptimizedOIAC:
    
    
    def __init__(self, dof):
        self.DOF = dof
        # Moderate initial impedance
        self.k_mat = np.eye(dof) * 60.0
        self.b_mat = np.eye(dof) * 15.0
        
        # State variables
        self.q = np.zeros((self.DOF, 1))
        self.q_d = np.zeros((self.DOF, 1))
        self.dq = np.zeros((self.DOF, 1))
        self.dq_d = np.zeros((self.DOF, 1))
        
        # Conservative parameters for stability
        self.a = 0.1
        self.b = 0.005
        self.k = 0.2
        
        # Reasonable impedance ranges
        self.k_min = 20.0
        self.k_max = 200.0
        self.b_min = 8.0
        self.b_max = 80.0
        
    def gen_pos_err(self):
        return (self.q - self.q_d)
    
    def gen_vel_err(self):
        return (self.dq - self.dq_d)
    
    def gen_track_err(self):
        return (self.k * self.gen_vel_err() + self.gen_pos_err())
    
    def gen_ad_factor(self):
        track_err_norm = la.norm(self.gen_track_err())
        denominator = max(1.0 + self.b * track_err_norm * track_err_norm, 0.1)
        return self.a / denominator
    
    def update_impedance(self, q, q_d, dq, dq_d, mode='AAN'):
        self.q = np.atleast_2d(np.atleast_1d(q)).T
        self.q_d = np.atleast_2d(np.atleast_1d(q_d)).T
        self.dq = np.atleast_2d(np.atleast_1d(dq)).T
        self.dq_d = np.atleast_2d(np.atleast_1d(dq_d)).T
        
        track_err = self.gen_track_err()
        pos_err = self.gen_pos_err()
        vel_err = self.gen_vel_err()
        ad_factor = max(self.gen_ad_factor(), 0.001)
        
        # Moderate scaling for both modes
        k_scale = 150.0
        b_scale = 100.0
        
        k_update = k_scale * (track_err @ pos_err.T) / ad_factor
        b_update = b_scale * (track_err @ vel_err.T) / ad_factor
        
        # Smooth adaptation
        alpha = 0.9
        self.k_mat = alpha * self.k_mat + (1 - alpha) * np.clip(k_update, self.k_min, self.k_max)
        self.b_mat = alpha * self.b_mat + (1 - alpha) * np.clip(b_update, self.b_min, self.b_max)
        
        return self.k_mat, self.b_mat

# ==================== Enhanced ILC Controller ====================

class EnhancedILC:
    def __init__(self, max_trials=10, target_length=2000):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
        self.reference_time = None
        self.learning_rates = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03]
        
    def update_learning(self, time_array, error_array, torque_array):
        """Enhanced learning update"""
        if self.reference_time is None:
            max_time = min(10.0, max(time_array))
            self.reference_time = np.linspace(0, max_time, len(time_array))
        
        # Align data
        interp_error = interpolate.interp1d(time_array, error_array, bounds_error=False, fill_value=0.0)
        aligned_error = interp_error(self.reference_time)
        
        if not self.learned_feedforward:
            ff = np.zeros_like(aligned_error)
        else:
            lr = self.learning_rates[min(self.current_trial, len(self.learning_rates)-1)]
            ff = self.learned_feedforward[-1] + lr * aligned_error
        
        # Limit feedforward magnitude
        ff = np.clip(ff, -30.0, 30.0)
        
        # Smoothing
        if len(ff) > 10:
            ff = np.convolve(ff, np.ones(7)/7.0, mode='same')
        
        self.learned_feedforward.append(ff)
        self.current_trial += 1
        
        print(f"ILC: Trial {self.current_trial}, Learning rate={self.learning_rates[min(self.current_trial-1, len(self.learning_rates)-1)]:.2f}, Feedforward range[{np.min(ff):.1f}, {np.max(ff):.1f}]")
        
        return ff
    
    def get_feedforward(self, t, trial_idx):
        """Get feedforward torque"""
        if trial_idx < 0 or trial_idx >= len(self.learned_feedforward):
            return 0.0
            
        idx = np.argmin(np.abs(self.reference_time - t))
        if idx < len(self.learned_feedforward[trial_idx]):
            return float(self.learned_feedforward[trial_idx][idx])
        return 0.0

# ==================== Simple Velocity-Based RAN Controller ====================

class SimpleVelocityBasedRANController:
    """
    Simple velocity-based RAN controller:
    - Positive velocity -> AAN mode (assist with ILC)
    - Negative velocity -> RAN mode (resistance, OIAC only)
    """
    def __init__(self, oiac, ilc):
        self.oiac = oiac
        self.ilc = ilc
        self.current_mode = 'AAN'
        
        # Velocity threshold for mode detection (rad/s)
        self.velocity_threshold = 0.0  # Simple threshold at zero
        
        # RAN resistance parameters
        self.ran_resistance_level = 0.05    # Base resistance torque (Nm)
        self.ran_velocity_factor = 0.3     # Velocity-dependent resistance coefficient
        
        # Mode history
        self.mode_history = []
        
        # Previous velocity for filtering
        self.prev_velocity = 0.0
        self.velocity_filter_alpha = 0.8  # Low-pass filter for velocity
        
        # Torque components for analysis
        self.tau_fb_log = []
        self.tau_ff_log = []
        self.tau_resistance_log = []
        self.tau_total_log = []
        
    def get_filtered_velocity(self, current_velocity):
        """Apply low-pass filter to velocity to reduce noise"""
        filtered = self.velocity_filter_alpha * self.prev_velocity + \
                   (1 - self.velocity_filter_alpha) * current_velocity
        self.prev_velocity = filtered
        return filtered
    
    def determine_mode(self, velocity, current_time):
        """
        Determine control mode based on velocity direction:
        - Positive velocity -> AAN mode
        - Negative velocity -> RAN mode
        """
        # Apply simple velocity-based switching
        if velocity > self.velocity_threshold:
            new_mode = 'AAN'  # Positive velocity -> Assist
        elif velocity < self.velocity_threshold:
            new_mode = 'RAN'  # Negative velocity -> Resist
        else:
            # Exactly zero velocity, maintain current mode
            new_mode = self.current_mode
        
        # Detect mode change
        if new_mode != self.current_mode:
            direction = "↑ Positive" if velocity > 0 else "↓ Negative"
            print(f"🔄 {self.current_mode}→{new_mode} at t={current_time:.1f}s - {direction} velocity (vel={math.degrees(velocity):.1f}°/s)")
            self.current_mode = new_mode
        
        return self.current_mode
    
    def compute_control(self, t, q, qdot, trial_idx):
        """Compute control with simple velocity-based mode switching"""
        
        # Filter velocity
        filtered_velocity = self.get_filtered_velocity(qdot)
        
        # Determine mode based on velocity
        current_mode = self.determine_mode(filtered_velocity, t)
        
        # Always use AAN trajectory for desired position
        q_des = target_angle_rad(t)
        dq_des = target_velocity_rad(t)
        error = q_des - q
        
        # Update impedance parameters
        K_mat, B_mat = self.oiac.update_impedance(q, q_des, qdot, dq_des, current_mode)
        
        # Compute base feedback torque
        pos_error_vec = np.array([[error]])
        vel_error_vec = np.array([[dq_des - qdot]])
        tau_fb = float((K_mat @ pos_error_vec + B_mat @ vel_error_vec).item())
        
        # Initialize torque components
        tau_ff = 0.0
        base_resistance = 0.0
        velocity_resistance = 0.0
        
        # Mode-specific control logic
        if current_mode == 'AAN':
            # AAN mode: Assist positive velocity motion with ILC feedforward
            tau_ff = self.ilc.get_feedforward(t, trial_idx-1) if trial_idx > 0 else 0.0
            total_torque = tau_ff + tau_fb
            
        else:  # RAN mode
            # RAN mode: Resist negative velocity motion (OIAC only, no ILC)
            # Calculate resistance torque (opposes negative velocity motion)
            # Since velocity is negative, we want positive resistance
            resistance_direction = 1.0  # Always resist in RAN mode
            base_resistance = self.ran_resistance_level * resistance_direction
            velocity_resistance = self.ran_velocity_factor * abs(qdot) * resistance_direction
            
            # Total RAN torque: OIAC feedback only + resistance
            total_torque = tau_fb + base_resistance + velocity_resistance
        
        # Record torque components for analysis
        self.tau_fb_log.append(tau_fb)
        self.tau_ff_log.append(tau_ff)
        self.tau_resistance_log.append(base_resistance + velocity_resistance)
        self.tau_total_log.append(total_torque)
        
        # Record mode
        self.mode_history.append(current_mode)
        
        return total_torque, q_des, error, current_mode

# ==================== Control Parameters ====================

# Moderate torque limits
TORQUE_MIN = -4.1
TORQUE_MAX = 4.1

# Desired trajectory
INITIAL_ANGLE = 55.0
AMP_DEG = 15.0
FREQ = 0.16

def target_angle_rad(t):
    phase = 2 * np.pi * FREQ * t
    angle_rad = math.radians(INITIAL_ANGLE) + math.radians(AMP_DEG) * math.sin(phase)
    return float(angle_rad)

def target_velocity_rad(t):
    phase = 2 * np.pi * FREQ * t
    return math.radians(AMP_DEG) * 2 * np.pi * FREQ * math.cos(phase)

# ==================== Main Control Loop ====================

print(f"\n=== Starting Simple Velocity-Based RAN Control ===")
print(f"Trajectory Range: {INITIAL_ANGLE-AMP_DEG:.0f}° to {INITIAL_ANGLE+AMP_DEG:.0f}° @ {FREQ}Hz")
print(f"Control Strategy:")
print(f"  • Positive velocity -> AAN Mode (ILC + OIAC)")
print(f"  • Negative velocity -> RAN Mode (OIAC only + Resistance)")
print(f"Torque range: [{TORQUE_MIN}, {TORQUE_MAX}]Nm")
print(f"RAN Resistance: Base={2.5}Nm + {1.5}*|velocity|")

# Instantiate controllers
oiac = TrueRANOptimizedOIAC(dof=1)
ilc = EnhancedILC(max_trials=10)
velocity_controller = SimpleVelocityBasedRANController(oiac, ilc)

all_avg_errors = []
all_max_errors = []
all_k_values = []
all_b_values = []
all_mode_distributions = []

for trial in range(ilc.max_trials):
    print(f"\n=== Trial {trial+1}/{ilc.max_trials} ===")
    
    # Reset state
    data.qpos[qpos_adr] = math.radians(INITIAL_ANGLE)
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)
    
    # Reset controllers
    oiac = TrueRANOptimizedOIAC(dof=1)
    velocity_controller.oiac = oiac
    velocity_controller.current_mode = 'AAN'
    velocity_controller.mode_history = []
    velocity_controller.prev_velocity = 0.0
    velocity_controller.tau_fb_log = []
    velocity_controller.tau_ff_log = []
    velocity_controller.tau_resistance_log = []
    velocity_controller.tau_total_log = []

    time_log = []
    q_log = []
    q_des_log = []
    torque_log = []
    error_log = []
    velocity_log = []
    k_log = []
    b_log = []
    mode_log = []

    last_debug_time = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()  # Initialize t0 right when viewer starts
        
        while viewer.is_running():
            t = time.time() - t0
            if t > 12.0:
                break

            # Get current state
            q = float(data.qpos[qpos_adr])
            qdot = float(data.qvel[dof_adr])
            
            # Compute control
            torque, q_des, error, current_mode = velocity_controller.compute_control(t, q, qdot, trial)
            
            # Saturation
            torque_clipped = max(TORQUE_MIN, min(torque, TORQUE_MAX))
            
            # Apply control
            data.ctrl[act_idx] = torque_clipped

            # Record data
            time_log.append(t)
            q_log.append(q)
            q_des_log.append(q_des)
            torque_log.append(torque_clipped)
            error_log.append(error)
            velocity_log.append(qdot)
            mode_log.append(current_mode)
            k_log.append(float(oiac.k_mat[0, 0]))
            b_log.append(float(oiac.b_mat[0, 0]))

            # Debug output
            if t - last_debug_time > 1.5:
                actual_deg = math.degrees(q)
                desired_deg = math.degrees(q_des)
                error_deg = math.degrees(error)
                velocity_deg = math.degrees(qdot)
                k_val = float(oiac.k_mat[0, 0])
                b_val = float(oiac.b_mat[0, 0])
                
                motion_dir = "↑" if qdot > 0 else "↓" if qdot < 0 else "•"
                mode_info = f"Mode={current_mode} {motion_dir}"
                if current_mode == 'RAN':
                    mode_info += " (Resist NEG)"
                else:
                    mode_info += " (Assist POS)"
                
                print(f"t={t:.1f}s: {mode_info}, Angle={actual_deg:6.1f}°, Desired={desired_deg:6.1f}°, Error={error_deg:5.1f}°, Vel={velocity_deg:5.1f}°/s, Torque={torque_clipped:6.1f}Nm")
                last_debug_time = t

            mujoco.mj_step(model, data)
            viewer.sync()

    # Trial results
    avg_error = np.mean(np.abs(error_log))
    max_error = np.max(np.abs(error_log))
    avg_k = np.mean(k_log)
    avg_b = np.mean(b_log)
    
    # Calculate mode distribution
    aan_count = mode_log.count('AAN')
    ran_count = mode_log.count('RAN')
    total_count = len(mode_log)
    aan_ratio = aan_count / total_count * 100
    ran_ratio = ran_count / total_count * 100
    
    # Calculate motion range
    min_angle = math.degrees(min(q_log))
    max_angle = math.degrees(max(q_log))
    motion_range = max_angle - min_angle
    
    all_avg_errors.append(avg_error)
    all_max_errors.append(max_error)
    all_k_values.append(avg_k)
    all_b_values.append(avg_b)
    all_mode_distributions.append((aan_ratio, ran_ratio))
    
    print(f"Trial results:")
    print(f"  Average error: {math.degrees(avg_error):.2f}°")
    print(f"  Maximum error: {math.degrees(max_error):.2f}°")
    print(f"  Motion range: {min_angle:.1f}° to {max_angle:.1f}° (span: {motion_range:.1f}°)")
    print(f"  Average K: {avg_k:.1f}, Average B: {avg_b:.1f}")
    print(f"  Mode distribution: AAN={aan_ratio:.1f}%, RAN={ran_ratio:.1f}%")

    # Check if we're achieving the desired motion range
    if motion_range < 25.0:
        print(f"⚠️  Warning: Motion range too small ({motion_range:.1f}°), expected ~30°")
        if trial == 0:
            print("   This is normal for first trial - ILC needs learning")
    
    # ILC learning update (only for AAN mode)
    if trial < ilc.max_trials - 1:
        ilc.update_learning(time_log, error_log, torque_log)

# Result analysis

print(f"\n=== FINAL RESULTS ===")
print(f"Completed trials: {len(all_avg_errors)}")
print(f"Final average error: {math.degrees(all_avg_errors[-1]):.2f}°")
print(f"Final maximum error: {math.degrees(all_max_errors[-1]):.2f}°")

# Calculate final motion range
final_min_angle = math.degrees(min(q_log))
final_max_angle = math.degrees(max(q_log))
final_motion_range = final_max_angle - final_min_angle
print(f"Final motion range: {final_min_angle:.1f}° to {final_max_angle:.1f}° (span: {final_motion_range:.1f}°)")

if len(all_avg_errors) > 1:
    improvement = math.degrees(all_avg_errors[0] - all_avg_errors[-1])
    print(f"Error improvement: {improvement:.2f}°")

final_aan_ratio, final_ran_ratio = all_mode_distributions[-1]
print(f"Final mode distribution: AAN={final_aan_ratio:.1f}% (Positive vel), RAN={final_ran_ratio:.1f}% (Negative vel)")

# ==================== DATA EXPORT TO CSV ====================

print(f"\n=== Exporting Final Trial Data to CSV ===")

# Create save directory
SAVE_PATH = Path("./saved_data")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Normalize time to start from 0
if time_log:
    time_offset = time_log[0]
    time_log_normalized = [t - time_offset for t in time_log]
else:
    time_log_normalized = time_log

# Calculate indices for AAN and RAN modes
aan_indices = [i for i, mode in enumerate(mode_log) if mode == 'AAN']
ran_indices = [i for i, mode in enumerate(mode_log) if mode == 'RAN']

# Convert lists to numpy arrays for easier indexing
torque_log_np = np.array(torque_log)
k_log_np = np.array(k_log)
b_log_np = np.array(b_log)
error_deg_log = [math.degrees(e) for e in error_log]
error_deg_log_np = np.array(error_deg_log)
velocity_deg_log = [math.degrees(v) for v in velocity_log]

# Calculate acceleration and jerk
acceleration_log = []
jerk_log = []
for i in range(1, len(velocity_log)):
    dt = time_log[i] - time_log[i-1]
    if dt > 0:
        acceleration = (velocity_log[i] - velocity_log[i-1]) / dt
        acceleration_log.append(acceleration)
        
        if i > 1:
            jerk = (acceleration_log[-1] - acceleration_log[-2]) / dt
            jerk_log.append(jerk)
    else:
        acceleration_log.append(0)
        if i > 1:
            jerk_log.append(0)

# Pad acceleration_log and jerk_log to match time_log length
if len(acceleration_log) < len(time_log):
    acceleration_log = [0] + acceleration_log
if len(jerk_log) < len(time_log):
    jerk_log = [0] * (len(time_log) - len(jerk_log)) + jerk_log

# Convert jerk to degrees
jerk_deg_log = [math.degrees(j) for j in jerk_log]

# Prepare data for CSV export
FILE_NAME = f"Final_Trial_Data.csv"
df = pd.DataFrame({
    'Time_s': time_log_normalized,
    'Control_Mode': mode_log,
    'Actual_Position_deg': [math.degrees(q) for q in q_log],
    'Desired_Position_deg': [math.degrees(q) for q in q_des_log],
    'Position_Error_deg': error_deg_log,
    'Feedback_Torque_Nm': velocity_controller.tau_fb_log,
    'Feedforward_Torque_Nm': velocity_controller.tau_ff_log,
    'Total_Torque_Nm': torque_log,
    'Jerk_deg_per_s3': jerk_deg_log
})

# Save to CSV
csv_path = SAVE_PATH / FILE_NAME
df.to_csv(csv_path, index=False)
print(f"✅ Data exported successfully to: {csv_path}")
print(f"   Total data points: {len(df)}")
print(f"   Columns: {list(df.columns)}")

# ==================== COMPREHENSIVE RESULTS DIAGRAM ====================

print(f"\n=== Generating Comprehensive Results Diagram ===")

# Create a professional results figure
plt.style.use('default')
fig = plt.figure(figsize=(20, 16))

# 1. Main trajectory tracking with mode switching
ax1 = plt.subplot(3, 3, 1)
actual_deg = [math.degrees(q) for q in q_log]
desired_deg = [math.degrees(q) for q in q_des_log]

# Plot with mode coloring
current_mode = mode_log[0]
start_idx = 0
for i, mode in enumerate(mode_log):
    if mode != current_mode:
        color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
        plt.axvspan(time_log[start_idx], time_log[i], alpha=0.3, color=color)
        current_mode = mode
        start_idx = i
# Last segment
color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
plt.axvspan(time_log[start_idx], time_log[-1], alpha=0.3, color=color)

plt.plot(time_log, actual_deg, 'b-', linewidth=2.5, label='Actual Angle', alpha=0.8)
plt.plot(time_log, desired_deg, 'r--', linewidth=2, label='Desired Angle', alpha=0.9)
plt.xlabel('Time (s)')
plt.ylabel('Angle (°)')
plt.title('Simple Velocity-Based RAN Control: Trajectory Tracking\n(Blue=AAN POS, Red=RAN NEG)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Velocity profile with mode indication
ax2 = plt.subplot(3, 3, 2)
plt.plot(time_log, velocity_deg_log, 'purple', linewidth=2, label='Velocity')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
plt.fill_between(time_log, 0, velocity_deg_log, 
                 where=[v > 0 for v in velocity_deg_log], 
                 color='blue', alpha=0.2, label='AAN (Positive)')
plt.fill_between(time_log, 0, velocity_deg_log, 
                 where=[v < 0 for v in velocity_deg_log], 
                 color='red', alpha=0.2, label='RAN (Negative)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (°/s)')
plt.title('Velocity Profile with Mode Indication', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Control Torque Profile - Total Output
ax3 = plt.subplot(3, 3, 3)
plt.plot(time_log, torque_log, 'orange', linewidth=3, label='Total Control Torque', alpha=0.9)

# Plot torque components if available
if len(velocity_controller.tau_fb_log) == len(time_log):
    plt.plot(time_log, velocity_controller.tau_fb_log, 'blue', linewidth=1.5, label='Feedback Torque', alpha=0.7)
    plt.plot(time_log, velocity_controller.tau_ff_log, 'green', linewidth=1.5, label='Feedforward Torque', alpha=0.7)
    plt.plot(time_log, velocity_controller.tau_resistance_log, 'red', linewidth=1.5, label='Resistance Torque', alpha=0.7)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Control Torque (Nm)')
plt.title('Control Torque Profile - Total Output\n(Includes all torque components)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Joint Error
ax4 = plt.subplot(3, 3, 4)
error_deg = [math.degrees(e) for e in error_log]
plt.plot(time_log, error_deg, 'red', linewidth=2.5, label='Joint Error')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Joint Error (°)')
plt.title('Joint Tracking Error', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Jerk profile
ax5 = plt.subplot(3, 3, 5)
plt.plot(time_log, jerk_deg_log, 'green', linewidth=2.5, label='Jerk')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Jerk (°/s³)')
plt.title('Jerk Profile (Smoothness Indicator)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Impedance adaptation
ax6 = plt.subplot(3, 3, 6)
trials = range(1, len(all_k_values)+1)
plt.plot(trials, all_k_values, 'g^-', linewidth=3, markersize=8, label='Stiffness (K)')
plt.plot(trials, all_b_values, 'mv-', linewidth=3, markersize=8, label='Damping (B)')
plt.xlabel('Trial Number')
plt.ylabel('Impedance Parameters')
plt.title('Adaptive Impedance Parameters', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Performance summary table
ax7 = plt.subplot(3, 3, 7)
ax7.axis('off')

# Calculate performance metrics
final_avg_error = math.degrees(all_avg_errors[-1])
final_max_error = math.degrees(all_max_errors[-1])
final_motion_range = final_max_angle - final_min_angle
final_aan_ratio, final_ran_ratio = all_mode_distributions[-1]
improvement = math.degrees(all_avg_errors[0] - all_avg_errors[-1]) if len(all_avg_errors) > 1 else 0

# Calculate jerk statistics
avg_jerk = np.mean(np.abs(jerk_deg_log)) if jerk_deg_log else 0
max_jerk = np.max(np.abs(jerk_deg_log)) if jerk_deg_log else 0

# Calculate torque statistics
avg_torque = np.mean(np.abs(torque_log)) if torque_log else 0
max_torque = np.max(np.abs(torque_log)) if torque_log else 0

summary_text = (
    "PERFORMANCE SUMMARY\n"
    f"Final Average Error: {final_avg_error:.2f}°\n"
    f"Final Maximum Error: {final_max_error:.2f}°\n"
    f"Motion Range: {final_motion_range:.1f}°\n"
    f"Average Jerk: {avg_jerk:.2f}°/s³\n"
    f"Maximum Jerk: {max_jerk:.2f}°/s³\n"
    f"Average Torque: {avg_torque:.2f}Nm\n"
    f"Maximum Torque: {max_torque:.2f}Nm\n"
    f"Mode Distribution:\n"
    f"  AAN (Positive): {final_aan_ratio:.1f}%\n"
    f"  RAN (Negative): {final_ran_ratio:.1f}%\n"
    f"Error Improvement: {improvement:.2f}°\n\n"
    "SIMPLE VELOCITY CONTROL\n"
    f"Trajectory: {INITIAL_ANGLE-AMP_DEG:.0f}° to {INITIAL_ANGLE+AMP_DEG:.0f}° @ {FREQ}Hz\n"
    "Positive vel → AAN (Assist)\n"
    "Negative vel → RAN (Resist)\n"
    f"RAN Resistance: {2.5}Nm + {1.5}*|vel|"
)

ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, fontsize=11, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# 8. Torque Components Analysis
ax8 = plt.subplot(3, 3, 8)
if len(velocity_controller.tau_fb_log) > 0:
    # Calculate average torque components by mode
    aan_tau_fb = np.mean([velocity_controller.tau_fb_log[i] for i in aan_indices]) if aan_indices else 0
    aan_tau_ff = np.mean([velocity_controller.tau_ff_log[i] for i in aan_indices]) if aan_indices else 0
    ran_tau_fb = np.mean([velocity_controller.tau_fb_log[i] for i in ran_indices]) if ran_indices else 0
    ran_tau_resistance = np.mean([velocity_controller.tau_resistance_log[i] for i in ran_indices]) if ran_indices else 0
    
    components = ['Feedback', 'Feedforward', 'Resistance']
    aan_values = [aan_tau_fb, aan_tau_ff, 0]
    ran_values = [ran_tau_fb, 0, ran_tau_resistance]
    
    x = np.arange(len(components))
    width = 0.35
    
    ax8.bar(x - width/2, aan_values, width, label='AAN Mode', color='blue', alpha=0.7)
    ax8.bar(x + width/2, ran_values, width, label='RAN Mode', color='red', alpha=0.7)
    
    ax8.set_xlabel('Torque Components')
    ax8.set_ylabel('Average Torque (Nm)')
    ax8.set_title('Average Torque Components by Mode', fontweight='bold', fontsize=12)
    ax8.set_xticks(x)
    ax8.set_xticklabels(components, rotation=45)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
else:
    ax8.text(0.5, 0.5, 'No torque component data available', 
             horizontalalignment='center', verticalalignment='center',
             transform=ax8.transAxes, fontsize=12)
    ax8.set_title('Torque Components Analysis', fontweight='bold', fontsize=12)

# 9. Joint Error Histogram
ax9 = plt.subplot(3, 3, 9)
error_deg = [math.degrees(e) for e in error_log]
plt.hist(error_deg, bins=50, color='red', alpha=0.7, edgecolor='black')
plt.xlabel('Joint Error (°)')
plt.ylabel('Frequency')
plt.title('Joint Error Distribution', fontweight='bold', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=3.0)
plt.show()

# ==================== KEY PERFORMANCE INDICATORS ====================

print(f"\n=== KEY PERFORMANCE INDICATORS ===")
print(f"1. TRACKING PERFORMANCE")
print(f"   • Final Average Error: {final_avg_error:.2f}°")
print(f"   • Final Maximum Error: {final_max_error:.2f}°")
print(f"   • Error Improvement: {improvement:.2f}°")

print(f"\n2. MOTION CHARACTERISTICS")
print(f"   • Motion Range: {final_min_angle:.1f}° to {final_max_angle:.1f}°")
print(f"   • Range Span: {final_motion_range:.1f}°")
print(f"   • Target Range: {INITIAL_ANGLE-AMP_DEG:.0f}° to {INITIAL_ANGLE+AMP_DEG:.0f}°")

print(f"\n3. CONTROL TORQUE ANALYSIS")
print(f"   • Average Control Torque: {avg_torque:.2f}Nm")
print(f"   • Maximum Control Torque: {max_torque:.2f}Nm")
print(f"   • Torque Range: [{TORQUE_MIN}, {TORQUE_MAX}]Nm")

print(f"\n4. SMOOTHNESS ANALYSIS")
print(f"   • Average Jerk: {avg_jerk:.2f}°/s³")
print(f"   • Maximum Jerk: {max_jerk:.2f}°/s³")
print(f"   • Jerk indicates motion smoothness (lower is better)")

print(f"\n5. VELOCITY-BASED MODE SWITCHING")
print(f"   • Final Mode Distribution:")
print(f"     - AAN (Positive vel): {final_aan_ratio:.1f}%")
print(f"     - RAN (Negative vel): {final_ran_ratio:.1f}%")
print(f"   • Switching Strategy: Simple velocity-based")
print(f"   • Velocity Threshold: 0°/s")

print(f"\n6. CONTROL PARAMETERS")
print(f"   • Final Average Stiffness: {all_k_values[-1]:.1f}")
print(f"   • Final Average Damping: {all_b_values[-1]:.1f}")
print(f"   • RAN Resistance: Base={velocity_controller.ran_resistance_level}Nm + {velocity_controller.ran_velocity_factor}*|velocity|")

print(f"\n7. SYSTEM PERFORMANCE")
if final_avg_error < 5.0 and final_motion_range > 25.0 and avg_jerk < 100.0:
    print(f"   ✅ EXCELLENT: Good tracking + full motion range + smooth motion")
elif final_avg_error < 8.0 and final_motion_range > 20.0 and avg_jerk < 200.0:
    print(f"   ⚠️  GOOD: Acceptable performance")
else:
    print(f"   ❌ NEEDS IMPROVEMENT: Consider tuning parameters")

# ==================== MODE TRANSITION ANALYSIS ====================

print(f"\n=== VELOCITY-BASED MODE TRANSITION ANALYSIS ===")
if mode_log and time_log:
    transitions = []
    current_mode = mode_log[0]
    start_time = time_log[0]
    
    for i, (t, mode) in enumerate(zip(time_log, mode_log)):
        if mode != current_mode:
            duration = t - start_time
            velocity_at_switch = math.degrees(velocity_log[i])
            torque_at_switch = torque_log[i]
            transitions.append((current_mode, duration, start_time, t, velocity_at_switch, torque_at_switch))
            current_mode = mode
            start_time = t
    
    # Add the last segment
    if time_log:
        duration = time_log[-1] - start_time
        velocity_at_end = math.degrees(velocity_log[-1])
        torque_at_end = torque_log[-1]
        transitions.append((current_mode, duration, start_time, time_log[-1], velocity_at_end, torque_at_end))
    
    print(f"Mode segments found: {len(transitions)}")
    for i, (mode, duration, start, end, velocity, torque) in enumerate(transitions):
        motion_type = "↑ Positive" if mode == 'AAN' else "↓ Negative"
        print(f"  Segment {i+1}: {mode} mode {motion_type}, Duration={duration:.1f}s, Vel={velocity:.1f}°/s, Torque={torque:.2f}Nm")

# Create a mode transition timeline
if mode_log and time_log:
    plt.figure(figsize=(14, 6))
    
    # Create two subplots
    ax1 = plt.subplot(2, 1, 1)
    for i, (mode, duration, start, end, velocity, torque) in enumerate(transitions):
        color = 'blue' if mode == 'AAN' else 'red'
        motion_symbol = '↑' if mode == 'AAN' else '↓'
        label = f'{mode} {motion_symbol}'
        plt.axhspan(0, 1, start, end, color=color, alpha=0.6)
        plt.text((start+end)/2, 0.5, f'{label}\n{duration:.1f}s', 
                 ha='center', va='center', fontweight='bold', fontsize=10)
    
    plt.yticks([])
    plt.xlabel('Time (s)')
    plt.title('Simple Velocity-Based Mode Transition Timeline (Blue=AAN↑ POS, Red=RAN↓ NEG)', fontweight='bold')
    plt.xlim(0, max(time_log) if time_log else 10)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add torque plot below for reference
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(time_log, torque_log, 'orange', linewidth=2, label='Total Control Torque')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Torque (Nm)')
    plt.title('Total Control Torque Output', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(time_log) if time_log else 10)
    
    plt.tight_layout()
    plt.show()

print(f"\n=== Control Strategy Summary ===")
print(f"✓ Positive velocity → AAN mode: Assist with ILC feedforward")
print(f"✓ Negative velocity → RAN mode: Resist with OIAC only")
print(f"✓ Simple velocity-based switching at zero threshold")
print(f"✓ Adaptive impedance parameters for both modes")
print(f"✓ Total control torque output includes all components")
print(f"✓ Joint error and jerk analysis for performance evaluation")
print(f"\nExperiment completed successfully!")

# ==================== COMPREHENSIVE RESULTS DIAGRAM (Vertical Layout) ====================

print(f"\n=== Generating Comprehensive Results Diagram (Vertical Layout) ===")

# Create a professional results figure with 4 subplots in vertical layout
plt.style.use('default')
fig = plt.figure(figsize=(16, 12))

# 1. Main trajectory tracking with mode switching - TOP
ax1 = plt.subplot(4, 1, 1)
actual_deg = [math.degrees(q) for q in q_log]
desired_deg = [math.degrees(q) for q in q_des_log]

# Plot with mode coloring
current_mode = mode_log[0]
start_idx = 0
for i, mode in enumerate(mode_log):
    if mode != current_mode:
        color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
        plt.axvspan(time_log[start_idx], time_log[i], alpha=0.3, color=color)
        current_mode = mode
        start_idx = i
# Last segment
color = 'lightblue' if current_mode == 'AAN' else 'lightcoral'
plt.axvspan(time_log[start_idx], time_log[-1], alpha=0.3, color=color)

plt.plot(time_log, actual_deg, 'b-', linewidth=2.5, label='Actual Angle', alpha=0.8)
plt.plot(time_log, desired_deg, 'r--', linewidth=2, label='Desired Angle', alpha=0.9)
plt.ylabel('Angle (°)')
plt.title('Simple Velocity-Based RAN Control: Trajectory Tracking\n(Blue=AAN POS, Red=RAN NEG)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks([])  # Remove x-axis labels for this subplot

# 2. Joint Error - SECOND
ax2 = plt.subplot(4, 1, 2)
error_deg = [math.degrees(e) for e in error_log]
plt.plot(time_log, error_deg, 'red', linewidth=2.5, label='Joint Error')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.ylabel('Joint Error (°)')
plt.title('Joint Tracking Error', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks([])  # Remove x-axis labels for this subplot

# 3. Control Torque Profile - THIRD
ax3 = plt.subplot(4, 1, 3)
plt.plot(time_log, torque_log, 'orange', linewidth=3, label='Total Control Torque', alpha=0.9)

# Plot torque components if available
if len(velocity_controller.tau_fb_log) == len(time_log):
    plt.plot(time_log, velocity_controller.tau_fb_log, 'blue', linewidth=1.5, label='Feedback Torque', alpha=0.7)
    plt.plot(time_log, velocity_controller.tau_ff_log, 'green', linewidth=1.5, label='Feedforward Torque', alpha=0.7)
    plt.plot(time_log, velocity_controller.tau_resistance_log, 'red', linewidth=1.5, label='Resistance Torque', alpha=0.7)

plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.ylabel('Control Torque (Nm)')
plt.title('Control Torque Profile - Total Output\n(Includes all torque components)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks([])  # Remove x-axis labels for this subplot

# 4. Jerk profile - BOTTOM
ax4 = plt.subplot(4, 1, 4)
plt.plot(time_log, jerk_deg_log, 'green', linewidth=2.5, label='Jerk')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Jerk (°/s³)')
plt.title('Jerk Profile (Smoothness Indicator)', fontweight='bold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout(pad=3.0)
plt.show()