
"""
README for RAN-Controlled Robotic Arm Simulation
================================================

Project Overview
----------------
This project implements and compares two different control strategies for a robotic arm using MuJoCo physics simulation. 
The controllers implement RAN (Robust Adaptive Nonlinear) control principles with mode switching between AAN (Assist-as-Needed) 
and RAN (Resistance) modes based on different criteria.

Two Control Strategies
----------------------
1. Hysteresis-Based RAN Control (First Code)
   - Switches between AAN and RAN modes based on tracking error with hysteresis
   - Uses error thresholds (5.6°-5.7°) with dwell times for smooth transitions
   - RAN mode provides resistance during trajectory tracking when error is low

2. Direction-Based RAN Control (Second Code)
   - Switches modes based on velocity direction
   - Positive velocity → AAN mode (assistance with ILC)
   - Negative velocity → RAN mode (resistance with OIAC only)
   - Includes data export functionality to CSV

Key Features
------------
- OIAC (Optimized Impedance Adaptive Control): Adaptive impedance parameters (stiffness K, damping B)
- ILC (Iterative Learning Control): Feedforward learning across multiple trials
- Hysteresis-based mode switching: Prevents rapid mode oscillations
- Comprehensive visualization: 9-panel results diagrams and vertical layout plots
- Performance metrics: Error analysis, jerk calculation, torque component analysis
- Data export (Second code): CSV export of all trial data for further analysis

Requirements
------------
See requirements.txt for detailed package versions.

Installation
------------
1. Clone this repository
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure you have the MuJoCo model file mergedCopy.xml in the same directory

Usage
-----
Running the Hysteresis-Based Controller:
   python PATERAAN.py

Running the Velocity-Based Controller:
   python DIARAAN.py

Output
------
Both scripts generate:
- Real-time simulation visualization
- Console output with trial progress and mode transitions
- Comprehensive 9-panel results diagram
- Vertical 4-panel summary diagram
- Mode transition timeline

The velocity-based controller additionally exports:
- CSV file with final trial data in ./saved_data/Final_Trial_Data.csv

Control Parameters
------------------
Trajectory: 55° ±15° @ 0.16Hz
Torque Limits: ±4.1 Nm
RAN Resistance (Hysteresis): Base: 0.004 Nm + 0.003·|vel|
RAN Resistance (Velocity):   Base: 0.05 Nm + 0.3·|vel|
Switching Criteria:
   - Hysteresis: Error-based (5.6°-5.7°)
   - Direction: Direction-based (0° threshold)
Dwell Times (Hysteresis): 0.1s (both directions)

Key Classes
-----------
Controllers:
- TrueRANOptimizedOIAC: Core impedance adaptation controller
- EnhancedILC: Iterative learning controller with progressive learning rates
- TrueRANMultifunctionalController: Hysteresis-based mode switching
- SimpleVelocityBasedRANController: Velocity-based mode switching

Utilities:
- Hysteresis: Threshold-based switching with dwell times

Results Interpretation
----------------------
Performance is evaluated based on:
- Average/Maximum Error: Tracking accuracy
- Motion Range: Achieved trajectory amplitude
- Jerk: Motion smoothness indicator
- Torque Components: Breakdown of control effort
- Mode Distribution: Time spent in AAN vs RAN modes

Performance Indicators
----------------------
EXCELLENT:
   Error < 5.0°, Range > 25.0°, Jerk < 100°/s³

GOOD:
   Error < 8.0°, Range > 20.0°, Jerk < 200°/s³

NEEDS IMPROVEMENT:
   Otherwise

Notes
-----
- The model file mergedCopy.xml must be present in the working directory
- The simulation runs for 12 seconds per trial with 10 learning trials
- First trials may show reduced motion range as ILC learns
- Adjust torque limits and resistance parameters based on your specific hardware

License
-------
This project is for research and educational purposes.

Requirements.txt Content
------------------------
mujoco>=2.3.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pandas>=1.3.0

Recommended Python Version: 3.8+

Optional Dependencies:
   pip install numba
   pip install ipympl
   pip install jupyter
"""
