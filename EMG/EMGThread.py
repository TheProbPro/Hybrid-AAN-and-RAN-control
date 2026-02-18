from Filtering import rt_filtering, rt_desired_Angle_lowpass
from EMGToAngle import EMG2Angle
from EMGInterface import DelsysEMG

import queue
import threading
import signal
import math
import time
import numpy as np

# Parameters
EMG_SAMPLE_RATE = 2000  # Hz
SAMPLE_RATE = 166.7  # Hz
ANGLE_MIN = math.radians(0)
ANGLE_MAX = math.radians(140)

USER_NAME = "your_name" # TODO: Change this to your name for calibration data loading

stop_event = threading.Event()

def read_EMG(raw_queue):
    # Initialize filters
    filter_bicep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    filter_tricep = rt_filtering(EMG_SAMPLE_RATE, 450, 20, 2)
    interpreter = EMG2Angle(theta_min=ANGLE_MIN, theta_max=ANGLE_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)
    Bicep_RMS_queue = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    emg = DelsysEMG(channel_range=(0,1))
    emg.start()

    time.sleep(1.0)  # Allow EMG sensor to stabilize

    while not stop_event.is_set():
        reading = emg.read()

        filtered_bicep = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get()
        Bicep_RMS_queue.put(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get()
        Tricep_RMS_queue.put(filtered_tricep)

        Bicep_RMS = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        desired_angle_deg = math.degrees(interpreter.compute_angle(activation[0], activation[1]))

        try:
            raw_queue.put_nowait(desired_angle_deg)
        except queue.Full:
            raw_queue.get_nowait()
            raw_queue.put_nowait(desired_angle_deg)
        
    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()

# Graceful Ctrl-C shutdown
def handle_sigint(sig, frame):
    stop_event.set()
signal.signal(signal.SIGINT, handle_sigint)