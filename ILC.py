import numpy as np
import math
from scipy import interpolate

######################################################################################
#                      Iterative Learning Control (ILC) class                        #
# This class implements a simple ILC algorithm for learning feedforward torque       #
# profiles based on tracking error history. The ILC update is performed at the       #
# end of each trial, and the learned feedforward is used in subsequent trials to     #
# improve tracking performance. The learning rate and other parameters can be tuned. #
######################################################################################

class ILC():
    def __init__(self, max_trials=10, trial_duration=10.0, frequency=166.7, lr=0.1):
        self.max_trials = max_trials
        self.current_trial = 0
        self.learned_feedforward = []
        self.trial_duration = trial_duration
        self.frequency = frequency
        self.error_history = []
        self.lr = lr  # learning rate
        self.ff_iterator = 0

    def update_learning(self, error_array):
        # reset iterator for feedforward retrieval
        self.ff_iterator = 0

        if len(error_array) == 0:
            print("[ILC] Warning: Empty error array, skipping update")
            return np.zeros(int(self.trial_duration * self.frequency))
        
        # Convert error_array to numpy array if it isn't already
        error_array = np.array(error_array)
        
        # Ensure error array is the same length as previous trials
        expected_len = int(self.trial_duration * self.frequency)
        
        # if the length does not match, carry ove the last learned feedforward without update
        if len(error_array) != expected_len:
            len_diff = expected_len - len(error_array)
            if len_diff > 0 and len(self.error_history) > 0:
                error_array = np.concatenate((error_array, self.error_history[-1][-len_diff:]))
            else:
                error_array = np.concatenate((error_array[:expected_len], np.zeros(len_diff)))
        assert len(error_array) == expected_len
        
        # Update learning
        if not self.learned_feedforward:
            ff = np.zeros_like(error_array)
        else:
            ff = self.learned_feedforward[-1] + self.lr * error_array
        
        # save learned feedforward and error history and iterate trial count
        self.learned_feedforward.append(ff)
        self.error_history.append(error_array)
        self.current_trial += 1

        # Performance metrics
        avg_error = np.mean(np.abs(error_array))
        max_error = np.max(np.abs(error_array))
        print(f"[ILC] Trial {self.current_trial} completed:")
        print(f"      Learning rate: {self.lr}")
        print(f"      Avg error: {math.degrees(avg_error)}°")
        print(f"      Max error: {math.degrees(max_error)}°")
        return ff
    
    def get_feedforward(self, trial_idx=-1):
        # retrieve feedforward value at current iterator position
        ff_value = self.learned_feedforward[trial_idx][self.ff_iterator]
        # increment iterator
        self.ff_iterator += 1
        return ff_value