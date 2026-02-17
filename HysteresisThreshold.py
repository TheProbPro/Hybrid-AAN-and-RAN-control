import time

############################################################################
#                       Hysteresis thresholding class                      #
############################################################################

class Hysteresis():
    """
    Hysteresis thresholding class.
    low_thresh: Lower threshold for switching off
    high_thresh: Upper threshold for switching on
    on_dwell_s: Minimum time to stay on after switching on
    off_dwell_s: Minimum time to stay off after switching off
    state: Initial state (True=on, False=off)
    """
    def __init__(self, low_thresh=5.6, high_thresh=5.7, on_dwell_s=0.0, off_dwell_s=0.0, initial_state=False):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.on_dwell_s = on_dwell_s
        self.off_dwell_s = off_dwell_s
        self.state = initial_state
        self._candidate = None
        self._candidate_start_t = None

    def update(self, value:float, current_time:float | None):
        """
        Update state based on input value and time.
        
        value: Current data sample
        current_time: Timestamp in second, if None uses time.monotonic()
        """
        if current_time is None:
            current_time = time.monotonic()

        # Define instantaneous intent based on hysteresis band
        if not self.state:
            # Currently OFF: only consider turning ON if we cross high
            if value > self.high_thresh:
                intent = "on"
            else:
                intent = None
        else:
            # Currently ON: only consider turning OFF if we go below low
            if value < self.low_thresh:
                intent = "off"
            else:
                intent = None

        # No intent => cancel any pending candidate
        if intent is None:
            self._candidate = None
            self._candidate_start_t = None
            return self.state

        # Start / continue candidate timing
        if intent != self._candidate:
            self._candidate = intent
            self._candidate_start_t = current_time

        elapsed = current_time - self._candidate_start_t
        required = self.on_dwell_s if intent == "on" else self.off_dwell_s

        # Commit if dwell satisfied
        if elapsed >= required:
            self.state = (intent == "on")
            self._candidate = None
            self._candidate_start_t = None
        return self.state