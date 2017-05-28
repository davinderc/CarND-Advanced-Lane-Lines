import numpy as np

class Line():
    def __init__(self):
        # Was the line detected in the last iteration?
        self.detected = False
        # x values of last n fits
        self.recent_xfitted = []
        # Average x values of the fitted line over the last n iterations
        self.best_x = None
        # Average polynomial coefficients over last n iterations
        self.best_fit = None
        # Polynomial coefficients for last fit
        self.current_fit = [np.array([0,0,0])]
        # Radius of curvature in meters
        self.rad_curvature = None
        # Offset from center of lane
        self.line_base_pos = None
        # Difference in fit coefficients from last fit to current fit
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

l_line = Line()
r_line = Line()
polygon = np.zeros((720,1280,1),np.uint8)
old_pts = np.zeros((1,1440,2))
first_poly = 1
