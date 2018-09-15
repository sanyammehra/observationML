from EyeTribe.peyetribe import EyeTribe
import time
import scipy.io
import numpy as np
import os


WIDTH           = 1280
HEIGHT          = 800
DONE            = False
NUM_READINGS    = 4
count = 0


"""
Reads the X and Y coordinates of the viewer's gaze from the tracker.
Returns co-ordinates after removing noise and smoothing. Makes certain that the
value being returned cooresponds to a FIXATION and not a SACCADE. By editing the
global variable NUMREADINGS, the extent of smoothing can be varied.

Note - Increasing the NUMREADINGS may lead to poorer performance due to higher
latency.

args:
    tracker: the tracker instance of the running eyetribe tracker

returns:
    X, Y: the X and Y coordinates of the viewer's gaze
"""
def getXY(tracker):
    X = 0.0
    Y = 0.0
    readings = 0
    while readings<NUM_READINGS:
        n = tracker.next()
        fix = False
        n_list = str(n).split(';')
        if n_list[3] == 'F':
            fix = True
        if fix:
            y_pos, x_pos = n_list[7:9]
            x_pos, y_pos = int(round(float(x_pos))), int(round(float(y_pos)))
            x_pos = max(0, min(HEIGHT-1, x_pos))
            y_pos = max(0, min(WIDTH-1, y_pos))
            X += x_pos
            Y += y_pos
            readings += 1
    X = int(round(X/NUM_READINGS))
    Y = int(round(Y/NUM_READINGS))
    return X, Y

if __name__ == "__main__":

#    Instantiate tracker
    tracker = EyeTribe()
    tracker.connect()
    n = tracker.next()
    tracker.pushmode()

    while not DONE:
        count += 1
        # Get coordinates
        x_pos, y_pos = getXY(tracker)
        print('{}\tX: {}\tY:{}'.format(count, x_pos, y_pos))
        if count >= 100:
            DONE = True

    tracker.pullmode()
    tracker.close()
