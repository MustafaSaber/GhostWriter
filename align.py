import numpy as np

def align(frame, depth_frame, depth_scale, threshold = 1000.00):

    # (width, height,_) = frame.shape
    threshold/=depth_scale
    new_depth = depth_frame.copy()
    newframe = frame.copy()
    new_depth[new_depth>threshold]=0
    new_depth[new_depth>0.0]=1

    new_depth = np.dstack((new_depth,new_depth,new_depth))

    newframe = np.multiply(newframe, new_depth.real,dtype="uint8")
    return frame, newframe

