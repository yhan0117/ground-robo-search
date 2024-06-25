import numpy as np
import cv2

if __name__ == "__main__":
    # Map of the world
    dim = 1000
    world = np.zeros((dim,dim))
    # initialize target locations 

    # Define robot
    robot_state = np.array([dim,dim/2])
    reachable = np.concatenate((np.repeat([-1,0,1],3), np.tile([1,2,3],3))).reshape(2,9)
    