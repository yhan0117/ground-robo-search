import skimage.exposure
import cv2
import numpy as np
import math
import skfmm

def blending(a, x):
    if a > 0:
        return 2*np.exp(-x/a)/(np.exp(-x/a) + 1)
    else:
        return 2/(np.exp(x/a) + 1)



def random_targets(grid_world, rng, num_target=4):
    h, w = grid_world.shape[:2]
    noise = rng.integers(0, 255, (h,w), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=25, sigmaY=25, borderType = cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,255)).astype(np.uint8)  
    
    cv2.circle(stretch, [w//2,h//2], w//8, 0, -1) # erase trivial targets too close to start
    
    stretch = cv2.GaussianBlur(stretch, (0,0), sigmaX=20, sigmaY=20, borderType = cv2.BORDER_DEFAULT)
    thresh = cv2.threshold(stretch, 140, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)        
    
    # Filter out small target areas
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:4]
    target_area = np.sum([cv2.contourArea(contour) for index, contour in enumerate(contours)])

    grid_world = grid_world.copy()
    cv2.drawContours(grid_world, contours, -1, 255, -1)

    return grid_world, target_area

def create_reward_map(grid_world):
    phi = np.where(grid_world > 0, 0, -1) + 0.5
    dist_field = skfmm.distance(phi, dx = 2)

    # blurred = cv2.GaussianBlur(grid_world, (0,0), sigmaX=100, sigmaY=100, borderType=cv2.BORDER_DEFAULT).astype(np.uint8)
    # stretch = skimage.exposure.rescale_intensity(blurred, in_range='image', out_range=(0,255)).astype(float)
    # dist_field = (blending(-70, stretch)-1)*255
    # odor_field = np.ceil(dist_field*15)

    if False:
        cv2.imshow("original", cv2.resize(grid_world, (500,500)))
        cv2.imshow("Stretch", cv2.resize(dist_field.astype(np.uint8), (500,500)))
        cv2.imshow("odor", cv2.resize(odor_field.astype(np.uint8)*15, (500,500)))
        cv2.waitKey(0)

    return None, dist_field

def circular_targets(grid_world, seed, num_target=4):
    h, w = grid_world.shape[:2]
    rng = np.random.default_rng(seed=seed)
    targets = rng.integers([0, 0, 70], [w, h, 150], (num_target,3), int, True)

    grid_world = grid_world.copy()
    for target in targets:
        cv2.circle(grid_world, target[:2], target[2], 255, -1)

    return grid_world
    
if __name__ == "__main__":
    seed = 5
    world = random_targets(np.zeros((1000, 1000), dtype=np.uint8), seed=seed)

    import torch
    from torch import nn
    import torch.nn.functional as F


    img = world[100:125,100:125]
    input = torch.tensor(np.expand_dims(img, axis=(0,1)), dtype=torch.float32)
    kernel = torch.tensor(np.expand_dims(np.ones((5,5)), axis=(0,1)), dtype=torch.float32)
    output = F.conv2d(input, kernel, stride=5, padding=0)
    output = (output.numpy().squeeze()/25).astype(np.uint8)

    output2 = cv2.resize(img, (5,5),  interpolation = cv2.INTER_AREA)
    # print(out.shape)

    cv2.imshow("og", cv2.resize(img, (200,200),  interpolation = cv2.INTER_AREA))
    cv2.imshow("con", cv2.resize(output, (200,200),  interpolation = cv2.INTER_AREA))
    cv2.imshow("con2", cv2.resize(output2, (200,200),  interpolation = cv2.INTER_AREA))

    cv2.waitKey(0)
    # create_odor_map(world)
