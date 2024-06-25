import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Map of the world
    dim = 1000
    # world = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    world = np.zeros((dim,dim)).astype("uint8")
    world_state = np.full((dim,dim),80).astype("uint8")
    seeds = (np.random.rand(10,2)*800).astype(int)

    # initialize target locations 
    for seed in seeds:
        w, h = (np.random.rand(2)*150 + 50).astype(int)
        world[seed[0]-h//2:seed[0]+h//2, seed[1]-w//2:seed[1]+w//2] = 255


    # Define robot
    robot_state = np.array([dim-1,dim/2]).astype(int)
    reachable = np.transpose(np.concatenate((np.repeat(-np.arange(5)-1,5), np.tile(np.arange(5)-np.floor(5/2),5))).reshape(2,25)).astype(int)
    visible = [[0,5],[-5,5],[-10,5],[-10,10],[-5,10],[0,-5],[-5,-5],[-10,-5]]

    for i in range(2000):
        # robot explore reachable region
        reachable_region = reachable + robot_state
        # if goal is reached
        if np.sum(world[reachable_region[:,0],reachable_region[:,1]]) > 255*25/2:
            break
        
        # mark region as explored
        world_state[reachable_region[:,0],reachable_region[:,1]] = 255
        new_explored = np.where(world[reachable_region[:,0],reachable_region[:,1]] < 255)
        world[reachable_region[new_explored,0],reachable_region[new_explored,1]] = 80

        # start search pattern 
        x = None
        y = None
        for grid in visible:
            search_area = reachable_region + grid

            if np.sum(world[search_area[:,0],search_area[:,1]]) > 255*25/2:
                world[reachable_region[new_explored,0],reachable_region[new_explored,1]] = 150
                y,x = np.divide(grid,5)
                print("saw goal")
                break
        if x == None:
            x = np.floor(np.random.rand()*3)-1
            if x == 0:
                y = -1
            else:
                y = 0
        
        robot_state += [int(y),int(x)]
    cv2.imshow("Exlpore", world_state)
    cv2.imshow("World", world)

    cv2.waitKey(0)

# # generate 2 2d grids for the x & y bounds
# y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

# z = (1 - x / 2. + x ** 15 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# # x and y are bounds, so z should be the value *inside* those bounds.
# # Therefore, remove the last value from the z array.
# z = z[:-1, :-1]
# z_min, z_max = -np.abs(z).max(), np.abs(z).max()

# fig, ax = plt.subplots()

# c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
# ax.set_title('pcolormesh')
# # set the limits of the plot to the limits of the data
# ax.axis([x.min(), x.max(), y.min(), y.max()])
# fig.colorbar(c, ax=ax)

# plt.show()