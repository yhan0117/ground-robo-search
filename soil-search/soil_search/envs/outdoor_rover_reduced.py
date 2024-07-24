import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from soil_search.envs.targets_generation import *
# from targets_generation import *
import cv2 
import skimage.exposure
from stable_baselines3.common.type_aliases import GymStepReturn
from typing import Dict, List, Optional, Tuple, Union

class SoilSearchEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None, size: int = 1000, grid_size: int = 10, render_fps=None, **kwargs): 
        # Environment definition
        self.size = size            # The size of the simulation environment 
        self.window_size = 600      # The size of the PyGame window
        self.grid_size = grid_size  # The size of each grid square

        # Fundamental spaces definition
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([-size//grid_size//2, -size//grid_size//2, 10]), high=np.array([size//grid_size//2, size//grid_size//2, 40]), shape=(3,), dtype=int),
                "scan": spaces.Box(0, 255, shape=(5,5), dtype=np.uint8),
            }
        )

        self.action_space = spaces.MultiDiscrete(np.array([21,21,31]))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_fps = int(render_fps) if render_fps is not None else self.metadata["render_fps"]
        self.window = None
        self.clock = None

    '''
    Observation model to compute observation from current state and action  
    '''    
    def _add_noise(self, scan_raw):
        '''
        Add noise to the raw scan to simulate impaired visibility on real systems
        '''
        # Create a noisy image and morph into clusters
        noise = cv2.GaussianBlur(
            self._rng.integers(0, 255, scan_raw.shape, np.uint8, True), 
            (25,25), 
            sigmaX=15, 
            sigmaY=15, 
            borderType = cv2.BORDER_DEFAULT
        )
        stretch = skimage.exposure.rescale_intensity(noise, in_range='image', out_range=(0,255))
        scaled_noise = blending(0.2, stretch).astype(np.uint8)*255 

        # Calculate level of visibility impairment as function of height
        noise_level = blending(-0.35, (self._agent[2]-10)/30)-1
        noise_level = np.clip(np.random.normal(noise_level, 0.08, 1), 0, 1)[0]
        blur_kernel = (int(noise_level*4)*4+1, int(noise_level*4)*4+1)
 
        # Overlay the noisy image onto the raw scan
        scan_blended = cv2.addWeighted(scan_raw, 1-noise_level, scaled_noise, noise_level, 0) 
        scan_blurred = cv2.GaussianBlur(scan_blended, blur_kernel, sigmaX=10, sigmaY=10, borderType = cv2.BORDER_DEFAULT)

        # Convert noisy scan into binary values
        binary = cv2.threshold(scan_blurred, max(45,np.mean(scan_blurred)-10), 255, cv2.THRESH_BINARY)[1]

        # Debug
        if False:
            print(f"Height: {self._agent[2]}")
            print(f"Noise level: {noise_level}")
            cv2.imshow('raw', cv2.resize(scan_raw, (200,200)))
            cv2.imshow('og', cv2.resize(noise, (200,200)))
            cv2.imshow('ya', cv2.resize(stretch.astype(np.uint8), (200,200)))
            cv2.imshow('sig', cv2.resize(scaled_noise, (200,200)))
            cv2.imshow('blend', cv2.resize(scan_blended.astype(np.uint8), (200,200)))
            cv2.imshow('blur', cv2.resize(scan_blurred.astype(np.uint8), (200,200)))
            cv2.imshow('binary', cv2.resize(binary.astype(np.uint8), (200,200)))
            cv2.waitKey(0)

        return binary

    def _get_obs(self):
        scan_size= int(2.5*self._agent[2])
        scan_region = np.array([
            [-scan_size//2, -scan_size//2], 
            [scan_size//2+1, scan_size//2+1] 
        ]) + self._get_pixel()

        scan = self._world_map[scan_region[0,0]:scan_region[1,0], scan_region[0,1]:scan_region[1,1]]
        if scan.size >= (scan_size+1)**2: 
            self._true_scan = cv2.resize(scan, (25,25))
            self._robot_view = cv2.resize(self._add_noise(scan), (25,25))
            
        return {"agent": self._agent, "scan": cv2.resize(self._robot_view, (5,5), interpolation=cv2.INTER_AREA)}
    
    def _get_info(self):
        return {
            "status": self._status,
            "true_scan": self._true_scan,
            "part_rew": self._reward,
            "pixel": self._get_pixel()
        }
    
    def _get_pixel(self):
        return self._agent[-2::-1]*self.grid_size + [self.size//2]*2

    '''
    Method to reset the environment at the start of each episode
    '''    
    def reset(self, seed=None, options={}):

        self._rng = np.random.default_rng(seed=seed)
        super().reset(seed=seed)

        assert np.all([key in ["visibility", "target", []] for key in options.keys()])  
        target =  options["target"] if "target" in options.keys() else "random"
        self._visibility = options["visibility"] if "visibility" in options.keys() else "high"

        # Generate targets
        if target == "random":
            # Reset target locations randomly until sufficient target area is acheived and target does not overlap origin
            while True:
                world, target_area = random_targets(np.zeros((self.size, self.size), dtype=np.uint8), self._rng)
                if 0.05 <= target_area/self.size**2 <= 0.8 and not world[self.size//2, self.size//2]:
                    break

        elif target == "sparse":
            pass

        self._world_map = world
        self._target_area = target_area
        self._odor_field, self._dist_field = create_reward_map(world)

        # Reset agent location & step count
        self._step_cnt = 0
        self._agent = np.array([0,0,self._rng.integers(15,35,1)[0]], dtype=int)
        self._reward = None
        self._status = [False]*3
        self._prev_rew = self._dist_field[self._get_pixel()[0], self._get_pixel()[1]]

        # Return initial observations
        observation = self._get_obs()
        info = self._get_info()
        info["target_area"] = self._target_area
        info["world_map"] = self._world_map
        info["seed"] = seed

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    '''
    Take a step in the simulation given the agent's action.
    Here state transition, observation, information, and reward are computed
    '''
    def _map_action(self, action):
        return action - [10,10,15]
    
    def step(self, action: Union[int, np.ndarray]) -> GymStepReturn:
        # State transition
        assert self.action_space.contains(action), f"Invalid action {action}"
        self._agent += action
        self._agent[2] = np.clip(self._agent[2], 10, 40)
        
        # Observation
        observation = self._get_obs()

        # Termination conditions
        found = np.mean(self._true_scan) > 250 and self._agent[2] < 15 # found target
        fail = np.any([
            self._get_pixel() <= 100, 
            self._get_pixel() >= self.size - 100
        ])  or self._step_cnt >= 300 # out of bounds or exceed step count

        # Reward model
        if found or fail:
            reward = 700*found - 500*fail 
        else:
            dist_penalty = int(np.ceil(5*(blending(3, -np.linalg.norm(action[:-1])+13))))
            step_penalty = 26
            odor_reward = int(np.ceil(self._dist_field[self._get_pixel()[0], self._get_pixel()[1]] - self._prev_rew))
            spot_reward = int(np.mean(self._true_scan)/10)
            reward = spot_reward + odor_reward - dist_penalty - step_penalty
            self._prev_rew = self._dist_field[self._get_pixel()[0], self._get_pixel()[1]]
            self._reward = [dist_penalty, step_penalty, odor_reward, spot_reward]

        # Update robot information
        self._status = [found, fail, self._step_cnt >= 300]
        self._step_cnt += 1

        # Render frame
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, found or fail, self._step_cnt >= 300, self._get_info()


    '''
    Visualize simulation using PyGame
    '''
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (int(self.window_size*1.5), self.window_size)
            )
            self.window.fill((255,255,255))
            pygame.display.set_caption('Soil Search')

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Prepare world map and robot view
        world = cv2.resize(cv2.bitwise_not(self._world_map), (self.window_size, self.window_size))
        world[world > 200] = 200
        world[world < 80] = 40
        world = np.array([world]*3).T
        view = cv2.resize(cv2.bitwise_not(self._robot_view), (int(self.window_size*0.3), int(self.window_size*0.3)))
        view[view > 200] = 200
        view[view < 80] = 40
        view = np.array([view]*3).T

        world = pygame.image.frombuffer(world.tobytes(), world.shape[:2], "RGB")
        view = pygame.image.frombuffer(view.tobytes(), view.shape[:2], "RGB")

        # Draw agent
        scan_size= int(2.5*self._agent[2])
        vertices = (np.array([
                [-scan_size//2, -scan_size//2], 
                [-scan_size//2, scan_size//2], 
                [scan_size//2, scan_size//2], 
                [scan_size//2, -scan_size//2]
            ]) + self._get_pixel()) * (self.window_size/self.size)
        pygame.draw.polygon(
            world,
            pygame.Color(70, 70, 190, a=255),
            vertices.astype(int).tolist() ,     
            0
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(world, world.get_rect())
            self.window.blit(view, (int(self.window_size*1.1), int(self.window_size*0.45)))
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.render_fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(world)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = SoilSearchEnv(render_mode="human")
    import time
    # for i in range(10):
    #     timer = time.time()
    env.reset(seed = 11)
    #     print(time.time() - timer)
    #     time.sleep(2)

    #     timer = time.time()
    done = False
    while not done:

        o, r, done, tc, i = env.step(np.array([0,1,0]))
        # print(time.time() - timer)
        # timer = time.time()
        pos = i["pixel"]
        reward = i["part_rew"]
        frame = env._dist_field.copy()
        frame = (frame/(np.max(frame) - np.min(frame))*255).astype(np.uint8).T
        frame = cv2.circle(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB), [pos[0],pos[1]], 10, (255,0,0),-1)
        cv2.imshow('a',cv2.resize(frame,(500,500)))
        print(reward)
        env.render()
        cv2.waitKey(0)

