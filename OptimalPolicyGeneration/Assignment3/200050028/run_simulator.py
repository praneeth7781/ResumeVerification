from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

'''
A global list to contain the centers of the four pits
'''
cen_list = None
false_count = 0


'''
Basically an equality checker for doubles
'''
def are_close(x, y):
    if (abs(x-y) < 1e-6):
        return True
    return False

dont_disturb = 0


'''
Once you feel that you can reach the road through an obstacle 
free path, use this function to decide the next move
'''
def decision(x, y, angle, type):
    action_steer = 1
    action_acc = 4
    
    if (angle >= np.pi/2 and angle <= np.pi):
        action_steer = 0
        action_acc = 2
    elif (angle > np.pi and angle < 3*np.pi/2):
        action_steer = 2
        action_acc = 2
    else:
        if (not(are_close(angle, np.pi/2)) and not(are_close(angle, -np.pi/2))):
            intercept = y + (350 - x)*np.tan(angle)

        '''
        Type 2 indicates car is in the upper half of the screen.
        Type 1 indicates car is in the lower half of the screen.
        '''
        if (type == 2):
            if (60 <= intercept and intercept <= 75):
                action_steer = 1
                action_acc = 4
            elif (intercept > 75):
                action_steer = 0
                action_acc = 1
            elif (intercept < 60):
                action_steer = 2
                action_acc = 1
        elif (type == 1):
            if (-75 <= intercept and intercept <=-60):
                action_steer = 1
                action_acc = 4
            elif (intercept > -60):
                action_steer = 0 
                action_acc = 1
            elif (intercept < -75):
                action_steer = 2
                action_acc = 1
    return (action_steer, action_acc)



class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        # Replace with your implementation to determine actions to be taken
        action_steer = 1
        action_acc = 0

        x = state[0]
        y = state[1]
        angle = np.radians(state[3])

        if (angle >= np.pi/2 and angle <= np.pi):
            action_steer = 0
            action_acc = 2
        elif (angle > np.pi/2 and angle < 3*np.pi/2):
            action_steer = 2
            action_acc = 2
        else:
            if (not(are_close(angle, np.pi/2)) and not(are_close(angle, -np.pi/2))):
                intercept = y + (350 - x)*np.tan(angle)

            if (-75 <= intercept and intercept <= 75):
                action_steer = 1
                action_acc = 4
            elif (intercept > 75):
                action_steer = 0 
                action_acc = 2
            elif (intercept < -75):
                action_steer = 2
                action_acc = 2
       

        action = np.array([action_steer, action_acc])  
        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken

        global dont_disturb

        '''
        Once you find an obstacle free path to the road, move without any diversions 
        '''
        if (dont_disturb):
            action = (1, 4)
            return action

        global cen_list


        action_steer = 1
        action_acc = 4

        '''
        Obtaining coordinates and heading angle from the state
        '''
        x = state[0]
        y = state[1]
        angle = state[3]
        if (angle < 0):
            angle += 360
        angle = np.radians(angle)

        '''
        I have coded the conditions for each quadrant separately
        '''
        if (y >= 100 and cen_list[2][0] + 75 < x and x < cen_list[0][0] - 75):
            if (abs(angle - 3*np.pi/2) < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2
        elif (y <= -100 and cen_list[3][0] + 50 < x and x < cen_list[1][0] - 50):
            if (abs(angle - np.pi/2) < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2

        elif (x <= 0 and y >= 0 and y <= cen_list[2][1] - 50):
            action_steer, action_acc = decision(x, y, angle, 1)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x <= 0 and y >= 0 and x <= cen_list[2][0] - 50):
            if (angle < 3*np.pi/2 - np.pi/18 and 3*np.pi/2 - np.pi/18 - angle < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2
        elif (x <=0  and y >= 0 and x >= cen_list[2][0] + 50):
            if (angle > 3*np.pi/2 and angle - 3*np.pi/2 < np.pi/60):
                action_steer = 1
                action_acc = 4
            else :
                action_steer = 2
                action_acc = 2
        elif (x <= 0 and y >= 0 and y >= cen_list[2][1] + 50):
            if (angle < np.pi/60):
                action_steer = 1
                action_acc = 4
            else:
                action_steer = 0
                action_acc = 2
            
        elif (x >= 0 and y >= 0 and y <= cen_list[0][1] - 50):
            action_steer, action_acc = decision(x, y, angle, 1)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x >= 0 and y >= 0 and x >= cen_list[0][0] + 50):
            action_steer, action_acc = decision(x, y, angle, 1)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x >= 0 and y >= 0 and x <= cen_list[0][0] - 50):
            if (angle < 3*np.pi/2 - np.pi/18 and 3*np.pi/2 - np.pi/18 - angle < np.pi/60):
                action_steer = 1
                action_acc = 4
            else :
                action_steer = 2
                action_acc = 2
        elif (x >= 0 and y >= 0 and y >= cen_list[0][1] + 50):
            if (angle < np.pi/60):
                action_steer = 1
                action_acc = 4
            else:
                action_steer = 0
                action_acc = 2

        elif (x >= 0 and y <= 0 and y >= cen_list[1][1] + 50):
            action_steer, action_acc = decision(x, y, angle, 2)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x >= 0 and y <= 0 and x >= cen_list[1][0] + 50):
            action_steer, action_acc = decision(x, y, angle, 2)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x >= 0 and y <= 0 and x <= cen_list[1][0] - 50):
            if (angle > np.pi/2 and angle - np.pi/2 < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2
        elif (x >= 0 and y <= 0 and y <= cen_list[1][1] - 50):
            if (angle < np.pi/60):
                action_steer = 1
                action_acc = 4
            else:
                action_steer = 0
                action_acc = 2
        
        elif (x <= 0 and y <= 0 and y >= cen_list[3][1] + 50):
            action_steer, action_acc = decision(x, y, angle, 2)
            if (action_acc == 4):
                dont_disturb = 1
        elif (x <= 0 and y <= 0 and x >= cen_list[3][0] + 50):
            if (angle < np.pi/2 and np.pi/2 - angle < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2
        elif (x <= 0 and y <= 0 and x <= cen_list[3][0] - 50):
            if (angle > np.pi/2 and angle - np.pi/2 < np.pi/60):
                action_acc = 4
                action_steer = 1
            else:
                action_acc = 2
                action_steer = 2
        elif (x <= 0 and y <= 0 and y <= cen_list[3][1] - 50):
            if (angle < np.pi/60):
                action_steer = 1
                action_acc = 4
            else:
                action_steer = 0
                action_acc = 2
            
       

        action = np.array([action_steer, action_acc])  
        return action


    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes

        global cen_list

        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]  
            cen_list = ran_cen_list          
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            global dont_disturb
            road_status = False
            dont_disturb = 0

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break
            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=8)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
