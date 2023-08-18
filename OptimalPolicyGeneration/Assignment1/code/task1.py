"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need

def KL(x, y):
    ans = 0
    if (abs(x-y) < 1e-6):
        ans = 0
    elif (abs(1-y) < 1e-6):
        ans = np.inf
    elif (abs(x) < 1e-6 and abs(1-y) > 1e-6):
        ans = -math.log(1-y)
    elif (abs(1-x) < 1e-6):
        ans = x*(math.log(x/y))
    else:
        ans = x*(math.log(x/y)) + (1-x)*(math.log((1-x)/(1-y)))
    return ans

def solve(low, high, ua, pa, rhs):
    epsilon = 1e-6
    while (high - low > epsilon):
        mid = (high+low)/2
        expression = KL(pa, mid) - rhs/ua
        if (abs(expression) < 1e-6):
            break
        if (expression > 0):
            high = mid
        else:
            low = mid
    return (low + high)/2

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.t = 0
        self.sampleCount = np.zeros(num_arms)
        self.empiricalReward = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        arm = 0
        if (self.t < self.num_arms):
            arm = self.t
        else :
            numerator = (np.sqrt(2*math.log(self.t)))*np.ones(self.num_arms)
            ucb = (self.empiricalReward/self.sampleCount) + numerator/(np.sqrt(self.sampleCount))
            arm = np.argmax(ucb)
        self.t += 1
        return arm


        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        self.sampleCount[arm_index] += 1
        self.empiricalReward[arm_index] += reward

        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.t = 0
        self.sampleCount = np.zeros(num_arms)
        self.empiricalReward = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        self.c = 3

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        arm = 0
        if (self.t < self.num_arms):
            arm = self.t%(self.num_arms)
        else :
            for i in range(self.num_arms):
                sampleCount = self.sampleCount[i]
                empProb = (self.empiricalReward[i])/(self.sampleCount[i])
                rhs = math.log(self.t) + self.c*math.log(math.log(self.t))
                self.ucb[i] = solve(empProb, 1, sampleCount, empProb, rhs)
        
            arm = np.argmax(self.ucb)

        return arm

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        self.t += 1

        self.sampleCount[arm_index] += 1
        self.empiricalReward[arm_index] += reward

        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.t = 0
        self.successes = np.zeros(self.num_arms)
        self.failures = np.zeros(self.num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE

        beta = np.random.beta(self.successes + 1, self.failures + 1)
        arm = np.argmax(beta)
        self.t += 1
        
        return arm

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if (reward == 0):
            self.failures[arm_index] += 1
        else:
            self.successes[arm_index] += 1
        # END EDITING HERE
