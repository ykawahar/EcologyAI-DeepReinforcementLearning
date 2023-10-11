import numpy as np
import random
# import matplotlib.pyplot as plt
import SimManager

ALPHA = 0.8
GAMMA = 0.6

NUM_EPISODES = 1000

#Training Values
MAX_STEPS = 500
EPSILON = 0.1
INIT_EPSILON = 1.0
DECAY = 0.0005

class QLearner:
    
    def __init__(self, sim):
        self.sim = sim
        self.gridSize = sim.gridSize
        numStates = sim.gridSize * sim.gridSize
        self.actionDict = {0:"right", 1: "down", 2:"left", 3:"up", 4:"stay"}
        self.statesDict = {}

        # Fill statesDict
        index = 0
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                self.statesDict[(x, y)] = index
                index += 1




    def bellman(self):
        #Q(s,a) = Q(s,a)+ A * [R(s,a) + G * maxQ(s',a')-Q(s,a)
        # A = rate, G = discount rate, R(s,a) = reward
        pass

    def update(self, epsilon):
        #for iteration in range(numSteps):
            # for agent in self.sim.agentList:
            # choose action
                #action = agent.epsilonGreedy(state, epsilon)
            ## have to convert position to state via state dictionary

            # get newstate, reward, if you reach prey
            #newState, reward, goalAchieved = step(action)

            # compute Qvalue
            ## bellman


            # set new state as state
            # repeat until termination/end of episode
            pass

    def epsilonGreedy(self, state, epsilon):
        if random.random() < epsilon:
            # choose random value
            #chosenAction = self.qTable[state, randomAction]
            pass
        else:
            # choose max Q-value in Q-table that is around you
            chosenAction = np.argmax(self.qTable[state])
            pass
        return chosenAction

    def greedy(self):
        # choose max Q-value in Q-table that is around you
        pass

    def train(self):
        count = 0
        ep_count = []
        epi_value = []
        for episode in range(NUM_EPISODES):
            count += 1
            ep_count.append(count)
            #decay epsilon over time, as agents should be random less and less over time (initially, agents should explore more)
            epsilon = EPSILON + (INIT_EPSILON - EPSILON) * np.exp(-DECAY*episode) # BASE_E + ADD_E * e^episide*decay
            epi_value.append(epsilon)
            print(epsilon)

            #reset environment to initial
            #TODO
            step = 0

            # for each iteration:
            #     update(epsilon)

        #
        # fig, ax = plt.subplots()
        # ax.plot(ep_count, epi_value)
        # ax.set(xlabel="episode", ylabel="epsilon value", title="epsilon decay rate")
        # ax.grid()
        # plt.show()

if __name__ == "__main__":
    pass