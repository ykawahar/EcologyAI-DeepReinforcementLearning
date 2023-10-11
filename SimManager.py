import numpy as np
import sys
import random
import copy
import time
import matplotlib.pyplot as plt
import pandas as pd

from Agent import Agent

np.set_printoptions(threshold=sys.maxsize)
predatorScopeOrder = {0: (0, 0),
                      1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0),
                      5: (-1, 1), 6: (1, 1), 7: (1, -1), 8: (-1, -1),
                      9: (0, 2), 10: (2, 0), 11: (0, -2), 12: (-2, 0),
                      13: (-1, 2), 14: (1, 2), 15: (2, 1), 16: (2, -1),
                      17: (1, -2), 18: (-1, -2), 19: (-2, -1), 20: (-2, 1),
                      21: (-2, 2), 22: (2, 2), 23: (2, -2), 24: (-2, -2)}

ALPHA = 0.8
GAMMA = 0.6

NUM_EPISODES = 10000

# Training Values
MAX_STEPS = 100
EPSILON = 0.1
INIT_EPSILON = 1.0
DECAY = 0.0005


class SimManager:
    """
    Manages the backend of the simulation, represents the screen as a matrix or grid.
    """

    def __init__(self, gridSize, initPredatorCount=0, initPreyCount=0, ruleset=None):
        self.historicalCountAgents = None
        self.predatorCount = None
        self.stepCount = 0
        self.preyCount = None
        self.qTable = None
        self.zeroMat = None
        self.agentMap = None
        self.mat = None
        self.gridSize = gridSize
        self.initPredatorCount = initPredatorCount
        self.initPreyCount = initPreyCount
        if ruleset is None:
            self.ruleset = "QLearning"
            # self.ruleset = "Random"
        else:
            self.ruleset = ruleset

        # self.qLearner = QLearner(self, gridSize)

        if self.ruleset == "QLearning":
            self.initQ()
        if self.ruleset == "Random":
            self.setupGrid()

            # ------------QLearner-----------------

            self.agentList = []
            self.predatorList = []
            self.preyList = []
            self.deadAgents = []

            # Initialize array
            self.historicalCountAgents = 0
            self.mat = np.zeros((self.gridSize, self.gridSize))  # Fill array (generate initial)
            self.agentMap = {}
            self.generateWalls()  # Make walls
            self.zeroMat = self.mat.copy()
            self.qTable = self.createQTable()

            self.stepCount = 0
            self.predatorCount = self.initPredatorCount
            self.preyCount = self.initPreyCount

            self.initializeAgents()  # spawn initial agents

            for row in range(self.gridSize):
                for col in range(self.gridSize):
                    self.agentMap[(row, col)] = []

            print("QTABLE", self.qTable)
            print(f"Game Matrix \n{self.mat}")

    def generateWalls(self, a=None, b=None):
        """
        Generates random walls in the matrix, denoted as 1
        """
        if a is None and b is None:
            for i in range(int((
                    self.gridSize ** 2 / 8))):  # The denominator indicates how much of the screen the walls take up
                # chance = random.random()
                x, y = self.getRandCoordinate()
                self.mat[x, y] = 1  # walls are denoted as 1 in the matrix
                # if chance > 0.5:
                #     self.generateWalls(x + 1, y)
        # else:
        #     self.mat[a, b] = 1
        #     self.generateWalls(a + 1, b)

        # print(self.mat)

    def getMatrix(self):
        return self.mat

    def getGridSize(self):
        """Returns size of grid"""
        return self.gridSize

    def getRuleset(self):
        """Returns the ruleset/environment being used"""
        return self.ruleset

    def getStepCount(self):
        """Returns the number of steps iterated so far"""
        return self.stepCount

    def getHistoricalAgentCount(self):
        """Returns the number of agents to ever exist in the sim"""
        # Used to assign ID# to new agents
        return self.historicalCountAgents

    def getRandCoordinate(self):
        """Generates and returns a random x,y coordinate on the grid"""
        return random.randrange(self.gridSize), random.randrange(self.gridSize)

    def initializeAgents(self):
        """Initializes prey and predators"""
        # Predators
        count = 0
        while count < self.initPredatorCount:
            x, y = self.getRandCoordinate()
            if self.mat[x, y] != 1:
                self.historicalCountAgents += 1
                newPredator = Agent(self.qTable, self.getRuleset(), "predator", (x, y), 10,
                                    self.historicalCountAgents)
                self.agentList.append(newPredator)
                self.predatorList.append(newPredator)
                self.agentMap[(x, y)].append(newPredator)
                self.mat[x, y] = 2
                count += 1
        # Prey
        count = 0
        while count < self.initPreyCount:
            x, y = self.getRandCoordinate()
            if self.mat[x, y] != 1:
                newPrey = Agent(self.qTable, self.getRuleset(), "prey", (x, y), 99, self.historicalCountAgents)
                self.agentList.append(newPrey)
                self.preyList.append(newPrey)
                self.agentMap[(x, y)].append(newPrey)
                self.mat[x, y] = 3
                count += 1

    def step(self):
        """Runs a step of the simulation"""
        # print(f"Updating")
        self.updateAgents()
        self.updateAgentQLearning()

        # update agents
        # go through agentlist
        # ask agent to do something

        self.updateMapContents()
        self.stepCount += 1
        # update sim

        # NOT YET IMPLEMENTED
        # update dead agents
        # add to dead agent list
        # remove from agent list
        print("QTABLE\n", self.qTable)
        print("Game Matrix\n", self.mat)

    def destinationPos(self, row, col, heading):
        """Returns the coordinates of the target cell"""
        if heading == "right":
            newCol = (col + 1) % self.gridSize
            return row, newCol
        elif heading == "down":
            newRow = (row + 1) % self.gridSize
            return newRow, col
        elif heading == "left":
            newCol = (col - 1) % self.gridSize
            return row, newCol
        elif heading == "up":
            newRow = (row - 1) % self.gridSize
            return newRow, col
        else:
            return row, col

    def correctPos(self, row, col):
        """Loops around if out of bounds"""
        newRow, newCol = row, col
        if row >= self.getGridSize():
            newRow = row % self.gridSize
        if col >= self.getGridSize():
            newCol = col % self.gridSize
        return newRow, newCol

    def scopePrey(self, agent):
        preyInRange = []  # queue of prey, in priority order (same distance is arbitrary hard order)
        visibleScope = copy.deepcopy(predatorScopeOrder)  # local copy of dictionary to mutate
        blockedCells = []  # cells that are blocked from view/walls
        currPos = agent.getPos()

        for key in visibleScope.keys():
            offset = visibleScope.get(key)  # key.get()
            viewPos = self.correctPos(currPos[0] + offset[0], currPos[1] + offset[1])
            # print(f"Grid size = {self.gridSize} \n viewPos = {viewPos} \n currPos = {currPos}")
            cellState = self.mat[viewPos]
            if cellState == 1:
                blockedCells.append(key)
                obstructedCells = self.checkObstructedCells(key, blockedCells)
                # if obstructedCells: -------How does python check lists empty? If it iterates,
                # then it is faster to just do so, but otherwise this should make runtime faster
                # for cell in obstructedCells: # You can't be changing the dictionary while you are iterating it
                # visibleScope.pop(cell)  # we don't need to add these cells to the blockedCells list,
                # as they are in outermost region; They do not block additional cells
                # visibleScope.pop(key) <-- this shouldn't be necessary,
                # so let's leave it out to not mutate the key we are on (although I think this is safe on python?)
            elif cellState == 3:
                preyInRange.append(self.agentMap.get(viewPos))
            else:
                pass

    def checkObstructedCells(self, currentCellNumber, blockedCells):
        obstructedCells = []
        if (currentCellNumber == 1):
            obstructedCells.append(9)
        elif (currentCellNumber == 2):
            obstructedCells.append(10)
            if (1 in blockedCells):
                obstructedCells.append(6)
        elif (currentCellNumber == 3):
            obstructedCells.append(11)
            if (2 in blockedCells):
                obstructedCells.append(7)
        elif (currentCellNumber == 4):
            obstructedCells.append(12)
            if (1 in blockedCells):
                obstructedCells.append(5)
            if (3 in blockedCells):
                obstructedCells.append(8)
        elif (currentCellNumber == 5):
            obstructedCells.append(21)
            if (1 in blockedCells):
                obstructedCells.append(13)
            if (4 in blockedCells):
                obstructedCells.append(20)
        elif (currentCellNumber == 6):
            obstructedCells.append(22)
            if (1 in blockedCells):
                obstructedCells.append(14)
            if (2 in blockedCells):
                obstructedCells.append(15)
        elif (currentCellNumber == 7):
            obstructedCells.append(23)
            if (2 in blockedCells):
                obstructedCells.append(16)
            if (3 in blockedCells):
                obstructedCells.append(17)
        elif (currentCellNumber == 8):
            obstructedCells.append(24)
            if (3 in blockedCells):
                obstructedCells.append(18)
            if (4 in blockedCells):
                obstructedCells.append(19)
        else:
            pass
        return obstructedCells

    def updateAgents(self):
        for agent in self.preyList:  # can you change this back to the way you originally had it?
            currPos = agent.getPos()
            # print("curent", currPos)

            # self.agentList.index(agent)

            while True:
                response = agent.chooseAction()
                intendedPos = self.destinationPos(currPos[0], currPos[1], response)
                intendedTarget = self.mat[intendedPos[0], intendedPos[1]]
                if intendedTarget != 1:
                    nextPos = self.destinationPos(currPos[0], currPos[1], response)
                    # print(response)
                    break
            agent.updatePos(nextPos[0], nextPos[1])

            # print("next", nextPos)

        # check if predator finds prey
        # for i, agent in enumerate(self.predatorList):
        #     self.scopePrey(agent)

        # To Do/Consider: Successful hunts take priority based on seniority; could we make it by proximity?

    def updateAgentQLearning(self):  # greedy move
        for agent in self.predatorList:  # can you change this back to the way you originally had it?
            currPos = agent.getPos()

            response = agent.chooseEpsilonGreedyAction()  # chooses a move based on qvalues
            intendedPos = self.destinationPos(currPos[0], currPos[1], response)
            intendedTarget = self.mat[intendedPos[0], intendedPos[1]]

            if intendedTarget == 1:
                self.updateQtable(intendedPos[0], intendedPos[1], -300)  ##reward for hitting wall is -300
                nextPos = self.destinationPos(currPos[0], currPos[1], "stay")
                agent.updatePos(nextPos[0], nextPos[1])
            else:
                nextPos = self.destinationPos(currPos[0], currPos[1], response)
                agent.updatePos(nextPos[0], nextPos[1])
            newPos = agent.getPos()
            # TODO: Currently checks for prey everytime, even when moving to empty space
            caughtPrey = False
            for prey in self.preyList:
                if prey.getPos() == agent.getPos():
                    self.updateQtable(newPos[0], newPos[1], 100)  ##reward for catching prey is 100
                    self.preyList.remove(prey)
                    self.agentList.remove(prey)
                    caughtPrey = True
                    break
            if not caughtPrey:
                self.updateQtable(newPos[0], newPos[1], -1)  ##reward for catching prey is 100

    def printMap(self):
        print(self.mat)

    def updateMapContents(self):
        self.mat = self.zeroMat.copy()
        for agent in self.agentList:
            pos = agent.getPos()
            self.mat[pos] = agent.getTypeInt()

    def createQTable(self):
        return np.zeros((self.gridSize ** 2, len(self.actionDict)))

    def updateQtable(self, row, col, valueChange):
        self.qTable[row, col] += valueChange

    def terminal(self):
        return len(self.preyList) <= 0

    # --------------------------- From this point on, methods used attributed to QLearner ------------------------------
    def initQ(self):
        self.actionDict = {0: "right", 1: "down", 2: "left", 3: "up", 4: "stay"}
        self.statesDict = {}

        # Fill statesDict
        index = 0
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                self.statesDict[(x, y)] = index
                index += 1

        self.setupGridQ()
        self.episodeCount = 0

    def setupGridQ(self):

        self.agentList = []
        self.predatorList = []
        self.preyList = []
        self.initialPreyPos = []
        self.initialPredPos = []

        # Initialize array
        self.historicalCountAgents = 0
        self.mat = np.zeros((self.gridSize, self.gridSize))  # Fill array (generate initial)
        self.generateWalls()  # Make walls
        self.setGoal()
        self.zeroMat = self.mat.copy()  # initial map with walls and goal
        self.qTable = self.createQTable()

        self.initializeAgentsQ()

        # Initialize reward table
        self.rewardTable = np.zeros((self.gridSize, self.gridSize))
        # for x in range(self.gridSize):
        #     for y in range(self.gridSize):
        #         if self.mat[x,y] == 1:
        #             self.rewardTable[x,y] = -100
        #         if self.mat[x,y] ==3:
        #             self.rewardTable[x,y] = 999
        print("Matrix\n", self.mat)

    def setGoal(self):
        count = 0
        while count < self.initPreyCount:
            x, y = self.getRandCoordinate()
            if self.mat[x, y] != 1:
                self.mat[x, y] = 3
                count +=1

    def resetEnvironment(self):
        #print("resetting")
        self.stepCount = 0
        self.agentList = []
        self.predatorList = []

        self.historicalCountAgents = 0
        self.mat = self.zeroMat.copy()
        self.reinitializeAgents()

    def initializeAgentsQ(self):
        """Initializes prey and predators"""
        # prey
        count = 0
        # while count < self.initPreyCount:
        #     x, y = self.getRandCoordinate()
        #     if self.mat[x, y] != 1:
        #         self.initialPreyPos.append((x, y))
        #         self.historicalCountAgents += 1
        #         newPrey = Agent(self.qTable, self.getRuleset(), "prey", (x, y), 99, self.historicalCountAgents)
        #         self.agentList.append(newPrey)
        #         self.preyList.append(newPrey)
        #         # self.agentMap[(x, y)].append(newPrey)
        #         self.mat[x, y] = 3
        #         count += 1

        # Predators
        count = 0
        while count < self.initPredatorCount:
            x, y = self.getRandCoordinate()
            if self.mat[x, y] != 1 or self.mat[x, y] != 3:
                self.initialPredPos.append((x, y))
                self.historicalCountAgents += 1
                newPredator = Agent(self.qTable, self.getRuleset(), "predator", (x, y), 10,
                                    self.historicalCountAgents)
                self.agentList.append(newPredator)
                self.predatorList.append(newPredator)
                self.mat[x, y] = 2
                count += 1

    def reinitializeAgents(self):
        # for agentCoord in self.initialPreyPos:
        #     self.historicalCountAgents += 1
        #     newPrey = Agent(self.qTable, self.getRuleset(), "prey", agentCoord, 99,
        #                     self.historicalCountAgents)
        #     self.agentList.append(newPrey)
        #     self.preyList.append(newPrey)
        #     # self.agentMap[agentCoord].append(newPredator)
        #     self.mat[agentCoord[0], agentCoord[1]] = 3

        for agentCoord in self.initialPredPos:
            self.historicalCountAgents += 1
            newPredator = Agent(self.qTable, self.getRuleset(), "predator", agentCoord, 10,
                                self.historicalCountAgents)
            self.agentList.append(newPredator)
            self.predatorList.append(newPredator)
            # self.agentMap[agentCoord].append(newPredator)
            self.mat[agentCoord[0], agentCoord[1]] = 2

    def stepQ(self, currPos, action):
        goalAchieved = False
        intendedPos = self.destinationPos(currPos[0], currPos[1], self.actionDict.get(action))
        intendedTarget = self.mat[intendedPos[0], intendedPos[1]]

        newState = self.statesDict.get(intendedPos)  # convert new position to state int (qtable row)
        reward = self.rewardTable[intendedPos]

        if intendedTarget == 1:
            nextPos = currPos
            self.rewardTable[intendedPos] -= 1
        else:
            nextPos = intendedPos
        if intendedTarget == 3:
            self.rewardTable[intendedPos] += 5
            goalAchieved = True

        return newState, reward, goalAchieved, nextPos

    def updateQ(self, epsilon):  # one step of q learning

        #self.updateAgents()

        for agent in self.predatorList:
            currPos = agent.getPos()
            # choose action
            state = self.statesDict.get(currPos)
            action = self.epsilonGreedy(state, epsilon)

            # get newstate, reward, if you reach prey
            newState, reward, goalAchieved, nextPos = self.stepQ(currPos, action)

            # compute Qvalue
            ## bellman
            qVal = self.qTable[state, action] + ALPHA * (
                        reward + GAMMA * np.max(self.qTable[newState] - self.qTable[state, action]))
            #print(f"Qval", qVal)
            self.qTable[state, action] = qVal

            agent.updatePos(nextPos[0], nextPos[1])  # state = newState
            self.updateMapContents()
            #print(f"Map\n", self.mat)
            self.stepCount += 1
            if goalAchieved:
                return True

        return False

    def epsilonGreedy(self, state, epsilon):
        if random.random() < epsilon:
            # choose random value
            chosenAction = random.randrange(0, len(self.actionDict))
        else:
            # choose max Q-value in Q-table that is around you
            chosenAction = np.argmax(self.qTable[state])
        return chosenAction  # return int of action

    def train(self):
        count = 0
        ep_count = []
        epi_value = []
        plt_stepCount = []
        start_time = time.time()
        for episode in range(NUM_EPISODES):
            print(f"Episode {episode}/{NUM_EPISODES}")
            count += 1
            ep_count.append(count)
            # decay epsilon over time, as agents should be random less and less over time (initially, agents should explore more)
            epsilon = EPSILON + (INIT_EPSILON - EPSILON) * np.exp(-DECAY * episode)  # BASE_E + ADD_E * e^episide*decay
            epi_value.append(epsilon)
            plt_stepCount.append(self.stepCount)
            #print(f"epsilon", epsilon)

            # reset environment to initial
            self.resetEnvironment()
            goalAchieved = False

            # for each iteration:
            for i in range(MAX_STEPS):
                goalAchieved = self.updateQ(epsilon)
                #print(self.qTable)
                if goalAchieved:
                    #print("Reached prey")
                    break

        elapsed_time = time.time() - start_time
        print(f"Runtime = {elapsed_time}")
        plt.ion()
        plt_stepCount = [i/len(self.predatorList) for i in plt_stepCount]
        fig, ax = plt.subplots()
        ax.plot(ep_count, plt_stepCount)
        ax.set(xlabel="Episode", ylabel="Average Step count", title="Predator Step Count Over Episode")
        plt.show()
        plt.savefig("graph_result.png")
        self.saveResults(ep_count, plt_stepCount)

    def trainStep(self):
        epsilon = EPSILON + (INIT_EPSILON - EPSILON) * np.exp(
            -DECAY * self.episodeCount)  # BASE_E + ADD_E * e^episide*decay
        print(f"epsilon", epsilon)

        # reset environment to initial
        self.resetEnvironment()
        step = 0
        goalAchieved = False

        # for each iteration:
        for i in range(100):
            goalAchieved = self.updateQ(epsilon)
            print(self.qTable)
            if goalAchieved:
                print("Reached prey")
                break
        self.episodeCount += 1

    def testAgentStep(self):

        print("QTABLE\n", self.qTable)
        print("Reward\n", self.rewardTable)
        #self.updateAgents()

        for agent in self.predatorList:
            currPos = agent.getPos()
            # choose action
            state = self.statesDict.get(currPos)
            action = np.argmax(self.qTable[state])
            newState, reward, goalAchieved, nextPos = self.stepQ(currPos, action)

            agent.updatePos(nextPos[0], nextPos[1])  # state = newState
            self.updateMapContents()
            self.stepCount += 1

        #     if goalAchieved:
        #         return True
        #
        # return False

    def saveResults(self, episodes, steps):
        m_df = pd.DataFrame(self.mat)
        q_df = pd.DataFrame(self.qTable)
        r_df = pd.DataFrame(self.rewardTable)
        m_df.to_csv("result_matrix.csv")
        q_df.to_csv("result_qtable.csv")
        r_df.to_csv("result_reward.csv")

        plt_df = pd.DataFrame.from_dict({"Episodes": episodes, "Steps": steps})
        plt_df.to_csv("stepsOverEpisodes.csv")




if __name__ == "__main__":
    sm = SimManager(10, 5, 1)
    # for i in range(5):
    #     sm.step()
    #     sm.printMap()
