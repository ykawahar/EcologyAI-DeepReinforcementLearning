import numpy as np
import random

class SMEnvi2:
    def __init__(self, gridSize, initPredatorCount=0, initPreyCount=0):
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

        # Different from here
        self.initQ()

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

    def resetEnvironment(self):
        #print("resetting")
        self.stepCount = 0
        self.agentList = []
        self.predatorList = []

        self.historicalCountAgents = 0
        self.mat = self.zeroMat.copy()
        self.reinitializeAgents()

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

    def getRandCoordinate(self):
        """Generates and returns a random x,y coordinate on the grid"""
        return random.randrange(self.gridSize), random.randrange(self.gridSize)

    def setGoal(self):
        count = 0
        while count < self.initPreyCount:
            x, y = self.getRandCoordinate()
            if self.mat[x, y] != 1:
                self.mat[x, y] = 3
                count +=1

    def createQTable(self):
        return np.zeros((self.gridSize ** 2, len(self.actionDict)))

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