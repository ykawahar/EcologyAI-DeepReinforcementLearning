import random
import numpy as np

EPSILON = 0.1  # 10% Epsilon value


class Agent:
    actions = ["right", "down", "left", "up", "stay"]
    types = ["predator", "prey"]

    def __init__(self, qTable,  ruleset=None, type=None, initPos=(-1, -1), initEnergy=None, id=None):
        self.qTable = qTable
        self.row, self.col = initPos
        # , self.heading
        self.energy = initEnergy
        self.id = id
        self.alive = True
        self.type = type
        self.setColor()
        self.ruleset = ruleset

    def setID(self, id):
        """Sets an ID for the agent"""
        self.id = id

    def getID(self):
        """Returns the agent's ID"""
        return self.id

    def setType(self, type):
        """Sets the type of agent (Predator or prey)"""
        self.type = type

    def getType(self):
        """Returns the agent's type"""
        return self.type

    def getTypeInt(self):
        if self.type == "predator":
            return 2
        elif self.type == "prey":
            return 3
        else:
            return 0

    def getPos(self):
        """Returns row, column of agent"""
        return self.row, self.col

    def updatePos(self, row, col):
        """Updates agent position"""
        self.row = row
        self.col = col

    def getEnergy(self):
        """Returns current energy value."""
        return self.energy

    def updateEnergy(self, value):
        """Updates energy value. If energy reaches zero, agent dies."""
        self.energy += value
        if self.energy <= 0:
            print("Died")
            self.alive = False

    def setColor(self, color=None):
        """Sets the color of the agent"""
        if color != None:
            self.color = color
        else:
            myType = self.getType()
            if myType == "predator":
                self.color = "red"
            elif myType == "prey":
                self.color = "blue"
            else:
                self.color = "magenta"

    def getColor(self):
        return self.color

    def chooseAction(self):
        # No ruleset
        return random.choice(self.actions)

    def getMaxQChoice(self, qTable):
        """Get maximum Q-value from q table using agent position,
        return choice of action according to q-value picked"""
        # TODO: Change it so that it uses SimManger method for looping coordinates
        # TODO: May need to change to qTable not reference from self
        # qChoices = {(self.qTable[self.row, self.col]): "stay",
        #             (self.qTable[self.row, (self.col + 1) % 10]): "right",
        #             (self.qTable[self.row, (self.col - 1) % 10]): "left",
        #             (self.qTable[(self.row + 1) % 10, self.col]): "down",
        #             (self.qTable[(self.row - 1) % 10, self.col]): "up"}

        qChoices = {(qTable[self.row, self.col]): "stay",
                    (qTable[self.row, (self.col + 1) % 10]): "right",
                    (qTable[self.row, (self.col - 1) % 10]): "left",
                    (qTable[(self.row + 1) % 10, self.col]): "down",
                    (qTable[(self.row - 1) % 10, self.col]): "up"}

        maxQ = max(qChoices.keys())
        choice = qChoices[maxQ]
        return choice

    def chooseEpsilonGreedyAction(self, qTable, epsilon):
        """ Implements Epsilon greedy algorithm of choosing highest  """
        if self.ruleset == "Random":
            if random.random() < EPSILON:
                # print("Chose Random")
                return random.choice(self.actions)
            else:
                # print("Chose Greedy")
                return self.getMaxQChoice(qTable)
        else:
            pass



    # def createQTable(self):
    #     #qTable = np.zeroes((gridSize, gridSize) self.actions.size))
    #     pass


# if __name__ == '__main__':
    # qTable = np.random.randint(-3, 2, (5, 5))
    # print(qTable)
    # agent = Agent(qTable, None, "predator", (2, 2))
    # print(agent.chooseGreedyAction())
