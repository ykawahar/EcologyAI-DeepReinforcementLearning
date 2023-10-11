import time
from tkinter import *
from SimManager import SimManager

SIZE_FACTOR = 20


class GUI:
    """
    Creates a GUI representation of the simulation, everything is assumed to be in a grid
    """

    def __init__(self, sim):
        self.sim = sim
        self.root = Tk()

        self.canvasSize = sim.gridSize * SIZE_FACTOR
        self.root.geometry(f"{self.canvasSize}x{self.canvasSize}")
        self.root.minsize(self.canvasSize * 3, self.canvasSize * 3)  # This makes it so that it is always slightly bigger
        self.canvas = Canvas(self.root, width=self.canvasSize, height=self.canvasSize, bg="white")
        self._initButtons()
        self.canvas.pack()
        self.drawCanvas()
        # self.root.after(1000,self.update)
        self.root.mainloop()

    def _initButtons(self):
        simFrame = Frame(self.root, bd=5, padx=10, pady=10, relief="groove")
        simFrame.grid(row=3, column=1, padx=5, pady=5)
        simFrame.pack()
        self.stepButton = Button(simFrame, text="Step simulation", command=self.stepSimulation)
        self.runButton = Button(simFrame, text="Run simulation", command=self.runSimulation)
        self.resetButton = Button(simFrame, text="Reset simulation", command=self.resetSimulation)
        self.clearButton = Button(simFrame, text="Clear simulation", command=self.clear)
        self.stepCount = Label(simFrame, text=self.sim.stepCount)
        # self.stepButton.grid(row=6, column=1, columnspan=2, pady=5)
        self.stepButton.pack()
        self.runButton.pack()
        self.resetButton.pack()
        self.clearButton.pack()
        self.stepCount.pack()

    def drawWalls(self):
        """
        Draws the walls for the matrix onto the canvas
        """
        row, col = self.sim.getMatrix().shape
        for x in range(row):
            for y in range(col):
                if self.sim.mat[x, y] == 1:
                    self.canvas.create_rectangle(x * SIZE_FACTOR, y * SIZE_FACTOR,
                                                 x * SIZE_FACTOR + SIZE_FACTOR, y * SIZE_FACTOR + SIZE_FACTOR,
                                                 fill="grey")

    def drawCanvas(self):
        row, col = self.sim.getMatrix().shape
        self.canvas.delete("all")
        for x in range(col):
            for y in range(row):
                if self.sim.mat[y, x] == 1:  # This is the correct way to display the row columns
                    self.canvas.create_rectangle(x * SIZE_FACTOR, y * SIZE_FACTOR,
                                                 x * SIZE_FACTOR + SIZE_FACTOR, y * SIZE_FACTOR + SIZE_FACTOR,
                                                 fill="grey")
                elif self.sim.mat[y, x] == 2:
                    self.canvas.create_rectangle(x * SIZE_FACTOR, y * SIZE_FACTOR,
                                                 x * SIZE_FACTOR + SIZE_FACTOR, y * SIZE_FACTOR + SIZE_FACTOR,
                                                 fill="red")
                elif self.sim.mat[y, x] == 3:
                    self.canvas.create_rectangle(x * SIZE_FACTOR, y * SIZE_FACTOR,
                                                 x * SIZE_FACTOR + SIZE_FACTOR, y * SIZE_FACTOR + SIZE_FACTOR,
                                                 fill="blue")
        self.stepCount["text"] = self.sim.stepCount

    def stepSimulation(self):
        # self.sim.step()
        self.sim.testAgentStep()
        self.drawCanvas()
        return True

    def runSimulation(self):
        # while not self.sim.terminal():
        #     self.sim.step()
        #     self.drawCanvas()
        #self.sim.trainStep()
        self.sim.train()
        self.drawCanvas()
        return True

    def resetSimulation(self):
        self.canvas.delete("all")
        self.sim.resetEnvironment()
        self.drawCanvas()

    def clear(self):
        self.canvas.delete("all")
        self.drawCanvas()

    def update(self):
        self.sim.step()
        self.drawCanvas()


if __name__ == "__main__":
    s = SimManager(10, 1, 1)
    g = GUI(s)
