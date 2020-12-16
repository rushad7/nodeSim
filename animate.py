import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animation:
    
    def __init__(self, dims, nodeCoordinates, ballCoordinate):
        self.dims = dims
        self.nodeCoordinates = nodeCoordinates
        self.balllCoordinate = ballCoordinate 
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')

    def setLim(self):
        self.ax.set_xlim(0, self.dims[0])
        self.ax.set_ylim(0, self.dims[1])
        return self.ln,

    def xCoord(self):
        nodeCoordX = self.nodeCoordinates[:, 0]
        ballCoordX = self.balllCoordinate[0]
        xVector = torch.cat((nodeCoordX, ballCoordX.unsqueeze(0)))
        print(xVector)
        return xVector.tolist()

    def yCoord(self):
        nodeCoordY = self.nodeCoordinates[:, 1]
        ballCoordY = self.balllCoordinate[1]
        yVector = torch.cat((nodeCoordY, ballCoordY.unsqueeze(0)))
        print(yVector)
        return yVector.tolist()

    def update(self, frame):
        print(self.xdata, self.ydata)
        self.xdata.clear()
        self.ydata.clear()
        self.xdata.append(self.xCoord())
        self.ydata.append(self.yCoord())
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,

    def start(self):
        _ = FuncAnimation(self.fig, self.update, init_func=self.setLim, blit=True)
        plt.show()

nodecoord = torch.randint(4, (4,2))
ballcoord = 4*torch.rand([2])
animation = Animation(dims=(4,4), nodeCoordinates=nodecoord, ballCoordinate=ballcoord)
animation.start()