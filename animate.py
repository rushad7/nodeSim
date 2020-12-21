import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Animation:

    def __init__(self, coordMap, dims):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, dims[0])
        self.ax.set_ylim(dims[1])
        self.coordMap = coordMap
        self.x = list(coordMap.keys())
        self.y = list(coordMap.values())
        self.line, = self.ax.plot(self.x, self.y, 'ro')

    def animate(self, i):
        self.line.set_ydata(self.coordMap[i])
        return self.line,

    def init(self):
        self.line.set_ydata(np.ma.array(self.x, mask=True))
        return self.line,

    def start(self):
        _ = animation.FuncAnimation(self.fig, self.animate, self.x, init_func=self.init, interval=100, blit=True)
        plt.show()