import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Animation:
    
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')

    def setLim(self):
        self.ax.set_xlim(0, 4)
        self.ax.set_ylim(0,4)
        return self.ln,

    def genX(self):
        x = np.random.randint(2)
        return [x, 2*x]

    def genY(self):
        y = np.random.randint(2)
        return [y, 2*y]

    def update(self, frame):
        print(self.xdata, self.ydata)
        self.xdata.clear()
        self.ydata.clear()
        self.xdata.append(self.genX())
        self.ydata.append(self.genY())
        self.ln.set_data(self.xdata, self.ydata)
        return self.ln,

    def start(self):
        _ = FuncAnimation(self.fig, self.update, init_func=self.setLim, blit=True)
        plt.show()

animation = Animation()
animation.start()