import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 4)
    ax.set_ylim(0,4)
    return ln,

def genX():
    x = np.random.randint(4)
    return x

def genY():
    y = np.random.randint(4)
    return y

def update(frame):
    print(xdata, ydata)
    xdata.clear()
    ydata.clear()
    xdata.append(genX())
    ydata.append(genY())
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, init_func=init, blit=True)
plt.show()