from functions import *
import numpy as np
from matplotlib.pyplot import *

a = np.array([1,2,3])


series = Lorenz(0.005, 10, 2000,1000)

print(series)

plot(series[0],series[1])
show()