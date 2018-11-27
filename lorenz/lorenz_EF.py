"""This script is rather unrelated to the project and can be seen as the hello world of reservoir computing.

It shows how a reservoir network can be used to learned a Lorenz attractor."""
import numpy as np
import matplotlib.pyplot as plt

from lorenz import functions
from lorenz.reservoir_lorenz import Reservoir

patterns = []

tlearn_lorenz = 2000
twash_lorenz = 500

print("Generate Lorenz Sequence... ")
lorenz = functions.Lorenz(0.005, 15, tlearn_lorenz + twash_lorenz, 5000)
lorenz_fun = lambda n: 2 * lorenz[:, n] - 1
patterns.append(lorenz_fun)

print("Initialize Network... ")
reservoir = Reservoir()

print("Run the network with the input sequence... ")
reservoir.run(patterns, t_learn=tlearn_lorenz, t_wash=twash_lorenz)

print("Get the networks response for recall period ...")
reservoir.recall(t_recall=900)

# Plot the driver
plt.subplot(3, 2, 1)
plt.plot(lorenz[0, 0:900], lorenz[1, 0:900])
plt.title('Original driving pattern')

# Plot the response
plt.subplot(3, 2, 2)
# undo rescaling
reservoir.Y_recalls[0] = (reservoir.Y_recalls[0] + 1) / 2
plt.plot(reservoir.Y_recalls[0][:, 0], reservoir.Y_recalls[0][:, 1])
plt.title('Free continuation of learned pattern')

# Plot some network activations
plt.subplot(3, 2, 3)
plt.plot(reservoir.TrainArgs[0:10, 0:100].T)
plt.title('Some network activations')

# Plot some network activations
plt.subplot(3, 2, 4)
plt.plot(reservoir.TrainArgs[110:120, 500:600].T)
plt.title('Some more network activations')

# Plot some output weights
plt.subplot(3, 2, 5)
plt.hist(reservoir.W_out[0, :], 50)
plt.title('Weights of output channel x')

# conceptor singval spectrum
plt.subplot(3, 2, 6)
U, s, V = np.linalg.svd(reservoir.C[0])
plt.plot(s);
plt.title("Singular Value Spectrum of the Conceptor")

# plt.matshow(A.C[0])
# plt.title('Conceptor Matrix')

plt.show()
