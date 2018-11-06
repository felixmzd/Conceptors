from reservoir_lorenz import *
import pattCat
import time
from matplotlib.pyplot import *
from functions import *



patterns = []

tlearn_lorenz = 2000
twash_lorenz = 500

print("Generate Lorenz Sequence... ")
lorenz = Lorenz(0.005, 15, tlearn_lorenz+twash_lorenz, 5000)
lorenz_fun = lambda n: 2 * lorenz[:,n]-1
patterns.append(lorenz_fun)

print("Initialize Network... ")
A = Reservoir()

t0 = time.clock()

print("Run the network with the input sequence... ")
A.run(patterns, t_learn= tlearn_lorenz, t_wash = twash_lorenz)

print("Get the networks response for recall period ...")
A.recall(t_recall=900)

print("Time needed except initialization and plotting: ")
print(time.clock()-t0)

# Plot the driver
subplot(3, 2, 1)
plot(lorenz[0,0:900],lorenz[1,0:900])
title('Original driving pattern')

# Plot the response
subplot(3, 2, 2)
# undo rescaling
A.Y_recalls[0] = (A.Y_recalls[0] + 1) / 2
plot(A.Y_recalls[0][:,0],A.Y_recalls[0][:,1])
title('Free continuation of learned pattern')

# Plot some network activations
subplot(3, 2, 3)
plot(A.TrainArgs[0:10,0:100].T)
title('Some network activations')

# Plot some network activations
subplot(3, 2, 4)
plot(A.TrainArgs[110:120,500:600].T)
title('Some more network activations')

# Plot some output weights
subplot(3,2,5)
hist(A.W_out[0,:],50)
title('Weights of output channel x')

# conceptor singval spectrum
subplot(3,2,6)
U, s, V = np.linalg.svd(A.C[0])
plot(s);
title("Singular Value Spectrum of the Conceptor")

#matshow(A.C[0])
#title('Conceptor Matrix')

show()



