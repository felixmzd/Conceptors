# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 17:08:43 2015

@author: User
"""

from reservoir_hrfc import *
import pattCat
import time
from matplotlib.pyplot import *
#from matplotlib2tikz import save as tikz_save

t0 = time.clock()

patterns = []

#for p in [53, 54, 10, 36]:
for p in [54,  36]:
    patterns.append(pattCat.patts[p])

A = ReservoirHRFC()
A.run(patterns)
A.recall()
A.denoise()



print(time.clock()-t0)

### PLOTTING ###

###
allDriverPL, allRecallPL, NRMSE = functions.plot_interpolate_1d(patterns, A.Y_recalls)
for i in range(len(patterns)):
    subplot(len(patterns), 1, (i+1))
    # driver and recall
    ylim([-1.1,1.1])
    text(0.4, -0.9, round(NRMSE[i],4), bbox=dict(facecolor='white', alpha=1))
    plot(allDriverPL[i,:], color='gray', linewidth=4.0)
    plot(allRecallPL[i,:], color='black')
    if((i+1)*3 -1 == 2):
        title('driver and recall')
print(np.shape(A.Z))

###
figure()
subplot(4,1,1)
ylim([-0.1,1.1])
plot(A.all['hypo3'].T)

subplot(4,1,2)
ylim([-0.1,1.1])
plot(A.all['hypo2'].T)

subplot(4,1,3)
ylim([-0.1,1.1])
plot(A.all['hypo1'].T)

subplot(4,1,4)
ylim([-0.1,1.1])
plot(A.all['trusts12'].T, 'b')
plot(A.all['trusts23'].T, 'g')

###
figure()
for i in range(len(patterns)):
    subplot(len(patterns),1,(i+1))
    l_idx = 4000*(i+1)-40
    r_idx = 4000*(i+1)
    # original pattern
    plot(A.all['driver'][:,l_idx:r_idx].T, color='gray', linewidth=4.0)
    # recall
    plot(A.all['y3'][:,l_idx:r_idx].T, color='black')
    # pattern plus noise
    plot(A.all['driver'][:,l_idx:r_idx].T + A.all['noise'][:,l_idx:r_idx].T, 'r')



show()
