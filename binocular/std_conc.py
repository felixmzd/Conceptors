from reservoir_rfc import *
import pattCat
import time
from matplotlib.pyplot import *
from functions import *


t0 = time.clock()
patterns = []
for p in [53, 54, 10, 36]:
    patterns.append(pattCat.patts[p])

A = ReservoirRFC()
A.run(patterns)
A.recall()


allDriverPL, allRecallPL, NRMSE = functions.plot_interpolate_1d(patterns, A.Y_recalls)


# plot adaptation of c's
for i in range(len(patterns)):
    # c sprectrum
    # subplot(len(patterns), 3, (i+1)*3 -2)
    # plot(np.flipud(np.sort(A.C[i])), color='black', linewidth='2')
    if((i+1)*3 -2 == 1):
        title('c spectrum')

    subplot(len(patterns), 1, (i+1) )
    # driver and recall
    ylim([-1.1,1.1])
    text(1, 0, round(NRMSE[i],4), bbox=dict(facecolor='white', alpha=1))
    plot(allDriverPL[i,:], color='gray', linewidth=4.0)
    plot(allRecallPL[i,:], color='black')
    if((i+1)*3 -1 == 2):
        title('driver and recall')

    # # c adaptation
    # subplot(len(patterns), 3, (i+1)*3)
    # num_plots = 40;
    # colormap = cm.gray
    # gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
    # plot(A.cColls[i,1:num_plots,:].T)
    # ylim([-0.1,1.1])
    # if((i+1)*3 == 3):
    #     title('c adaptation')



show()