import pickle

import numpy as np
import matplotlib.pyplot as plt

from binocular.reservoir_rfc import ReservoirRandomFeatureConceptor
from binocular import pattern_functions
from binocular import utils

np.random.seed(0)

patterns = []
for p in [53, 54, 10, 36]:
    patterns.append(pattern_functions.patterns[p])

reservoir = ReservoirRandomFeatureConceptor.init_random()
reservoir.fit(patterns)
Y_recalls = reservoir.recall()

allDriverPL, allRecallPL, NRMSE = utils.plot_interpolate_1d(patterns, Y_recalls)

results = [allDriverPL, allRecallPL, reservoir.conceptors, reservoir.history["c"]]

with open('../tests/data/FigureObject.fig_rfc.pickle', "wb") as fp:
    pickle.dump(results, fp)

# plot adaptation of c's
for i in range(len(patterns)):
    # c spectrum
    plt.subplot(len(patterns), 3, (i + 1) * 3 - 2)
    plt.plot(np.flipud(np.sort(reservoir.conceptors[i])), color='black', linewidth='2')
    if (i + 1) * 3 - 2 == 1:
        plt.title('c spectrum')

    plt.subplot(len(patterns), 3, (i + 1) * 3 - 1)
    # driver and recall
    plt.ylim([-1.1, 1.1])
    # text(1, 0, round(NRMSE[i],4), bbox=dict(facecolor='white', alpha=1))
    plt.plot(allDriverPL[i, :], color='gray', linewidth=4.0)
    plt.plot(allRecallPL[i, :], color='black')
    if (i + 1) * 3 - 1 == 2:
        plt.title('driver and recall')

    # c adaptation
    plt.subplot(len(patterns), 3, (i + 1) * 3)
    n_steps_to_plot = 40
    colormap = plt.cm.gray
    plt.gca().set_prop_cycle(plt.cycler('color', [colormap(i) for i in np.linspace(0, 0.9, n_steps_to_plot)]))
    plt.plot(reservoir.history["c"][i, :, 1:n_steps_to_plot])
    plt.ylim([-0.1, 1.1])
    if (i + 1) * 3 == 3:
        plt.title('c adaptation')

plt.show()
