from binocular.reservoir_binocular import *
from binocular import pattern_functions
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from matplotlib.pyplot import *

patterns = []

# for p in [53, 54, 10, 36]:
# quite good :
# for p in [50,  20]:
# for p in [50,  27]:
for p in [50, 53]:
    patterns.append(pattern_functions.patterns[p])

np.random.seed(3)
reservoir = ReservoirBinocular.init_random()
reservoir.fit(patterns)
reservoir.recall()
reservoir.binocular(t_run=5000)

### PLOTTING ###
allDriverPL, allRecallPL, NRMSE = utils.plot_interpolate_1d(patterns, reservoir.Y_recalls, plotrange=100)
loading = [allDriverPL, allRecallPL, NRMSE]

for i in range(len(patterns)):
    plt.subplot(len(patterns), 1, (i + 1))
    # driver and recall
    plt.ylim([-1.1, 1.1])
    plt.text(2, -1, 'NRMSE: {0}'.format(round(NRMSE[i], 4)), bbox=dict(facecolor='white', alpha=1))
    plt.plot(allDriverPL[i, :], color='gray', linewidth=4.0, label='Driver')
    plt.plot(allRecallPL[i, :], color='black', label='Recall')
    plt.legend()
    plt.title('Original driver and recalled signal for Sine {0}'.format(i + 1))

with open('FigureObject.fig_loading.pickle', "wb") as fp:
    pickle.dump(loading, fp)


# save driver input
savedriver = []
savedriver.append(reservoir.all['driver_sine1'][:, 0:100].T)
savedriver.append(reservoir.all['driver_sine2'][:, 0:100].T)
savedriver.append(reservoir.all['driver_noise'][:, 0:100].T)
savedriver.append(reservoir.all['driver'][:, 0:100].T)
with open('FigureObject.fig_rawinput.pickle', "wb") as fp:
    pickle.dump(savedriver, fp, protocol=2)

save_real_input = []
save_real_input.append(reservoir.all['real_input_w-o_noise'][:, 0:100].T)
save_real_input.append(reservoir.all['real_input'][:, 0:100].T)
with open('FigureObject.fig_realinput.pickle', "wb") as fp:
    pickle.dump(save_real_input, fp, protocol=2)

res = []

res.append(reservoir.all['hypo3'])
res.append(reservoir.all['hypo2'])
res.append(reservoir.all['hypo1'])
res.append(reservoir.all['trusts12'])
res.append(reservoir.all['trusts23'])
with open('FigureObject.fig_result.pickle', "wb") as fp:
    pickle.dump(res, fp, protocol=2)

##

with open('FigureObject.fig_predictions.pickle', "wb") as fp:
    pickle.dump(reservoir.all['y3'], fp, protocol=2)

# recall
figure()
plot(reservoir.all['y3'][:, 3960:4000].T, color='black')
plot(reservoir.all['y3'][:, 3960:4000].T - reservoir.all['driver'][:, 3960:4000].T, color='gray', linewidth=4.0)
plt.title('output and difference ')

# discrepancys on every level
figure()
plot(reservoir.all['trusts1'].T, 'b')
plot(reservoir.all['trusts2'].T, 'g')
plot(reservoir.all['trusts3'].T, 'y')
plt.title('trusts')

figure()
plot(reservoir.all['unexplained1'][:, 3960:4000].T, 'b')
plot(reservoir.all['unexplained2'][:, 3960:4000].T, 'g')
plot(reservoir.all['unexplained3'][:, 3960:4000].T, 'y')

# plot(reservoir.all['driver'][:, 0:500].T, 'g')
plt.title('unexplained')

# compute dominance times per signal / eye
hypo = reservoir.all['hypo3']

maxidx = np.argmax(hypo, axis=0)

crossings = maxidx[0:-1] - maxidx[1:]
cross_idx = np.nonzero(crossings)
dom_times = cross_idx[0][1:] - cross_idx[0][0:-1]

dom_times = dom_times / np.mean(dom_times)

# as the signals are alternating, even indices belong to
# one, odd indices to the dominance periods of the other signal
dom_sg1 = dom_times[::2]
dom_sg2 = dom_times[1::2]

# delete all that are shorter than a specified value (maybe 25 timesteps),
# they are not perceivalbe and unsignitifcant
cutval = 25
del_idx_1 = np.nonzero(dom_sg1 <= cutval)[0]
del_idx_2 = np.nonzero(dom_sg2 <= cutval)[0]
dom_sg1 = np.delete(dom_sg1, del_idx_1)
dom_sg2 = np.delete(dom_sg2, del_idx_2)

dom_times = np.sort(dom_times)
# dismiss rapid changes as thex might not be conscious
dom_times = dom_times[40:]

dom_times = dom_times / np.mean(dom_times)

# normalize

domtimes = []
domtimes.append(dom_sg1)
domtimes.append(dom_sg2)
with open('FigureObject.fig_domtimes.pickle', "wb") as fp:
    pickle.dump(domtimes, fp, protocol=2)

plt.show()
