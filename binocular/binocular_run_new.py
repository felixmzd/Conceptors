"""Run this script to create pickle files for plotting."""
from binocular.reservoir_binocular_new import ReservoirBinocular
from binocular import pattern_functions
from binocular import utils
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
from matplotlib.pyplot import *

np.random.seed(1)


def main():
    # pattern_idxs = [50, 27]
    # pattern_idxs = [50, 20]
    pattern_idxs = [50, 53]
    patterns = [pattern_functions.patterns[p] for p in pattern_idxs]

    reservoir, Y_recalls = run_reservoir(patterns, t_run=5000)

    plot_and_save_loading(reservoir, patterns, Y_recalls)
    # save_driver_input(reservoir)
    # save_real_input(reservoir)
    # save_result(reservoir)
    # save_predictions(reservoir)
    make_plots(reservoir)

    dominance_times = compute_dominance_times(reservoir)
    save_dominance_times(dominance_times)

    plt.show()


def run_reservoir(patterns, t_run):
    reservoir = ReservoirBinocular.init_random(N=100,M=700,NetSR=1.4,bias_scale=0.2,aperture=10,inp_scale=1.2,t_learn=600,t_learn_conceptor=2000,)
    reservoir.fit(patterns)
    Y_recalls = reservoir.recall()
    reservoir.binocular(t_run=t_run)
    return reservoir, Y_recalls


def plot_and_save_loading(reservoir, patterns, Y_recalls):
    allDriverPL, allRecallPL, NRMSE = utils.plot_interpolate_1d(
        patterns, Y_recalls, plotrange=100
    )
    loading = [allDriverPL, allRecallPL, NRMSE]
    for i in range(len(patterns)):
        plt.subplot(len(patterns), 1, (i + 1))
        # driver and recall
        plt.ylim([-1.1, 1.1])
        plt.text(
            2,
            -1,
            "NRMSE: {0}".format(round(NRMSE[i], 4)),
            bbox=dict(facecolor="white", alpha=1),
        )
        plt.plot(allDriverPL[i, :], color="gray", linewidth=6.0, label="Driver")
        plt.plot(allRecallPL[i, :], color="black", label="Recall")
        plt.legend()
        plt.title("Original driver and recalled signal for Sine {0}".format(i + 1))

    with open("FigureObject.fig_loading.pickle", "wb") as fp:
        pickle.dump(loading, fp)

#
# def save_driver_input(reservoir):
#     driver_input = [
#         reservoir.all["driver_sine1"][:, 0:100].T,
#         reservoir.all["driver_sine2"][:, 0:100].T,
#         reservoir.all["driver_noise"][:, 0:100].T,
#         reservoir.all["driver"][:, 0:100].T,
#     ]
#     with open("FigureObject.fig_rawinput.pickle", "wb") as fp:
#         pickle.dump(driver_input, fp, protocol=2)
#
#
# def save_real_input(reservoir):
#     real_input = [
#         reservoir.all["real_input_w-o_noise"][:, 0:100].T,
#         reservoir.all["real_input"][:, 0:100].T,
#     ]
#     with open("FigureObject.fig_realinput.pickle", "wb") as fp:
#         pickle.dump(real_input, fp, protocol=2)


def save_result(reservoir):
    res = [
        reservoir.all["hypo3"],
        reservoir.all["hypo2"],
        reservoir.all["hypo1"],
        reservoir.all["trusts12"],
        reservoir.all["trusts23"],
    ]
    with open("FigureObject.fig_result.pickle", "wb") as fp:
        pickle.dump(res, fp)
    #
    # with open("../tests/data/FigureObject.fig_result_new.pickle", "wb") as fp:
    #     pickle.dump(res, fp)


def save_predictions(reservoir):
    with open("FigureObject.fig_predictions.pickle", "wb") as fp:
        pickle.dump(reservoir.all["y3"], fp, protocol=2)


def make_plots(reservoir):
    # recall
    plot_range = slice(3960, 4000)
    plot_range = slice(0, 50)
    plt.figure()
    plt.plot(reservoir.history["y"][plot_range, 3], color="black", label="output")
    # plt.plot(
    #     reservoir.all["y3"][:, 3960:4000].T - reservoir.all["driver"][:, 3960:4000].T,
    #     color="gray",
    #     linewidth=4.0,
    #     label="difference",
    # )
    plt.plot(
        reservoir.history["y"][plot_range, 0],
        color="blue",
        linewidth=4.0,
        label="driver",
    )
    plt.title("output and difference ")
    plt.legend()
    # discrepancys on every level
    plt.figure()
    plt.plot(reservoir.history["trusts"][0], "b", label="level1")
    plt.plot(reservoir.history["trusts"][1], "g", label="level2")
    # plt.plot(reservoir.all["trusts3"].T, "y", label="level3")
    plt.title("trusts")
    # plt.figure()
    # plt.plot(reservoir.all["unexplained1"][:, 3960:4000].T, "b", label="level1")
    # plt.plot(reservoir.all["unexplained2"][:, 3960:4000].T, "g", label="level2")
    # plt.plot(reservoir.all["unexplained3"][:, 3960:4000].T, "y", label="level3")
    # # plot(reservoir.all['driver'][:, 0:500].T, 'g')
    # plt.title("unexplained")
    # plt.legend()
    #
    # fig, ax = plt.subplots()
    # # ax.plot(reservoir.all["real_input"].T[1500:1600], label="real input")
    # ax.plot(
    #     reservoir.all["real_input_w-o_noise"].T[1500:1600], label="input without noise"
    # )
    # ax.plot(reservoir.all["y3"].T[1500:1600], label="prediction")
    # fig.legend()
    # ax.set(title="inputs")


def compute_dominance_times(reservoir):
    # Compute dominance times per signal.
    hypo = reservoir.history["hypotheses"][:, -1]
    max_idx = np.argmax(hypo, axis=0)
    crossings = max_idx[:-1] - max_idx[1:]
    cross_idx = np.flatnonzero(crossings)
    dominance_times = cross_idx[1:] - cross_idx[:-1]
    # As the signals are alternating, even indices belong to
    # one, odd indices to the dominance periods of the other signal.
    dominance_times_signal_1 = dominance_times[::2]
    dominance_times_signal_2 = dominance_times[1::2]

    # # delete all that are shorter than a specified value (maybe 25 timesteps),
    # # they are not perceivalbe and unsignitifcant
    # cutval = 0
    # del_idx_1 = np.nonzero(dom_sg1 <= cutval)[0]
    # del_idx_2 = np.nonzero(dom_sg2 <= cutval)[0]
    # dom_sg1 = np.delete(dom_sg1, del_idx_1)
    # dom_sg2 = np.delete(dom_sg2, del_idx_2)

    # Normalize.
    dominance_times_signal_1 = (
        dominance_times_signal_1 / dominance_times_signal_1.mean()
    )
    dominance_times_signal_2 = (
        dominance_times_signal_2 / dominance_times_signal_2.mean()
    )
    return [dominance_times_signal_1, dominance_times_signal_2]


def save_dominance_times(domtimes):
    with open("FigureObject.fig_domtimes.pickle", "wb") as fp:
        pickle.dump(domtimes, fp, protocol=2)


if __name__ == "__main__":
    main()
