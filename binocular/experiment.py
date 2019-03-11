import pickle
import numpy as np
import matplotlib.pyplot as plt
from sacred import Experiment
from binocular import pattern_functions
from binocular.reservoir_binocular import ReservoirBinocular
from binocular import utils

ex = Experiment("binocular")


@ex.config
def config():
    N = 100
    M = 700
    NetSR = 1.4
    bias_scale = 0.2
    aperture = 10
    inp_scale = 1.2
    t_run = 5000
    t_learn = 600
    t_learn_conceptor = 2000
    t_washout = 200
    TyA_wout = 1
    TyA_wload = 0.01
    c_adapt_rate = 0.5
    pattern_idxs = [50, 53]
    seed = 1


@ex.capture
def run_reservoir(
    patterns,
    N,
    M,
    NetSR,
    bias_scale,
    aperture,
    inp_scale,
    t_run,
    t_learn,
    t_learn_conceptor,
    t_washout,
    TyA_wout,
    TyA_wload,
    c_adapt_rate,
):
    reservoir = ReservoirBinocular.init_random(
        N=N,
        M=M,
        NetSR=NetSR,
        bias_scale=bias_scale,
        alpha=aperture,
        inp_scale=inp_scale,
    )
    reservoir.fit(
        patterns,
        t_learn=t_learn,
        t_learnc=t_learn_conceptor,
        t_wash=t_washout,
        TyA_wout=TyA_wout,
        TyA_wload=TyA_wload,
        c_adapt_rate=c_adapt_rate,
    )
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
        plt.plot(allDriverPL[i, :], color="gray", linewidth=4.0, label="Driver")
        plt.plot(allRecallPL[i, :], color="black", label="Recall")
        plt.legend()
        plt.title("Original driver and recalled signal for Sine {0}".format(i + 1))

    with open("FigureObject.fig_loading.pickle", "wb") as fp:
        pickle.dump(loading, fp)


def save_driver_input(reservoir):
    driver_input = [
        reservoir.all["driver_sine1"][:, 0:100].T,
        reservoir.all["driver_sine2"][:, 0:100].T,
        reservoir.all["driver_noise"][:, 0:100].T,
        reservoir.all["driver"][:, 0:100].T,
    ]
    with open("FigureObject.fig_rawinput.pickle", "wb") as fp:
        pickle.dump(driver_input, fp, protocol=2)


def save_real_input(reservoir):
    real_input = [
        reservoir.all["real_input_w-o_noise"][:, 0:100].T,
        reservoir.all["real_input"][:, 0:100].T,
    ]
    with open("FigureObject.fig_realinput.pickle", "wb") as fp:
        pickle.dump(real_input, fp, protocol=2)


def save_result(reservoir):
    res = [
        reservoir.all["hypo3"],
        reservoir.all["hypo2"],
        reservoir.all["hypo1"],
        reservoir.all["trusts12"],
        reservoir.all["trusts23"],
    ]
    with open("FigureObject.fig_result.pickle", "wb") as fp:
        pickle.dump(res, fp, protocol=2)

    with open("../tests/data/FigureObject.fig_result.pickle", "wb") as fp:
        pickle.dump(res, fp)


def save_predictions(reservoir):
    with open("FigureObject.fig_predictions.pickle", "wb") as fp:
        pickle.dump(reservoir.all["y3"], fp, protocol=2)


def make_plots(reservoir):
    plot_range = slice(0, 50)

    # recall
    plt.figure()
    plt.plot(reservoir.all["y3"][:, plot_range].T, color="black", label="output")
    plt.plot(
        reservoir.all["y3"][:, plot_range].T - reservoir.all["driver"][:, plot_range].T,
        color="gray",
        linewidth=4.0,
        label="difference",
    )
    plt.plot(
        reservoir.all["driver"][:, plot_range].T,
        color="blue",
        linewidth=4.0,
        label="driver",
    )
    plt.title("output and difference ")
    plt.legend()
    # discrepancys on every level
    plt.figure()
    plt.plot(reservoir.all["trusts1"].T, "b", label="level1")
    plt.plot(reservoir.all["trusts2"].T, "g", label="level2")
    plt.plot(reservoir.all["trusts3"].T, "y", label="level3")
    plt.title("trusts")
    plt.figure()
    plt.plot(reservoir.all["unexplained1"][:, plot_range].T, "b", label="level1")
    plt.plot(reservoir.all["unexplained2"][:, plot_range].T, "g", label="level2")
    plt.plot(reservoir.all["unexplained3"][:, plot_range].T, "y", label="level3")
    # plot(reservoir.all['driver'][:, 0:500].T, 'g')
    plt.title("unexplained")
    plt.legend()

    fig, ax = plt.subplots()
    # ax.plot(reservoir.all["real_input"].T[1500:1600], label="real input")
    # plot_range = slice(1500, 1600)
    ax.plot(
        reservoir.all["real_input_w-o_noise"].T[plot_range], label="input without noise"
    )
    ax.plot(reservoir.all["y3"].T[plot_range], label="prediction")
    fig.legend()
    ax.set(title="inputs")


def compute_dominance_times(reservoir):
    # Compute dominance times per signal.
    hypo = reservoir.all["hypo3"]
    max_idx = np.argmax(hypo, axis=0)
    crossings = max_idx[:-1] - max_idx[1:]
    cross_idx = np.flatnonzero(crossings)
    dominance_times = cross_idx[1:] - cross_idx[:-1]
    # As the signals are alternating, even indices belong to
    # one, odd indices to the dominance periods of the other signal.
    dominance_times_signal_1 = dominance_times[::2]
    dominance_times_signal_2 = dominance_times[1::2]

    # Normalize.
    dominance_times_signal_1 = (
        dominance_times_signal_1 / dominance_times_signal_1.mean()
    )
    dominance_times_signal_2 = (
        dominance_times_signal_2 / dominance_times_signal_2.mean()
    )
    return dominance_times_signal_1, dominance_times_signal_2


def save_dominance_times(domtimes):
    with open("FigureObject.fig_domtimes.pickle", "wb") as fp:
        pickle.dump(domtimes, fp, protocol=2)


@ex.automain
def run(pattern_idxs):
    patterns = [pattern_functions.patterns[p] for p in pattern_idxs]

    reservoir, Y_recalls = run_reservoir(patterns)

    plot_and_save_loading(reservoir, patterns, Y_recalls)
    save_driver_input(reservoir)
    save_real_input(reservoir)
    save_result(reservoir)
    save_predictions(reservoir)
    make_plots(reservoir)

    dominance_times = compute_dominance_times(reservoir)
    save_dominance_times(dominance_times)

    plt.show()
