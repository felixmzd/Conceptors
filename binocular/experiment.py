import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sacred import Experiment
import scipy.stats as stats
from binocular import pattern_functions
from binocular.reservoir_binocular import ReservoirBinocular
from binocular import utils
from sacred.observers import FileStorageObserver
import tempfile
from pathlib import Path

mpl.rc("savefig", dpi=500)
PLOT_EXT = "png"

ex = Experiment("binocular")
ex.observers.append(FileStorageObserver.create(basedir="runs"))



@ex.config
def config():
    N = 100
    M = 700
    NetSR = 1.4
    bias_scale = 0.2
    aperture = 15
    inp_scale = 1.0
    t_run = 5000
    t_learn = 600
    t_learn_conceptor = 2000
    t_washout = 200
    TyA_wout = 1
    TyA_wload = 0.01
    TyA_G = 0.1
    c_adapt_rate = 0.5
    pattern_idxs = [50, 53]
    SNR = 1.0
    trust_smooth_rate = 0.99
    trust_adapt_steepness12 = 8
    trust_adapt_steepness23 = 8
    drift = 0.0001
    hypo_adapt_rate = 0.002

    seed = 1
    artifact_dir = Path(tempfile.mkdtemp())


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
    SNR,
    trust_smooth_rate,
    trust_adapt_steepness12,
    trust_adapt_steepness23,
    drift,
    hypo_adapt_rate,
):
    reservoir = ReservoirBinocular.init_random(
        N=N,
        M=M,
        NetSR=NetSR,
        bias_scale=bias_scale,
        alpha=aperture,
        inp_scale=inp_scale,
        SNR=SNR,
        TyA_wout=TyA_wout,
        TyA_wload=TyA_wload,
        c_adapt_rate=c_adapt_rate,
        trust_smooth_rate=trust_smooth_rate,
        trust_adapt_steepness12=trust_adapt_steepness12,
        trust_adapt_steepness23=trust_adapt_steepness23,
        drift=drift,
        hypo_adapt_rate=hypo_adapt_rate,
    )
    reservoir.fit(
        patterns, t_learn=t_learn, t_learnc=t_learn_conceptor, t_wash=t_washout
    )
    Y_recalls = reservoir.recall()
    reservoir.binocular(t_run=t_run)
    return reservoir, Y_recalls


@ex.capture
def savefig(fig, filename, artifact_dir, _run):
    fig.tight_layout()
    filepath = artifact_dir / filename
    fig.savefig(filepath, bbox_inches="tight")
    _run.add_artifact(filename=filepath, name=filename)


def plot_loading(patterns, Y_recalls):
    allDriverPL, allRecallPL, NRMSE = utils.plot_interpolate_1d(
        patterns, Y_recalls, plotrange=100
    )

    fig, ax = plt.subplots(nrows=2)
    for i in range(2):
        # driver and recall
        ax[i].set(
            ylim=[-1.1, 1.1],
            title=f"Original driver and recalled signal for Sine {i + 1}",
        )
        ax[i].text(
            2,
            -1,
            "NRMSE: {0}".format(round(NRMSE[i], 4)),
            bbox=dict(facecolor="white", alpha=1),
        )
        ax[i].plot(allDriverPL[i, :], color="gray", linewidth=4.0, label="Driver")
        ax[i].plot(allRecallPL[i, :], color="black", label="Recall")
    fig.legend(
        *ax[-1].get_legend_handles_labels(),
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
    )
    savefig(fig, f"loading.{PLOT_EXT}")


def plot_real_input(reservoir):
    plot_range = slice(100, 200)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(
        reservoir.all["real_input_w-o_noise"][:, plot_range].T,
        color="dimgray",
        linewidth=2,
        label="clean signal",
    )
    ax.plot(
        reservoir.all["real_input"][:, plot_range].T,
        color="black",
        linewidth=1,
        label="signal + noise",
    )
    ax.set(
        xlabel="simulated timesteps",
        ylabel="signal",
        ylim=[-3.5, 3.5],
        xticks=np.arange(0, 101, 10),
        yticks=np.arange(-4, 5, 2.0),
    )
    ax.tick_params(direction="in")
    fig.legend(
        bbox_to_anchor=(0.985, 1.05),
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    savefig(fig, f"real_input.{PLOT_EXT}")


def plot_hypothesis(reservoir):
    legend_kwargs = dict(
        bbox_to_anchor=(1.25, 0.5),
        borderaxespad=0.0,
        edgecolor="white",
        loc="center right",
        framealpha=0,
    )
    label_signal_1 = "Sine 1"
    label_signal_2 = "Sine 2"
    steps_to_plot = 3000
    linespec_1 = "k:"
    linespec_2 = "k-"

    fig, ax = plt.subplots(4, sharex=True)
    ax[0].set_ylim([-0.1, 1.1])
    ax[0].plot(
        reservoir.all["hypo3"][0][:steps_to_plot].T,
        linespec_1,
        linewidth=1.3,
        label=label_signal_1,
    )
    ax[0].plot(
        reservoir.all["hypo3"][1][:steps_to_plot].T,
        linespec_2,
        linewidth=1.3,
        label=label_signal_2,
    )
    ax[0].legend(**legend_kwargs)
    ax[0].set(title="Level 3 Hypotheses")

    ax[1].set_ylim([-0.1, 1.1])
    ax[1].plot(
        reservoir.all["hypo2"][0][:steps_to_plot].T,
        linespec_1,
        linewidth=1.3,
        label=label_signal_1,
    )
    ax[1].plot(
        reservoir.all["hypo2"][1][:steps_to_plot].T,
        linespec_2,
        linewidth=1.3,
        label=label_signal_2,
    )
    ax[1].legend(**legend_kwargs)
    ax[1].set(title="Level 2 Hypotheses")

    ax[2].set_ylim([-0.1, 1.1])
    ax[2].plot(
        reservoir.all["hypo1"][0][:steps_to_plot].T,
        linespec_1,
        linewidth=1.3,
        label=label_signal_1,
    )
    ax[2].plot(
        reservoir.all["hypo1"][1][:steps_to_plot].T,
        linespec_2,
        linewidth=1.3,
        label=label_signal_2,
    )
    ax[2].legend(**legend_kwargs)
    ax[2].set(title="Level 1 Hypotheses")

    ax[3].set_ylim([-0.1, 1.1])
    ax[3].plot(reservoir.all["trusts12"][0][:steps_to_plot].T, linespec_1, label="Trust 12")
    ax[3].plot(reservoir.all["trusts23"][0][:steps_to_plot].T, linespec_2, label="Trust 23")
    ax[3].legend(**legend_kwargs)
    ax[3].set(title="Trusts", xlabel="simulated timsteps")

    savefig(fig, f"hypotheses.{PLOT_EXT}")


def plot_predictions(reservoir):
    y3 = reservoir.all["y3"]

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(np.arange(370, 570, 1), y3[:, 370:570].T, "k-", linewidth=1.3)
    ax[0].set(xlim=[370, 570], title="Transition from Sine 1 to Sine 2")

    ax[1].plot(np.arange(630, 830, 1), y3[:, 630:830].T, "k-", linewidth=1.3)
    ax[1].set(xlim=[630, 830], title="Transition from Sine 1 to Sine 2")

    ax[2].plot(np.arange(1050, 1250, 1), y3[:, 1050:1250].T, "k-", linewidth=1.3)
    ax[2].set(title="Transition from Sine 1 to Sine 2")

    savefig(fig, f"predictions.{PLOT_EXT}")


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
    plt.title("discrepancies")
    plt.legend()

    plt.figure()
    plt.plot(reservoir.all["unexplained1"][:, plot_range].T, "b", label="level1")
    plt.plot(reservoir.all["unexplained2"][:, plot_range].T, "g", label="level2")
    plt.plot(reservoir.all["unexplained3"][:, plot_range].T, "y", label="level3")
    plt.title("unexplained")
    plt.legend()

    fig, ax = plt.subplots()
    ax.plot(reservoir.all["real_input"].T[plot_range], label="real input")
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


def plot_dominance_times(reservoir):
    bins = 15
    dom_sg1, dom_sg2 = compute_dominance_times(reservoir)
    ylabel = "Density"

    linewidth = 2
    alpha = 0.8
    fig, ax = plt.subplots(2)

    n, bins, patches = ax[0].hist(
        dom_sg1,
        bins=bins,
        density=True,
        facecolor="#D0D1D3",
        edgecolor="black",
        alpha=1,
    )

    # fit gamma distribution
    params = stats.gamma.fit(dom_sg1, floc=0)
    print("gamma params signal 1", params)

    x = np.linspace(0, dom_sg1.max(), 100)
    fit_pdf = stats.gamma.pdf(x, *params)
    ax[0].plot(x, fit_pdf, "k-", lw=linewidth, alpha=alpha)
    ax[0].set(
        xlabel=r"Normalized dominance duration for sine wave 1 ($t_1 / \overline{t_1}$)",
        ylabel=ylabel,
    )
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["top"].set_visible(False)
    ax[0].tick_params(direction="in")
    ax[1].hist(
        dom_sg2,
        bins=bins,
        density=True,
        facecolor="#D0D1D3",
        edgecolor="black",
        alpha=1,
    )

    # fit gamma distribution
    params = stats.gamma.fit(dom_sg2, floc=0)
    print("gamma params signal 2", params)

    x = np.linspace(0, dom_sg2.max(), 100)
    fit_pdf = stats.gamma.pdf(x, *params)
    ax[1].plot(x, fit_pdf, "k-", lw=linewidth, alpha=alpha)
    ax[1].set(
        xlabel=r"Normalized dominance duration for sine wave 2 ($t_2 / \overline{t_2}$)",
        ylabel=ylabel,
    )
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["top"].set_visible(False)
    ax[1].tick_params(direction="in")

    savefig(fig, f"domtimes.{PLOT_EXT}")


@ex.automain
def run(pattern_idxs, artifact_dir):
    patterns = [pattern_functions.patterns[p] for p in pattern_idxs]

    reservoir, y_recalls = run_reservoir(patterns)

    plot_loading(patterns, y_recalls)
    plot_real_input(reservoir)
    plot_hypothesis(reservoir)
    plot_predictions(reservoir)
    make_plots(reservoir)

    plot_dominance_times(reservoir)

    shutil.rmtree(artifact_dir)
