import matplotlib.pyplot as plt
import pickle

# from matplotlib2tikz import save as tikz_save
import scipy.stats as stats
import numpy as np

data = pickle.load(open("../FigureObject.fig_domtimes.pickle", "rb"))
bins = 15
dom_sg1 = data[0]
dom_sg2 = data[1]

fig, ax = plt.subplots(2)

n, bins, patches = ax[0].hist(
    dom_sg1, bins=bins, density=True, facecolor="gray", edgecolor="black", alpha=0.75
)

# fit gamma distribution
params = stats.gamma.fit(dom_sg1, floc=0)
print(params)

x = np.linspace(0, dom_sg1.max(), 100)
fit_pdf = stats.gamma.pdf(x, *params)
ax[0].plot(x, fit_pdf, "k-", lw=5, alpha=0.6)
ax[0].set(
    xlabel="Dominance duration in simulated timesteps",
    title="Distribution of dominance times for Sine 1",
)

n, bins, patches = ax[1].hist(
    dom_sg2, bins=bins, density=True, facecolor="gray", edgecolor="black", alpha=0.75
)

# fit gamma distribution
params = stats.gamma.fit(dom_sg2, floc=0)
print(params)

x = np.linspace(0, dom_sg2.max(), 100)
fit_pdf = stats.gamma.pdf(x, *params)
ax[1].plot(x, fit_pdf, "k-", lw=5, alpha=0.6)
ax[1].set(
    xlabel="Dominance duration in simulated timesteps",
    title="Distribution of dominance times for Sine 2",
)

# calculate power of the signals


# tikz_save('figure_tikz/fig_domtimes.tex', figureheight='5cm', figurewidth='12cm',  draw_rectangles=True)


plt.show()
