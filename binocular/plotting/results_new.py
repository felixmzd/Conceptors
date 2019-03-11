import matplotlib.pyplot as plt
import pickle
from matplotlib2tikz import save as tikz_save

data = pickle.load(open("../FigureObject.fig_result.pickle", "rb"))

label_signal_1 = "Sine 1"
label_signal_2 = "Sine 2"
steps_to_plot = 3000

fig, ax = plt.subplots(4, sharex=True)
ax[0].set_ylim([-0.1, 1.1])
ax[0].plot(data[0][:steps_to_plot], "k-", linewidth=1.3, label=label_signal_1)
ax[0].plot(data[0][:steps_to_plot], "k--", linewidth=1.3, label=label_signal_2)
ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax[0].set_title("Level 3 Hypotheses")

ax[1].set_ylim([-0.1, 1.1])
ax[1].plot(data[1][:steps_to_plot], "k-", linewidth=1.3, label=label_signal_1)
ax[1].plot(data[1][:steps_to_plot], "k--", linewidth=1.3, label=label_signal_2)
ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax[1].set_title("Level 2 Hypotheses")

ax[2].set_ylim([-0.1, 1.1])
ax[2].plot(data[2][:steps_to_plot], "k--", linewidth=1.3, label=label_signal_1)
ax[2].plot(data[2][:steps_to_plot], "k-", linewidth=1.3, label=label_signal_2)
ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax[2].set_title("Level 1 Hypotheses")

ax[3].set_ylim([-0.1, 1.1])
ax[3].plot(data[3][:steps_to_plot], "k-", label="Trust 12")
ax[3].plot(data[4][:steps_to_plot], "k--", label="Trust 23")
ax[3].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
ax[3].set_title("Trust variables")

# tikz_save('figure_tikz/fig_result.tex', figureheight='4cm', figurewidth='12cm')


plt.show()
