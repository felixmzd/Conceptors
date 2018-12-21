from matplotlib.pyplot import *
import pickle
from matplotlib2tikz import save as tikz_save

data = pickle.load(open('../FigureObject.fig_loading.pickle', 'rb'))

allDriverPL = data[0]
allRecallPL = data[1]
NRMSE = data[2]

for i in range(2):
    subplot(2, 1, (i+1))
    # driver and recall
    ylim([-1.1,1.1])
    text(2, -1, 'NRMSE: {0}'.format(round(NRMSE[i],4)), bbox=dict(facecolor='white', alpha=1))
    plot(allDriverPL[i,:], color='gray', linewidth=4.0, label = 'Driver')
    plot(allRecallPL[i,:], color='black', label='Recall')
    legend()
    title('Original driver and recalled signal for Sine {0}'.format(i+1))

# tikz_save('figure_tikz/fig_loading.tex', figureheight='3cm', figurewidth='12cm')


show()