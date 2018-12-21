from matplotlib.pyplot import *
import pickle
from matplotlib2tikz import save as tikz_save


data = pickle.load(open('../FigureObject.fig_realinput.pickle', 'rb'))
ylim([-4.5,4.5])
yticks(np.arange(-4, 5, 2.0))
plot(data[0], color='gray', linewidth=4.0)
plot(data[1], color='black')

# tikz_save('figure_tikz/fig_realinput.tex', figureheight='3cm', figurewidth='10cm')
show()
