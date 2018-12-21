from matplotlib.pyplot import *
import pickle
from matplotlib2tikz import save as tikz_save

data = pickle.load(open('../FigureObject.fig_predictions.pickle', 'rb'))
y3 = data
#figure()
#plot(A.all['y3'][:,0:2000].T, 'k-', linewidth=1.3)

figure()

subplot(3, 1, 1)
plot(np.arange(370, 570, 1),y3[:,370:570].T, 'k-', linewidth=1.3)
xlim([370, 570])
title('Transition from Sine 1 to Sine 2')

subplot(3, 1, 2)
plot(np.arange(630, 830, 1),y3[:,630:830].T, 'k-', linewidth=1.3)
xlim([630, 830])
title('Transition from Sine 1 to Sine 2')


subplot(3, 1, 3)
plot(np.arange(1050, 1250, 1),y3[:,1050:1250].T, 'k-', linewidth=1.3)
title('Transition from Sine 1 to Sine 2')

# tikz_save('figure_tikz/fig_predictions.tex', figureheight='3cm', figurewidth='12cm')


show()