from matplotlib.pyplot import *
import pickle
from matplotlib2tikz import save as tikz_save


data = pickle.load(open('../FigureObject.fig_result.pickle', 'rb'))

figure()
ax = subplot(4,1,1)
ylim([-0.1,1.1])
leg = 'Sine {0}'.format(1)
plot(data[0][0][0:3000].T, 'k-', linewidth= 1.3, label = leg )
leg = 'Sine {0}'.format(2)
plot(data[0][1][0:3000].T, 'k--',linewidth= 1.3, label = leg )
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
title('Level 3 Hypotheses')

subplot(4,1,2, sharex=ax)
ylim([-0.1,1.1])
leg = 'Sine {0}'.format(1)
plot(data[1][0][0:3000].T,'k-', linewidth= 1.3, label = leg )
leg = 'Sine {0}'.format(2)
plot(data[1][1][0:3000].T,'k--',linewidth= 1.3, label = leg )
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
title('Level 2 Hypotheses')


subplot(4,1,3, sharex=ax)
ylim([-0.1,1.1])
leg = 'Sine {0}'.format(1)
plot(data[2][0][0:3000].T, 'k--', linewidth= 1.3, label = leg )
leg = 'Sine {0}'.format(2)
plot(data[2][1][0:3000].T, 'k-',linewidth= 1.3, label = leg )
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
title('Level 1 Hypotheses')


subplot(4,1,4, sharex=ax)
ylim([-0.1,1.1])
leg = 'Trust {0}'.format(12)
plot(data[3][0][0:3000].T,  'k-',  label = leg )
leg = 'Trust {0}'.format(23)
plot(data[4][0][0:3000].T, 'k--', label = leg )
legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
title('Trust variables')


# tikz_save('figure_tikz/fig_result.tex', figureheight='4cm', figurewidth='12cm')


show()
