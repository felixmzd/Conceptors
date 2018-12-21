from matplotlib.pyplot import *
import pickle
from matplotlib2tikz import save as tikz_save


data = pickle.load(open('figure_sourcefiles/FigureObject.fig_rfc.pickle', 'rb'))

figure()

allDriverPL = data[0]
allRecallPL = data[1]
C = data[2]
cColls = data[3]

# plot adaptation of c's
for i in range(4):
    #c sprectrum
    subplot(4, 3, (i+1)*3 -2)
    plot(np.flipud(np.sort(C[i])), color='black', linewidth=2.0)
    if((i+1)*3 -2 == 1):
        title('c spectrum')

    subplot(4, 3, (i+1)*3 -1)
    # driver and recall
    ylim([-1.1,1.1])
    #text(1, 0, round(NRMSE[i],4), bbox=dict(facecolor='white', alpha=1))
    plot(allDriverPL[i,:], color='gray', linewidth=4.0)
    plot(allRecallPL[i,:], color='black')
    if((i+1)*3 -1 == 2):
        title('driver and recall')

    # c adaptation
    subplot(4, 3, (i+1)*3)
    num_plots = 40;
    colormap = cm.gray
    gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
    indices = np.linspace(0,1999,100)
    indices = indices.astype(int)
    print(indices)
    plot(cColls[i,1:num_plots,indices])
    ylim([-0.1,1.1])
    if((i+1)*3 == 3):
        title('c adaptation')

tikz_save('figure_tikz/fig_rfc.tex', figureheight='4cm', figurewidth='4cm')

show()