from matplotlib.pyplot import *
import pickle
# from matplotlib2tikz import save as tikz_save
import scipy.stats as stats


data = pickle.load(open('../FigureObject.fig_domtimes.pickle', 'rb'))
dom_sg1 = data[0]
dom_sg2 = data[1]


figure()
subplot(2,1,1)
n, bins, patches = hist(dom_sg1, 15, normed=True, facecolor='gray', alpha=0.75)

#fit gamma distribution
params=stats.gamma.fit(dom_sg1, floc=0)
print(params)

x = np.linspace(0, 800, 100)
fit_pdf = stats.gamma.pdf(x, *params)
plot(x, fit_pdf, 'k-', lw=5, alpha=0.6)
xlabel('Dominance duration in simulated timesteps')
title('Distribution of dominance times for Sine 1')


subplot(2,1,2)
n, bins, patches = hist(dom_sg2, 15, normed=True, facecolor='gray', alpha=0.75)

#fit gamma distribution
params=stats.gamma.fit(dom_sg2, floc=0)
print(params)



x = np.linspace(0, 800, 100)
fit_pdf = stats.gamma.pdf(x, *params)
plot(x, fit_pdf, 'k-', lw=5, alpha=0.6)
xlabel('Dominance duration in simulated timesteps')
title('Distribution of dominance times for Sine 2')

# calculate power of the signals



# tikz_save('figure_tikz/fig_domtimes.tex', figureheight='5cm', figurewidth='12cm',  draw_rectangles=True)


show()