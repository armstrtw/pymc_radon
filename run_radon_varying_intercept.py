import pymc
import radon_varying_intercept
from pylab import hist, show
from pymc import Matplot

M = pymc.MCMC(radon_varying_intercept)
M.sample(iter=50e3, burn=10e3, thin=5)

fit = M.stats()
for k in fit.keys():
     print(k,fit[k]['mean'])

Matplot.plot(M)
