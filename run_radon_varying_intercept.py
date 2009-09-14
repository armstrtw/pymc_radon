import pymc
import radon_varying_intercept
from pylab import hist, show

M = pymc.MCMC(radon_varying_intercept)
M.sample(iter=10e3, burn=3e3, thin=5)

fit = M.stats()
for k in fit.keys():
     print(k,fit[k]['mean'])
