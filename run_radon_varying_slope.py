import pymc
import radon_varying_slope
from pylab import hist, show

M = pymc.MCMC(radon_varying_slope)
M.sample(iter=50e3, burn=10e3, thin=10)

fit = M.stats()
for k in fit.keys():
     print(k,fit[k]['mean'])
