import pymc
import radon_inv_wishart
from pylab import hist, show
from pymc import Matplot
from timeit import Timer

M = pymc.MCMC(radon_inv_wishart)
M.sample(iter=2e3, burn=1e3, thin=5)

fit = M.stats()
print('mu',fit['mu']['mean'])
print('xi',fit['xi']['mean'])
print('sigma_y',fit['sigma_y']['mean'])
print('tau_y',fit['tau_y']['mean'])
for i in range(85):
     print(fit['B_raw_%i' % i]['mean'])

Matplot.plot(M)
